"""
Auto-correlation fingerprint for fabric pattern matching.
Exploits the cyclic nature of fabric patterns directly.
Each pattern's repeat structure is a unique fingerprint.

Fingerprint = set of repeat vectors (direction + period) extracted
from 2D auto-correlation peaks, plus motif-energy profile.
"""
import cv2
import numpy as np
from scipy.ndimage import maximum_filter


def extract_acorr_fingerprint(gray, size=256):
    """
    Extract repeat-pattern fingerprint from auto-correlation.

    Returns dict with:
      - peaks: list of (dy, dx, strength) for top N secondary peaks
      - motif_w, motif_h: estimated motif size
      - radial_profile: 1D energy vs distance from center
      - angular_profile: 1D energy vs angle
    """
    g = cv2.resize(gray, (size, size)).astype(np.float32)
    g = (g - np.mean(g)) / (np.std(g) + 1e-8)

    # Auto-correlation via FFT
    f = np.fft.fft2(g, s=(size * 2, size * 2))
    psd = np.abs(f) ** 2
    acorr = np.fft.fftshift(np.fft.ifft2(psd).real)
    # Crop to center
    acorr = acorr[size // 2:3 * size // 2, size // 2:3 * size // 2]
    acorr = acorr / (acorr[size // 2, size // 2] + 1e-8)

    cy = cx = size // 2

    # === 1. Extract repeat peaks ===
    # Mask DC region (r < 4)
    masked = acorr.copy()
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            y, x = cy + dy, cx + dx
            if 0 <= y < size and 0 <= x < size:
                masked[y, x] = 0

    # Local maxima
    local_max = (masked == maximum_filter(masked, size=10))
    peak_thresh = max(0.08, masked.max() * 0.3)
    peaks_xy = np.argwhere(local_max & (masked > peak_thresh))

    peaks = []
    for y, x in peaks_xy:
        dy = y - cy
        dx = x - cx
        dist = np.sqrt(dy ** 2 + dx ** 2)
        if 4 < dist < size // 2:
            angle = np.arctan2(dy, dx)  # [-pi, pi]
            peaks.append({
                'dy': float(dy), 'dx': float(dx),
                'dist': float(dist),
                'angle': float(angle),
                'strength': float(masked[y, x]),
            })

    # Sort by strength, keep top 8
    peaks.sort(key=lambda p: p['strength'], reverse=True)
    top_peaks = peaks[:8]

    # === 2. Radial profile (scale-invariant texture energy) ===
    radial = np.zeros(size // 2)
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    vals = masked.ravel()
    rr_flat = rr.ravel()
    mask_rr = rr_flat < size // 2
    radial = np.bincount(rr_flat[mask_rr], weights=vals[mask_rr],
                         minlength=size // 2)
    radial = radial / (radial.sum() + 1e-8)

    # === 3. Angular profile (directionality) ===
    n_angles = 36
    angular = np.zeros(n_angles)
    angle_flat = np.arctan2((yy - cy).ravel(), (xx - cx).ravel())
    angle_bins = ((angle_flat + np.pi) / (2 * np.pi) * n_angles).astype(int)
    mask_a = (rr_flat > 4) & (rr_flat < size // 2)
    angular = np.bincount(np.clip(angle_bins[mask_a], 0, n_angles - 1),
                          weights=vals[mask_a], minlength=n_angles)
    angular = angular / (angular.sum() + 1e-8)

    # === 4. Motif size estimate ===
    has_harmonic = False
    if top_peaks:
        primary_period = top_peaks[0]['dist'] / size  # normalized [0, 0.5]
        if len(top_peaks) > 1:
            d1, d2 = top_peaks[0]['dist'], top_peaks[1]['dist']
            if d1 > 0 and abs(d2 / d1 - 1.5) < 0.3:
                has_harmonic = True
    else:
        primary_period = 0.1

    return {
        'peaks': top_peaks,  # list of {dy, dx, dist, angle, strength}
        'primary_period': primary_period,
        'has_harmonic': has_harmonic,
        'n_strong_peaks': sum(1 for p in top_peaks if p['strength'] > 0.15),
        'peak_strength_sum': sum(p['strength'] for p in top_peaks[:4]),
        'radial_profile': radial,      # (size//2,) normalized
        'angular_profile': angular,    # (36,) normalized
    }


def fingerprint_similarity(fp_p, fp_f):
    """
    Compare two acorr fingerprints with SCALE NORMALIZATION.
    Key insight: resize both so that their primary period = same size,
    then compare structure features at consistent scale.
    """
    s = {}

    # Scale factor: make photo's period match fabric's period
    pp_p = fp_p.get('primary_period', 0.1)
    pp_f = fp_f.get('primary_period', 0.1)
    if pp_p > 0.01 and pp_f > 0.01:
        scale_factor = pp_f / (pp_p + 1e-8)
    else:
        scale_factor = 1.0

    # === A. Period ratio (how much rescaling needed) ===
    # Perfect match = ratio close to 1.0 (or 0.5, 2.0 for harmonics)
    if scale_factor > 0:
        log_ratio = abs(np.log2(scale_factor))
        # Score: high if log_ratio ~ 0, also accept near-integer ratios
        nearest_int = round(log_ratio)
        if abs(log_ratio - nearest_int) < 0.2:
            s['period_ratio'] = 1.0  # harmonic match!
        else:
            s['period_ratio'] = 1.0 / (1.0 + log_ratio * 2)
    else:
        s['period_ratio'] = 0

    # === B. Peak direction vectors (scale-normalized) ===
    if fp_p['peaks'] and fp_f['peaks']:
        p_peaks = fp_p['peaks'][:4]
        f_peaks = fp_f['peaks'][:4]

        # Normalize peak distances by primary period
        p_vecs = [(p['dy'] / (pp_p * 256 + 1), p['dx'] / (pp_p * 256 + 1), p['angle'], p['strength'])
                  for p in p_peaks]
        f_vecs = [(p['dy'] / (pp_f * 256 + 1), p['dx'] / (pp_f * 256 + 1), p['angle'], p['strength'])
                  for p in f_peaks]

        # Greedy match on angle (most stable feature)
        matched = 0
        angle_diffs = []
        used_f = set()
        for pdy, pdx, pa, ps in p_vecs:
            best_da = 999
            best_j = -1
            for j, (fdy, fdx, fa, fs) in enumerate(f_vecs):
                if j in used_f:
                    continue
                # Circular angle difference
                da = abs(pa - fa)
                da = min(da, 2 * np.pi - da)
                if da < best_da and da < np.pi / 4:  # within 45 degrees
                    best_da = da
                    best_j = j
            if best_j >= 0:
                used_f.add(best_j)
                matched += 1
                angle_diffs.append(best_da)

        s['peak_angle_match'] = matched / max(len(p_peaks[:4]), 1)
        s['peak_angle_error'] = 1.0 - min(np.mean(angle_diffs) / (np.pi / 2), 1.0) if angle_diffs else 0
    else:
        s['peak_angle_match'] = 0
        s['peak_angle_error'] = 0

    # === C. Peak count (scale-invariant) ===
    dn = abs(fp_p['n_strong_peaks'] - fp_f['n_strong_peaks'])
    s['peak_count'] = 1.0 / (1.0 + dn * 0.5)

    # === D. Radial profile COMPARED AT SCALE-NORMALIZED INDICES ===
    rp_p = fp_p['radial_profile']
    rp_f = fp_f['radial_profile']
    # Stretch/compress to same length (128 bins)
    target_len = 128
    src_indices = np.linspace(0, len(rp_p) - 1, target_len)
    rp_p_resampled = np.interp(src_indices, np.arange(len(rp_p)), rp_p)
    rp_f_resampled = np.interp(src_indices, np.arange(len(rp_f)), rp_f)
    s['radial'] = max(0, np.corrcoef(rp_p_resampled, rp_f_resampled)[0, 1])

    # === E. Angular profile (rotation-invariant!) ===
    ap_p = fp_p['angular_profile']
    ap_f = fp_f['angular_profile']
    # Try all rotations, find best alignment
    best_corr = -1
    for shift in range(0, 36, 3):  # try every 30 degrees
        ap_f_shifted = np.roll(ap_f, shift)
        corr = np.corrcoef(ap_p, ap_f_shifted)[0, 1]
        best_corr = max(best_corr, corr)
    s['angular'] = max(0, best_corr)

    # === F. Harmonic structure ===
    s['harmonic'] = 1.0 if fp_p.get('has_harmonic') == fp_f.get('has_harmonic') else 0.3

    # Weighted fusion: prioritize peak angles (structure) and radial (texture scale)
    total = (s['peak_angle_match'] * 0.25 + s['peak_angle_error'] * 0.15 +
             s['period_ratio'] * 0.15 + s['peak_count'] * 0.05 +
             s['radial'] * 0.15 + s['angular'] * 0.15 +
             s['harmonic'] * 0.10)

    return max(0, total), s
