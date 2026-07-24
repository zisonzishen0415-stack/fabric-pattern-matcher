"""Fast feature extraction — all vectorized, no nested Python loops."""
import cv2
import numpy as np
from scipy.ndimage import maximum_filter


def imread_unicode(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


# ============================================================
# Compact signature for KD-tree
# ============================================================

def extract_compact_signature(img):
    """7-dim color-INVARIANT signature for robust KD-tree screening."""
    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Hue ENTROPY (not dominant hue — invariant to global color shift)
    h_hist = cv2.calcHist([hsv], [0], None, [45], [0, 180]).flatten()
    h_hist = h_hist / (h_hist.sum() + 1e-8)
    h_entropy = -np.sum(h_hist * np.log(h_hist + 1e-8)) / np.log(45)

    # Saturation & Value stats (mildly illumination-dependent but useful)
    sat_mean = np.mean(hsv[:, :, 1]) / 255.0
    sat_std = np.std(hsv[:, :, 1]) / 255.0
    val_mean = np.mean(hsv[:, :, 2]) / 255.0

    # Edge density (structure)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (128 * 128)

    # FFT mid-energy ratio (texture scale, lighting-invariant)
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(np.abs(f))
    fshift[64, 64] = 0
    yy, xx = np.mgrid[0:128, 0:128]
    rr = np.sqrt((yy - 64) ** 2 + (xx - 64) ** 2).astype(int)
    radial = np.bincount(rr.ravel()[rr.ravel() < 64],
                         weights=fshift.ravel()[rr.ravel() < 64], minlength=64)
    radial = radial / (radial.sum() + 1e-8)
    mid_energy = np.sum(radial[16:48])

    return np.array([h_entropy, sat_mean, sat_std, val_mean,
                     edge_density, mid_energy], dtype=np.float32)


# ============================================================
# Fast feature extraction (no GLCM, no shape, no acorr loops)
# ============================================================

def extract_features(img, scale=256):
    """Fast features: hue, spatial, LBP, FFT, grad. All vectorized."""
    img = cv2.resize(img, (scale, scale))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    feats = {}

    # — Hue histogram —
    h_hist = cv2.calcHist([hsv], [0], None, [90], [0, 180])
    feats['hue_hist'] = cv2.normalize(h_hist, h_hist).flatten()

    # — Saturation histogram —
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
    feats['sat_hist'] = cv2.normalize(s_hist, s_hist).flatten()

    # — Spatial hue (8×8 grid) —
    grid = 8
    ch, cw = scale // grid, scale // grid
    spatial_parts = []
    for i in range(grid):
        for j in range(grid):
            cell = hsv[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            hist = cv2.calcHist([cell], [0], None, [18], [0, 180])
            spatial_parts.append(cv2.normalize(hist, hist).flatten())
    feats['spatial_hue'] = np.concatenate(spatial_parts)

    # — FFT (vectorized) —
    size = 256
    gray256 = cv2.resize(gray, (size, size)).astype(np.float32)
    window = np.outer(np.hanning(size), np.hanning(size))
    f = np.fft.fft2(gray256 * window)
    fshift = np.fft.fftshift(np.abs(f))
    fshift[128, 128] = 0
    yy, xx = np.mgrid[0:size, 0:size]
    rr = np.sqrt((yy - 128) ** 2 + (xx - 128) ** 2).astype(int)
    radial = np.bincount(rr.ravel()[rr.ravel() < 128],
                         weights=fshift.ravel()[rr.ravel() < 128], minlength=128)
    radial = radial / (radial.sum() + 1e-8)
    n = len(radial)
    low_e = np.sum(radial[:n // 4])
    mid_e = np.sum(radial[n // 4:n // 2])
    high_e = np.sum(radial[n // 2:])
    valid_fft = radial[4:100]
    if valid_fft.sum() > 0:
        pk = np.argmax(valid_fft) + 4
        pk_freq = pk / size
        period = (1.0 / pk_freq) / size if pk_freq > 0 else 1.0
        sharpness = np.sum(radial[max(0, pk - 2):min(128, pk + 3)])
    else:
        pk_freq, period, sharpness = 0.1, 0.1, 0.01

    # Angular energy (vectorized)
    n_angles = 36
    angle_energy = np.zeros(n_angles)
    mask = (rr > 8) & (rr < 102)
    angles = np.arctan2(yy[mask] - 128, xx[mask] - 128) % np.pi
    bins = (angles / np.pi * n_angles).astype(int)
    angle_energy = np.bincount(np.clip(bins, 0, n_angles - 1),
                               weights=fshift[mask], minlength=n_angles)
    angle_energy = angle_energy / (angle_energy.sum() + 1e-8)
    entropy = -np.sum(angle_energy * np.log(angle_energy + 1e-8))
    peaks = np.sum((valid_fft[1:-1] > np.mean(valid_fft) * 1.5) &
                   (valid_fft[1:-1] > valid_fft[:-2]) &
                   (valid_fft[1:-1] > valid_fft[2:]))

    feats['fft_vec'] = np.array([
        low_e, mid_e, high_e, pk_freq, period, sharpness,
        entropy / np.log(n_angles), np.var(angle_energy), min(peaks, 10) / 10.0,
    ])

    # — LBP (vectorized) —
    gs = cv2.resize(gray, (128, 128))
    rolled = [np.roll(gs, (-di, -dj), axis=(0, 1))
              for di, dj in [(0, 1), (1, 1), (1, 0), (1, -1),
                             (0, -1), (-1, -1), (-1, 0), (-1, 1)]]
    lbp = sum((r >= gs).astype(np.uint8) * (1 << k) for k, r in enumerate(rolled))
    lbp_hist = cv2.calcHist([lbp[1:-1, 1:-1]], [0], None, [256], [0, 256])
    feats['lbp_hist'] = cv2.normalize(lbp_hist, lbp_hist).flatten()

    # — Gradient orientation —
    gx = cv2.Sobel(gs, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gs, cv2.CV_64F, 0, 1, ksize=3)
    mag_g = np.sqrt(gx ** 2 + gy ** 2)
    orient = np.arctan2(gy, gx) * 180 / np.pi % 180
    grad_hist = np.zeros(36)
    for i in range(36):
        grad_hist[i] = np.sum(mag_g[(orient >= i * 5) & (orient < (i + 1) * 5)])
    feats['grad_hist'] = grad_hist / (grad_hist.sum() + 1e-8)

    # — Color moments —
    moments = []
    for ch_idx in range(3):
        channel = hsv[:, :, ch_idx].astype(np.float32)
        mean = np.mean(channel)
        std = np.std(channel)
        skew = np.mean(((channel - mean) / (std + 1e-8)) ** 3)
        moments.extend([mean / 256.0, std / 256.0, np.clip(skew, -5, 5) / 5.0])
    feats['color_moments'] = np.array(moments)

    return feats


# ============================================================
# ORB
# ============================================================

def extract_orb(img, max_keypoints=500):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Resize to consistent 512px
    h, w = gray.shape
    target = 512
    if max(h, w) != target:
        scale = target / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))
    # Low fastThreshold = more keypoints (fabric patterns have subtle edges)
    orb = cv2.ORB_create(nfeatures=max_keypoints,
                         scaleFactor=1.2, nlevels=8,
                         fastThreshold=5)
    kp, des = orb.detectAndCompute(gray, None)
    return kp, des


def match_orb(des1, des2, kp1=None, kp2=None):
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return 0, 0, 100.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good = [m for m in matches if m.distance < 50]
    avg_dist = np.mean([m.distance for m in good]) if good else 100.0

    # RANSAC geometric verification: filter out random matches
    inlier_count = 0
    if kp1 is not None and kp2 is not None and len(good) >= 8:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0, maxIters=1000)
            if mask is not None:
                inlier_count = int(mask.sum())
        except Exception:
            inlier_count = 0

    return len(good), len(matches), avg_dist, inlier_count


def orb_confidence(good_count, total_count, avg_dist):
    if good_count == 0:
        return 0.0
    return np.exp(-avg_dist / 30.0) * good_count / (good_count + 5.0)


# ============================================================
# Similarity helpers
# ============================================================

def fft_similarity(v1, v2):
    diff = v1 - v2
    return 1.0 / (1.0 + np.sqrt(np.mean(diff ** 2)) * 5)


# Dummy stubs for removed features
def shape_similarity(fp, ff): return {'_shape_aggregate': 0.5, '_acorr_aggregate': 0.5}
def glcm_similarity(fp, ff): return {}
def logpolar_similarity(fp, ff): return {}
