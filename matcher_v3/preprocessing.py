"""
Preprocessing: color calibration and image normalization.
Handles the key failure mode: color shift from uncontrolled lighting.
"""
import cv2
import numpy as np


def imread_unicode(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def compute_library_hsv_profile(fabric_paths):
    """
    Compute the average HSV distribution across all fabric library images.
    This serves as the calibration target for query photos.
    Returns: (avg_h_hist, avg_s_hist, avg_v_hist) — 256-bin histograms
    """
    h_hists, s_hists, v_hists = [], [], []

    for path in fabric_paths:
        img = imread_unicode(path)
        img = cv2.resize(img, (256, 256))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])

        h_hists.append(cv2.normalize(h_hist, h_hist).flatten())
        s_hists.append(cv2.normalize(s_hist, s_hist).flatten())
        v_hists.append(cv2.normalize(v_hist, v_hist).flatten())

    return (
        np.mean(h_hists, axis=0),
        np.mean(s_hists, axis=0),
        np.mean(v_hists, axis=0),
    )


def histogram_match_channel(src_channel, target_hist):
    """
    Match source channel's histogram to target distribution.
    Returns matched channel (uint8).
    """
    # Compute source CDF
    src_hist = cv2.calcHist([src_channel], [0], None, [256], [0, 256])
    src_hist = src_hist.flatten() / src_hist.sum()
    src_cdf = np.cumsum(src_hist)

    # Target CDF
    target_cdf = np.cumsum(target_hist)

    # Build lookup table
    lut = np.zeros(256, dtype=np.uint8)
    t_idx = 0
    for s_idx in range(256):
        while t_idx < 255 and target_cdf[t_idx] < src_cdf[s_idx]:
            t_idx += 1
        lut[s_idx] = t_idx

    return cv2.LUT(src_channel, lut)


def calibrate_to_library(img, lib_h_profile, lib_s_profile, lib_v_profile):
    """
    Calibrate photo to match the fabric library's color distribution.
    Only adjusts V (brightness) and mildly S (saturation).
    H (hue) is preserved to avoid destroying the actual color information.
    """
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Histogram-match V to library V profile
    v_cal = histogram_match_channel(v, lib_v_profile)

    # Mild S adjustment: stretch contrast, but don't remap
    s_cal = cv2.equalizeHist(s)

    # H unchanged
    hsv_cal = cv2.merge([h, s_cal, v_cal])
    return cv2.cvtColor(hsv_cal, cv2.COLOR_HSV2BGR)


def calibrate_photo_adaptive(img, lib_profile):
    """
    Try both calibrated and raw, return both versions.
    The matcher can blend scores from both.
    """
    lib_h, lib_s, lib_v = lib_profile
    calibrated = calibrate_to_library(img, lib_h, lib_s, lib_v)
    return {
        'raw': img,
        'calibrated': calibrated,
    }


def white_patch_retinex(img, percentile=99):
    """
    White Patch Retinex color constancy.
    Assumes the brightest pixel is white → scales each channel independently.
    More robust than Gray World for scenes with dominant colors.
    """
    b, g, r = cv2.split(img.astype(np.float32))
    # Find the percentile-brightest value in each channel
    max_b = np.percentile(b, percentile)
    max_g = np.percentile(g, percentile)
    max_r = np.percentile(r, percentile)
    # Scale
    scale_b = 255.0 / max_b if max_b > 0 else 1
    scale_g = 255.0 / max_g if max_g > 0 else 1
    scale_r = 255.0 / max_r if max_r > 0 else 1
    # Clamp gain
    scale_b = np.clip(scale_b, 0.3, 3.0)
    scale_g = np.clip(scale_g, 0.3, 3.0)
    scale_r = np.clip(scale_r, 0.3, 3.0)
    b_c = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g_c = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r_c = np.clip(r * scale_r, 0, 255).astype(np.uint8)
    return cv2.merge([b_c, g_c, r_c])


def apply_clahe(img):
    """CLAHE contrast enhancement on L channel of LAB"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
