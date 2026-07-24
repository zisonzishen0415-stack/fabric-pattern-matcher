"""
Fast Gabor energy features for fabric texture analysis.
Multi-scale, multi-orientation filter bank.
Lighting-invariant (operates on gradients/edges, not raw pixels).
"""
import cv2
import numpy as np


def build_gabor_bank():
    """Build a bank of Gabor filters (4 scales × 6 orientations = 24 filters)."""
    kernels = []
    for scale in [4, 8, 12, 16]:
        for theta in [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21), sigma=scale, theta=theta,
                lambd=scale * 2, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            kernels.append(kernel)
    return kernels


# Pre-build once
_GABOR_BANK = None


def get_gabor_bank():
    global _GABOR_BANK
    if _GABOR_BANK is None:
        _GABOR_BANK = build_gabor_bank()
    return _GABOR_BANK


def extract_gabor_vector(gray):
    """
    Extract Gabor energy features.
    Returns: 48-dim vector (24 filters × 2 stats: mean energy + std energy)
    """
    gray = cv2.resize(gray, (128, 128)).astype(np.float32)
    # Normalize to zero mean unit variance (remove DC lighting)
    gray = (gray - np.mean(gray)) / (np.std(gray) + 1e-8)

    features = []
    for kernel in get_gabor_bank():
        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
        energy = np.abs(filtered)
        features.append(float(np.mean(energy)))
        features.append(float(np.std(energy)))

    feat = np.array(features, dtype=np.float32)
    # L2 normalize
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm


def gabor_distance(v1, v2):
    """Cosine distance between two Gabor vectors (both already L2-normalized)."""
    return float(np.dot(v1, v2))
