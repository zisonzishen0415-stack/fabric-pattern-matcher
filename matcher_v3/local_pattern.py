"""
Local pattern features using Dense SIFT + VLAD encoding.
Captures actual fabric motifs (local patterns), not just global stats.
Lighting-invariant (SIFT is gradient-based), scale-robust.
"""
import cv2
import numpy as np


def extract_dense_sift(gray, step=8, sizes=(8, 16, 24)):
    """Extract dense SIFT descriptors on a grid."""
    sift = cv2.SIFT_create()
    kps = []
    h, w = gray.shape
    for size in sizes:
        for y in range(size // 2, h - size // 2, step):
            for x in range(size // 2, w - size // 2, step):
                kp = cv2.KeyPoint(float(x), float(y), float(size))
                kps.append(kp)
    _, des = sift.compute(gray, kps)
    return des  # N x 128


def build_vlad_centers(fabric_images, n_centers=64, sample_per_image=200):
    """
    Build VLAD cluster centers from the fabric library.
    Samples SIFT descriptors from all fabrics, runs k-means.
    Returns: (centers, pca_matrix) or (centers, None)
    """
    all_descriptors = []
    for img in fabric_images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Resize to standard size
        gray = cv2.resize(gray, (256, 256))
        des = extract_dense_sift(gray, step=16, sizes=(16, 24, 32))
        if des is not None and len(des) > 0:
            # Sample to control total size
            if len(des) > sample_per_image:
                idx = np.random.choice(len(des), sample_per_image, replace=False)
                des = des[idx]
            all_descriptors.append(des)

    if not all_descriptors:
        return np.zeros((n_centers, 128), dtype=np.float32)

    all_des = np.vstack(all_descriptors)
    print(f"  VLAD: clustering {all_des.shape[0]} SIFT descriptors into {n_centers} centers...")

    # K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    compactness, labels, centers = cv2.kmeans(
        all_des.astype(np.float32), n_centers, None, criteria, 3,
        cv2.KMEANS_RANDOM_CENTERS
    )
    return centers  # n_centers x 128


def encode_vlad(gray, centers):
    """
    VLAD encoding of an image:
    1. Extract dense SIFT
    2. Assign each descriptor to nearest center
    3. Accumulate residuals (descriptor - center) per cluster
    4. Normalize
    Returns: VLAD vector (n_centers * 128 dimensions)
    """
    gray = cv2.resize(gray, (256, 256))
    des = extract_dense_sift(gray, step=16, sizes=(16, 24, 32))

    if des is None or len(des) == 0 or len(centers) == 0:
        return np.zeros(len(centers) * 128, dtype=np.float32)

    n_centers = len(centers)
    dim = 128
    vlad = np.zeros((n_centers, dim), dtype=np.float32)

    # For each descriptor, find nearest center and accumulate residual
    for d in des:
        # Find nearest center
        distances = np.sum((centers - d) ** 2, axis=1)
        nearest = np.argmin(distances)
        vlad[nearest] += (d - centers[nearest])

    # Intra-normalization (per-cluster L2 norm)
    for i in range(n_centers):
        norm = np.linalg.norm(vlad[i])
        if norm > 1e-8:
            vlad[i] /= norm

    # Global L2 normalization
    vlad_flat = vlad.flatten()
    norm = np.linalg.norm(vlad_flat)
    if norm > 1e-8:
        vlad_flat /= norm

    return vlad_flat


def vlad_similarity(v1, v2):
    """Cosine similarity between two VLAD vectors."""
    if len(v1) == 0 or len(v2) == 0:
        return 0.5
    dot = np.dot(v1, v2)
    sim = dot  # Already L2-normalized, so dot product = cosine
    # Map from [-1, 1] to [0, 1]
    return (sim + 1) / 2
