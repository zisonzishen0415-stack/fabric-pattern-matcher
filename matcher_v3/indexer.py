"""Indexer: full features + gabor + acorr fingerprints."""
import cv2
import numpy as np
import os
from .preprocessing import imread_unicode, compute_library_hsv_profile
from .features import extract_features, extract_orb
from .gabor_features import extract_gabor_vector
from .acorr_fingerprint import extract_acorr_fingerprint


class FabricIndex:
    def __init__(self):
        self.fabric_names = []
        self.lib_hsv_profile = None
        self.full_features = {}
        self.gabor_vectors = {}
        self.acorr_fingerprints = {}
        self.orb_descriptors = {}
        self.orb_keypoints = {}
        self.fabric_grays = {}
        self.is_built = False

    def build(self, fabric_dir):
        self._fabric_dir = os.path.abspath(fabric_dir)
        files = sorted(os.listdir(fabric_dir))
        print(f"  Indexing {len(files)} fabrics...")

        images = {}
        for f in files:
            try:
                img = imread_unicode(os.path.join(fabric_dir, f))
                if img is not None and img.size > 0:
                    images[f] = img
                    self.fabric_names.append(f)
            except Exception:
                pass
        n = len(self.fabric_names)
        print(f"  Valid: {n}")

        self.lib_hsv_profile = compute_library_hsv_profile(
            [os.path.join(fabric_dir, f) for f in self.fabric_names])

        for i, f in enumerate(self.fabric_names):
            img = images[f]
            self.full_features[f] = extract_features(img, scale=256)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.gabor_vectors[f] = extract_gabor_vector(gray)
            self.acorr_fingerprints[f] = extract_acorr_fingerprint(gray)
            kp, des = extract_orb(img)
            self.orb_keypoints[f] = kp
            self.orb_descriptors[f] = des
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n}")

        self.is_built = True
        print(f"  Done. {n} fabrics.")
        return self
