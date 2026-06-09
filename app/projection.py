"""Project 512-d CLIP embeddings to 2D for the live memory map.

The PCA basis is fit offline by scripts/prepare_footage.py on the mission
footage itself, so the live projection is stable and well-spread.
"""

import numpy as np

from .constants import PROJECTION_FILE


class MemoryMapProjector:
    def __init__(self):
        self.mean = None
        self.components = None  # (512, 2)
        self.lo = None
        self.hi = None

    def load(self):
        data = np.load(PROJECTION_FILE)
        self.mean = data["mean"]
        self.components = data["components"]
        self.lo = data["lo"]
        self.hi = data["hi"]

    def project(self, embedding: np.ndarray) -> tuple[float, float]:
        """Returns (x, y) in [0, 1] with a small margin for outliers."""
        xy = (embedding - self.mean) @ self.components
        norm = (xy - self.lo) / (self.hi - self.lo + 1e-9)
        x, y = float(np.clip(norm[0], 0.0, 1.0)), float(np.clip(norm[1], 0.0, 1.0))
        return x, y
