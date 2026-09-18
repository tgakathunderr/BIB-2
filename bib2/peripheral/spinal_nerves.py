"""Spinal Segmental Nerves Interface (31 Bilateral Pairs)."""

from typing import Dict, List, Optional
import numpy as np
from ..types import SpinalNerveSignal

SPINAL_SEGMENTS: List[str] = (
    [f"C{i}" for i in range(1, 9)] +
    [f"T{i}" for i in range(1, 13)] +
    [f"L{i}" for i in range(1, 6)] +
    [f"S{i}" for i in range(1, 6)] +
    ["Co1"]
)


class SpinalNervesInterface:
    """Manages the 31 bilateral spinal nerves dermatomal afferents and myotomal efferents."""

    def __init__(self, dermatome_dim: int = 16, myotome_dim: int = 8):
        self.dermatome_dim = dermatome_dim
        self.myotome_dim = myotome_dim
        self.segments: Dict[str, SpinalNerveSignal] = {}

        for seg in SPINAL_SEGMENTS:
            self.segments[seg] = SpinalNerveSignal(
                segment=seg,
                dermatome=np.zeros(dermatome_dim, dtype=np.float32),
                myotome=np.zeros(myotome_dim, dtype=np.float32)
            )

    def set_dermatome(self, segment: str, data: np.ndarray) -> None:
        """Inject somatosensory (tactile, nociceptive, proprioceptive) dermatomal input."""
        if segment not in self.segments:
            raise KeyError(f"Unknown spinal segment: {segment}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.shape[0] != self.dermatome_dim:
            padded = np.zeros(self.dermatome_dim, dtype=np.float32)
            sz = min(arr.size, self.dermatome_dim)
            padded[:sz] = arr.flat[:sz]
            arr = padded
        self.segments[segment].dermatome = arr

    def get_dermatome(self, segment: str) -> np.ndarray:
        """Read sensory dermatomal input from a segment."""
        if segment not in self.segments:
            raise KeyError(f"Unknown spinal segment: {segment}")
        return self.segments[segment].dermatome

    def set_myotome(self, segment: str, data: np.ndarray) -> None:
        """Set motor efferent command for a spinal segment myotome."""
        if segment not in self.segments:
            raise KeyError(f"Unknown spinal segment: {segment}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.shape[0] != self.myotome_dim:
            padded = np.zeros(self.myotome_dim, dtype=np.float32)
            sz = min(arr.size, self.myotome_dim)
            padded[:sz] = arr.flat[:sz]
            arr = padded
        self.segments[segment].myotome = arr

    def get_myotome(self, segment: str) -> np.ndarray:
        """Read motor efferent command for a spinal segment myotome."""
        if segment not in self.segments:
            raise KeyError(f"Unknown spinal segment: {segment}")
        return self.segments[segment].myotome

    def get_all_afferents(self) -> np.ndarray:
        """Concatenate all 31 dermatomes into a unified somatosensory vector (31 * 16 = 496)."""
        return np.concatenate([self.segments[seg].dermatome for seg in SPINAL_SEGMENTS])

    def set_all_efferents(self, vector: np.ndarray) -> None:
        """Distribute a flattened motor vector across all 31 segment myotomes (31 * 8 = 248)."""
        expected = len(SPINAL_SEGMENTS) * self.myotome_dim
        arr = np.asarray(vector, dtype=np.float32)
        if arr.size < expected:
            padded = np.zeros(expected, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        for idx, seg in enumerate(SPINAL_SEGMENTS):
            start = idx * self.myotome_dim
            end = start + self.myotome_dim
            self.segments[seg].myotome = arr[start:end].copy()
