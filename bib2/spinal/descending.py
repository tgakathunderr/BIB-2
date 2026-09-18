"""Descending Motor Pathways (Lateral/Anterior Corticospinal Tracts)."""

from typing import Tuple
import numpy as np


class CorticospinalTract:
    """
    Corticospinal Tract.
    Originates in Layer V Betz / pyramidal cells of M1 -> internal capsule -> medullary pyramids.
    - 85-90% decussates at pyramidal decussation -> Lateral Corticospinal Tract (distal limb control).
    - 10-15% descends uncrossed -> Anterior Corticospinal Tract (axial posture).
    """

    def __init__(self, motor_dim: int = 64):
        self.motor_dim = motor_dim
        self.decussation_ratio = 0.85

    def conduct(self, cortical_drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(cortical_drive, dtype=np.float32)
        if arr.shape[0] != self.motor_dim:
            padded = np.zeros(self.motor_dim, dtype=np.float32)
            sz = min(arr.size, self.motor_dim)
            padded[:sz] = arr.flat[:sz]
            arr = padded

        lateral_motor = arr * self.decussation_ratio
        anterior_motor = arr * (1.0 - self.decussation_ratio)
        return lateral_motor, anterior_motor


class DescendingTractsManager:
    """Coordinates descending motor tracts from cortex and brainstem to spinal motor pools."""

    def __init__(self, motor_dim: int = 64):
        self.corticospinal = CorticospinalTract(motor_dim=motor_dim)
        self.rubrospinal_gain = 0.4
        self.vestibulospinal_gain = 0.5
        self.reticulospinal_gain = 0.3

    def conduct_motor_signals(self, cortical_drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.corticospinal.conduct(cortical_drive)
