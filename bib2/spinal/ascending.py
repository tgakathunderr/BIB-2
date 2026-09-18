"""Ascending Somatosensory Pathways (DCML & Spinothalamic Tracts)."""

from typing import Tuple
import numpy as np
from ..types import TractData


class DCMLTract:
    """
    Dorsal Column-Medial Lemniscal (DCML) Pathway.
    Carries discriminative fine touch, vibration, and conscious proprioception.
    Ascends uncrossed in dorsal funiculus (Gracilis/Cuneatus) -> Medulla decussation -> VPL.
    """

    def __init__(self, out_dim: int = 64):
        self.out_dim = out_dim

    def transmit(self, touch_afferents: np.ndarray) -> np.ndarray:
        arr = np.asarray(touch_afferents, dtype=np.float32)
        out = np.zeros(self.out_dim, dtype=np.float32)
        # Expand and normalize somatotopic map
        if arr.size > 0:
            repeats = int(np.ceil(self.out_dim / arr.size))
            tiled = np.tile(arr, repeats)[:self.out_dim]
            out = tiled * 0.95
        return out


class SpinothalamicTract:
    """
    Anterolateral System / Spinothalamic Pathway.
    Carries nociception, thermal sensations, and crude touch.
    Decussates across anterior white commissure within 1-2 segments -> Contralateral VPL.
    """

    def __init__(self, out_dim: int = 64):
        self.out_dim = out_dim

    def transmit(self, pain_temp_afferents: np.ndarray) -> np.ndarray:
        arr = np.asarray(pain_temp_afferents, dtype=np.float32)
        out = np.zeros(self.out_dim, dtype=np.float32)
        if arr.size > 0:
            repeats = int(np.ceil(self.out_dim / arr.size))
            tiled = np.tile(arr, repeats)[:self.out_dim]
            out = tiled * 1.05
        return np.clip(out, 0.0, 1.0)


class AscendingTractsManager:
    """Coordinates ascending spinal projections into the brainstem and thalamus."""

    def __init__(self, out_dim: int = 64):
        self.dcml = DCMLTract(out_dim=out_dim)
        self.spinothalamic = SpinothalamicTract(out_dim=out_dim)

    def conduct(self, touch_data: np.ndarray, pain_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dcml_sig = self.dcml.transmit(touch_data)
        stt_sig = self.spinothalamic.transmit(pain_data)
        return dcml_sig, stt_sig
