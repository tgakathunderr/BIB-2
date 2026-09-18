"""Mesencephalon (Midbrain): Superior/Inferior Colliculi, PAG, and Red Nucleus."""

from typing import Dict, Tuple
import numpy as np


class SuperiorColliculus:
    """
    Superior Colliculus (Optic Tectum).
    Coordinates conjugate visual tracking, foveation, and rapid reflexive gaze saccades.
    """

    def __init__(self):
        pass

    def compute_saccade(self, visual_sdr: np.ndarray) -> np.ndarray:
        """Compute 2D saccadic gaze delta vector (dx, dy) towards salient stimulus."""
        arr = np.asarray(visual_sdr, dtype=np.float32)
        if arr.size == 0 or np.max(arr) == 0.0:
            return np.zeros(2, dtype=np.float32)

        peak_idx = int(np.argmax(arr))
        # Map 1D peak index across visual hemifield into normalized [-1.0, 1.0] coordinates
        norm_pos = (peak_idx / max(1, arr.size - 1)) * 2.0 - 1.0
        dx = float(norm_pos)
        dy = float(np.sin(norm_pos * np.pi) * 0.5)
        return np.array([dx, dy], dtype=np.float32)


class InferiorColliculus:
    """
    Inferior Colliculus.
    Principal midbrain auditory center relaying tonotopic signals from lateral lemniscus to MGN.
    """

    def __init__(self, audio_dim: int = 64):
        self.audio_dim = audio_dim

    def relay_auditory(self, cochlear_input: np.ndarray) -> np.ndarray:
        arr = np.asarray(cochlear_input, dtype=np.float32)
        out = np.zeros(self.audio_dim, dtype=np.float32)
        sz = min(arr.size, self.audio_dim)
        out[:sz] = arr.flat[:sz]
        return out


class PeriaqueductalGray:
    """
    Periaqueductal Gray (PAG).
    Coordinates descending pain modulation/analgesia and survival defensive programs (freeze vs fight/flight).
    """

    def __init__(self):
        self.analgesia_level: float = 0.0

    def process_threat_arousal(self, threat_level: float) -> Dict[str, bool]:
        """Determine whether defensive state requires freezing or active fight/flight."""
        if threat_level > 0.8:
            return {"freeze": False, "fight_flight": True}
        elif threat_level > 0.4:
            return {"freeze": True, "fight_flight": False}
        return {"freeze": False, "fight_flight": False}


class RedNucleus:
    """Red Nucleus driving rubrospinal flexor motor synergy execution."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def generate_rubrospinal_drive(self, cerebellar_input: np.ndarray) -> np.ndarray:
        arr = np.asarray(cerebellar_input, dtype=np.float32)
        out = np.zeros(self.dim, dtype=np.float32)
        sz = min(arr.size, self.dim)
        out[:sz] = np.tanh(arr.flat[:sz]) * 0.8
        return out


class Mesencephalon:
    """Midbrain integration complex."""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.superior_colliculus = SuperiorColliculus()
        self.inferior_colliculus = InferiorColliculus(audio_dim=feature_dim)
        self.pag = PeriaqueductalGray()
        self.red_nucleus = RedNucleus(dim=feature_dim)
