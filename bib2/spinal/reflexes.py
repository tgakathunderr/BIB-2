"""Spinal Reflex Arcs: Monosynaptic Stretch & Polysynaptic Crossed-Extensor."""

from typing import Tuple
import numpy as np


class SpinalReflexManager:
    """Computes fast segmental sensorimotor reflex arcs without prior cerebral delay."""

    def __init__(self, myotome_dim: int = 8):
        self.myotome_dim = myotome_dim

    def process_stretch_reflex(self, segment: str, stretch_afferent: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Monosynaptic Stretch Reflex (Ia Myotatic Loop).
        Group Ia annulospiral stretch excites homonymous alpha motor neurons,
        while Ia inhibitory interneurons apply glycinergic reciprocal inhibition to antagonists.
        """
        arr = np.asarray(stretch_afferent, dtype=np.float32)
        ia_stretch = float(arr[0]) if arr.size > 0 else 0.0

        efferent = np.zeros(self.myotome_dim, dtype=np.float32)
        if ia_stretch > 0.3:
            # Monosynaptic excitation of homonymous alpha motor pool
            efferent[0] = float(np.clip(ia_stretch * 1.1, 0.0, 1.0))
            if self.myotome_dim > 1:
                efferent[1] = float(np.clip(ia_stretch * 0.7, 0.0, 1.0))

        # Reciprocal inhibition intensity onto antagonist pool
        antagonist_inhibition = float(np.clip(ia_stretch * 0.85, 0.0, 1.0))
        return efferent, antagonist_inhibition

    def process_crossed_extensor(self, segment: str, nociceptive_afferent: np.ndarray) -> Tuple[float, float]:
        """
        Polysynaptic Flexor Withdrawal and Crossed-Extensor Reflex.
        Painful cutaneous input engages multi-segmental interneurons:
        - Ipsilateral flexors excited / extensors inhibited (withdrawal).
        - Contralateral extensors excited / flexors inhibited (postural stabilization).
        """
        arr = np.asarray(nociceptive_afferent, dtype=np.float32)
        # Pain intensity sampled from A-delta / C-fiber channels (indices 2 to 6)
        pain_idx = 4 if arr.size > 4 else (arr.size - 1 if arr.size > 0 else 0)
        pain_intensity = float(arr[pain_idx]) if arr.size > 0 else 0.0

        if pain_intensity > 0.4:
            ipsi_flexion = float(np.clip(pain_intensity * 0.95, 0.0, 1.0))
            contra_extension = float(np.clip(pain_intensity * 0.85, 0.0, 1.0))
        else:
            ipsi_flexion = 0.0
            contra_extension = 0.0

        return ipsi_flexion, contra_extension
