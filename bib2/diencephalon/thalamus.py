"""Thalamus Complex and Specific Relay Nuclei."""

from typing import Dict, List, Optional
import numpy as np

RELAY_NUCLEI = [
    "LGN",     # Lateral Geniculate: Retinogeniculate vision (magnocellular & parvocellular)
    "MGN",     # Medial Geniculate: Ascending audition from inferior colliculus to A1
    "VPL",     # Ventral Posterolateral: Somatosensation (body via DCML & Spinothalamic)
    "VPM",     # Ventral Posteromedial: Facial sensation (CN V)
    "VA_VL",   # Ventral Anterior & Lateral: Motor gating (Basal Ganglia & Cerebellum to M1)
    "Anterior",# Anterior Nucleus: Episodic memory (mammillothalamic tract in Papez loop)
    "Pulvinar",# Pulvinar: Attentional routing to parietal-temporal-occipital cortices
    "DM",      # Dorsomedial: Executive and affective integration with prefrontal cortex
]


class ThalamusComplex:
    """
    Diencephalic Sensory and Motor Gateway.
    Routes ascending sensory streams and subcortical motor proposals to the neocortex.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.relay_nuclei: Dict[str, np.ndarray] = {
            nucleus: np.zeros(dim, dtype=np.float32) for nucleus in RELAY_NUCLEI
        }

    def relay_sensory(self, nucleus: str, data: np.ndarray) -> np.ndarray:
        """Route incoming sensory afferent through specific relay nucleus."""
        if nucleus not in self.relay_nuclei:
            raise KeyError(f"Unknown thalamic relay nucleus: {nucleus}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]

        self.relay_nuclei[nucleus] = arr.copy()
        return self.relay_nuclei[nucleus]

    def read_nucleus(self, nucleus: str) -> np.ndarray:
        """Read current activation of a thalamic nucleus."""
        if nucleus not in self.relay_nuclei:
            raise KeyError(f"Unknown thalamic relay nucleus: {nucleus}")
        return self.relay_nuclei[nucleus]

    def set_nucleus(self, nucleus: str, data: np.ndarray) -> None:
        """Set activation of a thalamic nucleus."""
        if nucleus not in self.relay_nuclei:
            raise KeyError(f"Unknown thalamic relay nucleus: {nucleus}")
        arr = np.asarray(data, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]
        self.relay_nuclei[nucleus] = arr
