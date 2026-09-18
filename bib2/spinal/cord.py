"""Spinal Cord Engine implementing Rexed Laminae I-X, reflex loops, and tract conduits."""

from typing import Dict, List, Tuple
import numpy as np
from .reflexes import SpinalReflexManager
from .ascending import AscendingTractsManager
from .descending import DescendingTractsManager

LAMINA_NAMES = [
    "Lamina_I_Marginal",
    "Lamina_II_SubstantiaGelatinosa",
    "Lamina_III_NucleusProprius",
    "Lamina_IV_NucleusPropriusDeep",
    "Lamina_V_WideDynamicRange",
    "Lamina_VI_ProprioceptiveCore",
    "Lamina_VII_IntermediateIML",
    "Lamina_VIII_CommissuralInterneurons",
    "Lamina_IX_AlphaMotorPool",
    "Lamina_X_CentralCanalDecussation",
]


class SpinalCordEngine:
    """
    Spinal Cord Engine.
    Coordinates the 10 Rexed Laminae, monosynaptic and polysynaptic reflex arcs,
    ascending somatosensory tracts (DCML, Spinothalamic), and descending corticospinal motor tracts.
    """

    def __init__(self, lamina_dim: int = 64, myotome_dim: int = 8):
        self.lamina_dim = lamina_dim
        self.myotome_dim = myotome_dim
        self.lamina_names: List[str] = LAMINA_NAMES
        self.laminae = np.zeros((10, lamina_dim), dtype=np.float32)

        self.reflexes = SpinalReflexManager(myotome_dim=myotome_dim)
        self.ascending = AscendingTractsManager(out_dim=lamina_dim)
        self.descending = DescendingTractsManager(motor_dim=lamina_dim)

    def process_segment_reflex(self, segment: str, afferent: np.ndarray) -> np.ndarray:
        """Process local segmental reflex and return motor efferent activation."""
        efferent, _ = self.reflexes.process_stretch_reflex(segment, afferent)
        return efferent

    def conduct_ascending(self, touch_data: np.ndarray, pain_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Route tactile and pain inputs through dorsal horn laminae into DCML and Spinothalamic tracts."""
        # Update dorsal horn laminae activity
        arr_touch = np.asarray(touch_data, dtype=np.float32)
        arr_pain = np.asarray(pain_data, dtype=np.float32)

        if arr_pain.size > 0:
            sz = min(arr_pain.size, self.lamina_dim)
            self.laminae[0, :sz] = arr_pain.flat[:sz]  # Lamina I
            self.laminae[1, :sz] = arr_pain.flat[:sz] * 0.9  # Lamina II
            self.laminae[4, :sz] = arr_pain.flat[:sz] * 0.8  # Lamina V WDR

        if arr_touch.size > 0:
            sz = min(arr_touch.size, self.lamina_dim)
            self.laminae[2, :sz] = arr_touch.flat[:sz]  # Lamina III
            self.laminae[3, :sz] = arr_touch.flat[:sz]  # Lamina IV

        return self.ascending.conduct(touch_data, pain_data)

    def conduct_descending(self, cortical_drive: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Route descending motor commands into Lamina IX lower motor neuron pools."""
        lateral, anterior = self.descending.conduct_motor_signals(cortical_drive)
        # Update Lamina IX motor pool activation
        sz = min(lateral.size, self.lamina_dim)
        self.laminae[8, :sz] = np.clip(lateral[:sz] + anterior[:sz], 0.0, 1.0)
        return lateral, anterior
