"""14 Specialized Brodmann Areas, Visual Stream Bifurcation, and Cerebral Neocortex."""

from typing import Dict, List, Tuple
import numpy as np
from .layer import CanonicalMicrocircuit

BRODMANN_AREAS = [
    "M1_PrimaryMotor",              # BA 4: Betz cells, somatotopic motor homunculus
    "Premotor_SMA",                 # BA 6: Motor sequencing, postural adjustment
    "FrontalEyeFields",             # BA 8: Voluntary conjugate saccades
    "Broca_SpeechProduction",       # BA 44/45: Expressive speech, syntactic sequencing
    "DLPFC_WorkingMemory",          # Dorsolateral PFC: Abstract planning & multi-slot WM
    "OFC_Valuation",                # Orbitofrontal Cortex: Valuation and outcome tracking
    "VMPFC_AffectiveDecision",      # Ventromedial PFC: Somatic marker emotional decision-making
    "S1_Somatosensory",             # BA 3a/3b/1/2: Contralateral tactile/pain homunculus
    "SPL_SpatialIntegration",       # BA 5/7: Multi-modal spatial coordinate transformations
    "IPL_MultimodalHub",            # BA 39/40 (Angular & Supramarginal): Cross-modal convergence
    "A1_PrimaryAuditory",           # BA 41/42 (Heschl's): Tonotopic acoustic representation
    "Wernicke_SpeechComprehension", # BA 22: Phonological decoding & receptive language
    "IT_ObjectRecognition",         # BA 20/21: Visual form and semantic categorisation
    "FFA_FaceArea",                 # BA 37 (Fusiform): Structural facial identity
    "V1_Visual",                    # BA 17: Retinotopic striate cortex with foveal magnification
    "Insula_Interoception",         # Insula: Visceral physiological monitoring & distress
]


class CorticalAreaRegistry:
    """Registry mapping 14 specialized Brodmann cortical areas to canonical microcircuits."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.areas: Dict[str, CanonicalMicrocircuit] = {
            area_name: CanonicalMicrocircuit(dim=dim) for area_name in BRODMANN_AREAS
        }

    def forward_area(self, area_name: str, thalamic_input: np.ndarray) -> np.ndarray:
        """Execute laminar pass through an area; returns Layer V subcortical output."""
        if area_name not in self.areas:
            raise KeyError(f"Unknown Brodmann area: {area_name}")
        _, _, l5_out, _ = self.areas[area_name].forward(thalamic_input)
        return l5_out


class CerebralNeocortex:
    """
    Complete Cerebral Neocortex.
    Coordinates 14 functional Brodmann areas, visual stream bifurcation (Dorsal vs Ventral),
    and long-range association fasciculi (e.g. Arcuate Fasciculus between Wernicke and Broca).
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.registry = CorticalAreaRegistry(dim=dim)

        # Visual stream bifurcation projection weights
        rng = np.random.default_rng(42)
        self.w_v1_dorsal  = rng.uniform(0.5, 0.8, (dim, dim)).astype(np.float32) # MT/V5 motion
        self.w_v1_ventral = rng.uniform(0.5, 0.8, (dim, dim)).astype(np.float32) # V4/IT form

        # Arcuate Fasciculus (Wernicke -> Broca bidirectional tract)
        self.arcuate_fasciculus = np.zeros(dim, dtype=np.float32)

    def process_visual_bifurcation(self, v1_sdr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bifurcate Primary Visual Cortex (V1) signal into:
        1. Dorsal "Where/How" stream (MT/V5 -> Posterior Parietal): Visuomotor spatial guidance.
        2. Ventral "What" stream (V4 -> IT/FFA): High-resolution form and semantic recognition.
        """
        v1_l5 = self.registry.forward_area("V1_Visual", v1_sdr)

        dorsal_where = np.maximum(0.0, np.tanh(np.dot(v1_l5, self.w_v1_dorsal) / np.sqrt(self.dim)))
        ventral_what = np.maximum(0.0, np.tanh(np.dot(v1_l5, self.w_v1_ventral) / np.sqrt(self.dim)))

        # Update downstream parietal and temporal areas
        self.registry.forward_area("SPL_SpatialIntegration", dorsal_where)
        self.registry.forward_area("IT_ObjectRecognition", ventral_what)

        return dorsal_where, ventral_what

    def process_language_repetition(self, auditory_phonemes: np.ndarray) -> np.ndarray:
        """
        Language repetition via Arcuate Fasciculus:
        Audition -> Wernicke (BA 22 decoding) -> Arcuate Fasciculus -> Broca (BA 44/45 motor speech).
        """
        wernicke_out = self.registry.forward_area("Wernicke_SpeechComprehension", auditory_phonemes)
        self.arcuate_fasciculus = wernicke_out.copy()
        broca_motor_speech = self.registry.forward_area("Broca_SpeechProduction", self.arcuate_fasciculus)
        return broca_motor_speech

    def get_total_synaptic_weight(self) -> float:
        """Calculate total synaptic weight across all cortical microcircuits (for SHY downscaling validation)."""
        total = 0.0
        for area in self.registry.areas.values():
            total += float(np.sum(np.abs(area.w_thal_l4)) +
                           np.sum(np.abs(area.w_l4_l23)) +
                           np.sum(np.abs(area.w_l23_l5)) +
                           np.sum(np.abs(area.w_l5_l6)))
        return total
