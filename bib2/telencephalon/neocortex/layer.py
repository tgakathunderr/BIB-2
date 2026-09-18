"""Canonical Laminar Microcircuit (6 Layers: L1 through L6)."""

from typing import Tuple
import numpy as np


class CanonicalMicrocircuit:
    """
    Douglas & Martin Neocortical Canonical Microcircuit.
    Conserved 6-layered cytoarchitectonic computational unit:
    - Layer I   : Molecular neuropil, apical dendrites, long-range feedback.
    - Layer II/III: External granular & pyramidal cells (corticocortical association).
    - Layer IV  : Internal granular stellates (primary thalamocortical input target).
    - Layer V   : Internal pyramidal cells / Betz (subcortical & motor projection).
    - Layer VI  : Multiform cells (corticothalamic reciprocal feedback).

    Information Flow:
    Thalamus -> Layer IV -> Layer II/III -> Layer V (Motor/Subcortical output)
                                      |         |
                                      v         v
                                    Layer VI -> Thalamic Feedback
    """

    def __init__(self, dim: int = 64, seed: int = 42):
        self.dim = dim
        rng = np.random.default_rng(seed)

        # Laminar feedforward projection matrices
        self.w_thal_l4 = rng.uniform(0.6, 0.9, (dim, dim)).astype(np.float32)
        self.w_l4_l23  = rng.uniform(0.5, 0.8, (dim, dim)).astype(np.float32)
        self.w_l23_l5  = rng.uniform(0.6, 0.9, (dim, dim)).astype(np.float32)
        self.w_l5_l6   = rng.uniform(0.4, 0.7, (dim, dim)).astype(np.float32)

        # Recurrent / feedback matrices
        self.w_l6_feedback = rng.uniform(0.3, 0.5, (dim, dim)).astype(np.float32)

        # Layer state buffers
        self.l1 = np.zeros(dim, dtype=np.float32)
        self.l23 = np.zeros(dim, dtype=np.float32)
        self.l4 = np.zeros(dim, dtype=np.float32)
        self.l5 = np.zeros(dim, dtype=np.float32)
        self.l6 = np.zeros(dim, dtype=np.float32)

    def forward(self, thalamic_input: np.ndarray, top_down_l1_feedback: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute forward pass through canonical laminar microcircuit.
        Returns: (L4, L2/3, L5_subcortical_out, L6_thalamic_feedback).
        """
        thal = np.asarray(thalamic_input, dtype=np.float32)
        if thal.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:thal.size] = thal.flat
            thal = padded
        else:
            thal = thal[:self.dim]

        # 1. Layer IV receives specific thalamocortical afferents
        self.l4 = np.maximum(0.0, np.tanh(np.dot(thal, self.w_thal_l4) / np.sqrt(self.dim)))

        # 2. Layer IV projects to superficial associative Layers II/III
        l23_raw = np.dot(self.l4, self.w_l4_l23) / np.sqrt(self.dim)
        # Layer I apical dendrites integrate top-down attention / feedback
        self.l1 = np.full(self.dim, top_down_l1_feedback, dtype=np.float32)
        self.l23 = np.maximum(0.0, np.tanh(l23_raw + self.l1 * 0.3))

        # 3. Layer II/III projects to deep Layer V (subcortical/motor output neurons)
        l5_raw = np.dot(self.l23, self.w_l23_l5) / np.sqrt(self.dim)
        self.l5 = np.maximum(0.0, np.tanh(l5_raw * 1.2))

        # 4. Layer V and II/III drive Layer VI (corticothalamic feedback)
        l6_raw = np.dot(self.l5, self.w_l5_l6) / np.sqrt(self.dim)
        self.l6 = np.maximum(0.0, np.tanh(l6_raw))

        feedback_to_thal = np.dot(self.l6, self.w_l6_feedback) / np.sqrt(self.dim)

        return self.l4.copy(), self.l23.copy(), self.l5.copy(), feedback_to_thal.astype(np.float32)
