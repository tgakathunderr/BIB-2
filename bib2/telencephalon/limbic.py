"""Hippocampal Formation and Circuit of Papez Episodic Memory Loop."""

from typing import List, Optional
import numpy as np


class DentateGyrus:
    """
    Dentate Gyrus (DG).
    Employs sparse high-dimensional projection with k-Winners-Take-All (KWTA)
    inhibition to perform robust pattern separation on overlapping entorhinal inputs.
    """

    def __init__(self, in_dim: int = 64, dg_dim: int = 256, sparsity: float = 0.08, seed: int = 42):
        self.in_dim = in_dim
        self.dg_dim = dg_dim
        self.k_winners = max(1, int(dg_dim * sparsity))

        rng = np.random.default_rng(seed)
        # Random non-linear expansion weights
        self.weights = rng.standard_normal((in_dim, dg_dim), dtype=np.float32)

    def separate(self, entorhinal_sdr: np.ndarray) -> np.ndarray:
        arr = np.asarray(entorhinal_sdr, dtype=np.float32)
        if arr.size < self.in_dim:
            padded = np.zeros(self.in_dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.in_dim]

        projection = np.dot(arr, self.weights)
        # KWTA sparsity
        separated = np.zeros(self.dg_dim, dtype=np.float32)
        top_indices = np.argpartition(projection, -self.k_winners)[-self.k_winners:]
        separated[top_indices] = np.maximum(0.1, projection[top_indices])
        return separated


class HippocampalCA3:
    """
    Hippocampal CA3.
    Contains dense recurrent axon collaterals forming an auto-associative attractor network.
    Executes pattern completion from noisy or partial cues.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.recurrent_matrix = np.zeros((dim, dim), dtype=np.float32)
        self.stored_patterns: List[np.ndarray] = []

    def store(self, pattern: np.ndarray, lr: float = 1.0) -> None:
        """Store pattern into auto-associative Hebbian recurrent matrix."""
        arr = np.asarray(pattern, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]

        dot_self = float(np.dot(arr, arr))
        if dot_self > 0:
            # Normalized Hebbian outer product
            self.recurrent_matrix += (lr / dot_self) * np.outer(arr, arr)
            self.stored_patterns.append(arr.copy())

    def complete(self, cue: np.ndarray, iterations: int = 3) -> np.ndarray:
        """Iteratively reconstruct complete stored pattern from degraded cue."""
        state = np.asarray(cue, dtype=np.float32).copy()
        if state.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:state.size] = state.flat
            state = padded
        else:
            state = state[:self.dim]

        for _ in range(iterations):
            drive = np.dot(state, self.recurrent_matrix).astype(np.float32)
            # Retain non-zero cue channels, fill in missing channels with recurrent drive
            missing_mask = (state == 0.0)
            state = np.where(missing_mask, drive, state)
            state = np.clip(state, 0.0, 1.0)

        return state.astype(np.float32)


class HippocampalCA1:
    """Hippocampal CA1: Temporal sequence binding and routing to Subiculum."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def forward(self, ca3_activation: np.ndarray) -> np.ndarray:
        return np.tanh(np.asarray(ca3_activation, dtype=np.float32)[:self.dim])


class HippocampalFormation:
    """Trisynaptic Hippocampal Formation (DG -> CA3 -> CA1 -> Subiculum)."""

    def __init__(self, dim: int = 64, dg_dim: int = 256, seed: int = 42):
        self.dim = dim
        self.dentate_gyrus = DentateGyrus(in_dim=dim, dg_dim=dg_dim, seed=seed)
        self.ca3 = HippocampalCA3(dim=dim)
        self.ca1 = HippocampalCA1(dim=dim)
        self.subiculum = np.zeros(dim, dtype=np.float32)

    def process(self, entorhinal_input: np.ndarray) -> np.ndarray:
        # 1. DG pattern separation
        dg_out = self.dentate_gyrus.separate(entorhinal_input)
        # 2. CA3 completion / associative recall
        ca3_in = entorhinal_input[:self.dim] if entorhinal_input.size >= self.dim else np.zeros(self.dim, dtype=np.float32)
        ca3_out = self.ca3.complete(ca3_in)
        # 3. CA1 temporal sequencing
        ca1_out = self.ca1.forward(ca3_out)
        self.subiculum = ca1_out.copy()
        return self.subiculum


class CircuitOfPapez:
    """
    The Circuit of Papez Episodic Memory Architecture:
    Subiculum -> Fornix -> Mammillary Bodies -> Mammillothalamic Tract ->
    Anterior Thalamic Nucleus -> Thalamocingulate Tract -> Cingulate Gyrus ->
    Cingulum -> Entorhinal Cortex -> Hippocampus.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.mammillary_bodies = np.zeros(dim, dtype=np.float32)
        self.anterior_thalamus = np.zeros(dim, dtype=np.float32)
        self.cingulate_cortex = np.zeros(dim, dtype=np.float32)
        self.entorhinal_cortex = np.zeros(dim, dtype=np.float32)

    def cycle(self, subiculum_episode: np.ndarray) -> np.ndarray:
        """Cycle an experiential episode through the complete anatomical loop."""
        arr = np.asarray(subiculum_episode, dtype=np.float32)[:self.dim]
        # Fornix -> Mammillary
        self.mammillary_bodies = np.tanh(arr * 0.95)
        # Mammillothalamic Tract of Vicq d'Azyr -> Anterior Thalamus
        self.anterior_thalamus = np.tanh(self.mammillary_bodies * 0.92)
        # Thalamocingulate Tract -> Cingulate Cortex
        self.cingulate_cortex = np.tanh(self.anterior_thalamus * 0.90)
        # Cingulum Bundle -> Parahippocampal / Entorhinal Cortex
        self.entorhinal_cortex = np.tanh(self.cingulate_cortex * 0.88)
        return self.entorhinal_cortex


class LimbicComplex:
    """Master Limbic System container."""

    def __init__(self, dim: int = 64):
        self.hippocampus = HippocampalFormation(dim=dim)
        self.papez = CircuitOfPapez(dim=dim)
