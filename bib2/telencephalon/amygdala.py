"""Amygdaloid Complex: BLA, CeA, and Intercalated Cells (ITC)."""

from typing import Tuple
import numpy as np


class AmygdaloidComplex:
    """
    Amygdaloid Complex.
    Affective salience and threat detection center:
    - BLA (Basolateral): Multi-modal sensory convergence, associative fear conditioning.
    - CeA (Central Nucleus): Efferent driver to Hypothalamus and PAG via stria terminalis.
    - ITC (Intercalated Cells): GABAergic gating cells mediating safety learning / extinction.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.bla_weights = np.zeros(dim, dtype=np.float32)
        self.itc_inhibition: float = 0.0  # Safety / extinction gate
        self.threat_memory = np.zeros(dim, dtype=np.float32)

    def condition_threat(self, stimulus_sdr: np.ndarray, unconditioned_pain: float, lr: float = 0.2) -> None:
        """Associate sensory stimulus (CS) with unconditioned pain/threat (US)."""
        arr = np.asarray(stimulus_sdr, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]

        # Hebbian LTP strengthening of BLA threat connections
        self.bla_weights += lr * arr * float(unconditioned_pain)
        np.clip(self.bla_weights, 0.0, 2.0, out=self.bla_weights)
        self.threat_memory = arr.copy()

    def evaluate_salience(self, stimulus_sdr: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate emotional and threat salience of an incoming sensory state.
        Returns: (overall_salience, cea_fear_output).
        """
        arr = np.asarray(stimulus_sdr, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]

        # BLA threat appraisal
        bla_activation = float(np.dot(arr, self.bla_weights) / max(1.0, np.sum(self.bla_weights)))
        # Central nucleus fear output gated by ITC extinction
        cea_output = float(np.clip(bla_activation - self.itc_inhibition, 0.0, 1.0))
        # Salience includes absolute threat level
        salience = float(np.clip(np.mean(arr) * 0.4 + cea_output * 0.6, 0.0, 1.0))

        return salience, cea_output

    def extinguish_threat(self, extinction_rate: float = 0.1) -> None:
        """Engage intercalated cells to extinguish conditioned fear."""
        self.itc_inhibition = min(1.0, self.itc_inhibition + extinction_rate)
