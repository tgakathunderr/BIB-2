"""Medulla Oblongata: Pre-Bötzinger Complex, Baroreflex, and Inferior Olive."""

from typing import Tuple
import numpy as np


class PreBotzingerComplex:
    """
    Pre-Bötzinger Complex (preBötC).
    Rhythmogenic core for eupneic inspiration derived from Dbx1/Vglut2 glutamatergic interneurons.
    Rhythms emerge via recurrent excitation, persistent sodium (I_NaP), and I_CAN conductances.
    Exhibits high sensitivity to mu-opioid receptor depression.
    """

    def __init__(self):
        self.phase: float = 0.0
        self.frequency: float = 0.25  # ~15 breaths per minute at 60s
        self.persistent_na_current: float = 1.0
        self.i_can_current: float = 1.0
        self.opioid_inhibition: float = 0.0
        self.burst_threshold: float = 0.85

    def apply_opioid_agonism(self, agonism: float) -> None:
        """Agonism of mu-opioid receptors hyperpolarizes rhythmogenic interneurons."""
        self.opioid_inhibition = float(np.clip(agonism, 0.0, 1.0))

    def step(self, dt: float = 0.1) -> int:
        """Advance pacemaker; returns 1 on inspiratory burst, 0 otherwise."""
        # Effective drive suppressed by opioids
        effective_drive = max(0.05, 1.0 - self.opioid_inhibition * 0.9)
        self.phase += self.frequency * dt * effective_drive

        if self.phase >= 1.0:
            self.phase -= 1.0
            return 1
        return 0


class InferiorOlive:
    """
    Inferior Olivary Complex.
    Sole generator of climbing fibers to cerebellar Purkinje cells.
    Fires low-frequency (1-2 Hz) complex spikes encoding motor prediction errors.
    """

    def __init__(self, output_dim: int = 64):
        self.output_dim = output_dim

    def generate_climbing_spikes(self, error_vector: np.ndarray) -> np.ndarray:
        """Transform motor performance discrepancy into climbing fiber complex spike burst."""
        err = np.asarray(error_vector, dtype=np.float32)
        if err.size < self.output_dim:
            padded = np.zeros(self.output_dim, dtype=np.float32)
            padded[:err.size] = err.flat
            err = padded
        else:
            err = err[:self.output_dim]

        # Complex spikes are non-linear, high-calcium bursts triggered by error threshold
        abs_err = np.abs(err)
        climbing_spikes = np.where(abs_err > 0.3, np.tanh(abs_err * 2.0), 0.0).astype(np.float32)
        return climbing_spikes


class MedullaOblongata:
    """Medullary autonomic and motor integration core."""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.pre_botzinger = PreBotzingerComplex()
        self.inferior_olive = InferiorOlive(output_dim=feature_dim)

    def process_baroreflex(self, bp_sys: float) -> Tuple[float, float]:
        """
        NTS receives visceral baroreceptor afferents:
        - Elevating arterial pressure excites NTS -> CVLM inhibits RVLM (lowering sympathetic tone).
        - Concurrently excites nucleus ambiguus / dorsal motor vagus (elevating parasympathetic tone).
        """
        deviation = bp_sys - 120.0
        if deviation > 0:
            # Hypertension -> downregulate sympathetic, upregulate vagus
            symp = max(0.05, 0.3 - (deviation / 100.0) * 0.35)
            parasymp = min(0.95, 0.7 + (deviation / 100.0) * 0.35)
        else:
            # Hypotension -> elevate sympathetic to preserve cerebral perfusion
            symp = min(0.95, 0.3 + (abs(deviation) / 100.0) * 0.4)
            parasymp = max(0.05, 0.7 - (abs(deviation) / 100.0) * 0.4)

        return float(symp), float(parasymp)
