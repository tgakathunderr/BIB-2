"""Metencephalic Pons: Pontine Nuclei, Locus Coeruleus, and Sublaterodorsal Atonia."""

from typing import Tuple
import numpy as np


class LocusCoeruleus:
    """
    Locus Coeruleus (LC).
    Primary noradrenergic hub of the central nervous system.
    Modulates vigilance, environmental salience, and sensory signal-to-noise ratio.
    Also regulates astrocytic volume and interstitial space compaction during wakefulness.
    """

    def __init__(self):
        self.tonic_baseline: float = 0.4
        self.phasic_burst: float = 0.0

    def step_arousal(self, salience: float, dt: float = 1.0) -> float:
        """Process environmental salience cues; returns active noradrenaline output."""
        if salience > 0.5:
            # Phasic burst response
            self.phasic_burst = float(np.clip(salience * 1.2, 0.0, 1.0))
        else:
            # Phasic decay back to tonic baseline
            self.phasic_burst = max(0.0, self.phasic_burst - 0.2 * dt)

        ne_output = float(np.clip(self.tonic_baseline + self.phasic_burst * 0.6, 0.0, 1.0))
        return ne_output


class SublaterodorsalNucleus:
    """
    Sublaterodorsal Nucleus (SLD).
    Coordinates somatic motor atonia during REM sleep via inhibitory glycinergic/GABAergic
    premotor interneurons in the ventromedial medulla and spinal cord.
    """

    def __init__(self):
        self.atonia_intensity: float = 0.0

    def compute_motor_inhibition(self, is_rem_sleep: bool) -> float:
        """Returns hyperpolarizing inhibitory drive on spinal alpha motor pools."""
        if is_rem_sleep:
            self.atonia_intensity = 0.95
        else:
            self.atonia_intensity = 0.0
        return self.atonia_intensity


class MetencephalicPons:
    """Pontine integration complex."""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.locus_coeruleus = LocusCoeruleus()
        self.sublaterodorsal = SublaterodorsalNucleus()
        self.corticopontine_buffer = np.zeros(feature_dim, dtype=np.float32)

    def relay_to_cerebellum(self, cortical_motor_copy: np.ndarray) -> np.ndarray:
        """Relay efference copies through middle cerebellar peduncle into cerebellar cortex."""
        arr = np.asarray(cortical_motor_copy, dtype=np.float32)
        if arr.size < self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        self.corticopontine_buffer = arr[:self.feature_dim].copy()
        return self.corticopontine_buffer
