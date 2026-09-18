"""Chemical Matrix: Neuromodulators, Fast Transmitters, and Clearance Kinetics."""

import numpy as np
from ..types import ChemicalState


class ChemicalMatrix:
    """
    Global Neuromodulatory Chemical Matrix.
    Tracks and diffuses concentrations of:
    1. Dopamine (DA) - VTA/SNc: Reward prediction error, striatal pro-kinetic gain.
    2. Norepinephrine (NE) - Locus Coeruleus: Vigilance, SNR, astrocytic interstitial regulation.
    3. Serotonin (5-HT) - Raphe: Affective stabilization, patience, behavioral inhibition.
    4. Acetylcholine (ACh) - PPT/LDT & Basal Forebrain: Cortical arousal, REM sleep, attention.
    5. Histamine - Tuberomammillary Nucleus: Baseline wakefulness maintenance.
    6. Adenosine - Cellular ATP breakdown accumulation (Process S sleep pressure).
    """

    def __init__(self):
        self.state = ChemicalState()
        self.clearance_rate: float = 0.08

    def release_dopamine(self, amount: float) -> None:
        self.state.dopamine = float(np.clip(self.state.dopamine + amount, 0.0, 1.0))

    def release_norepinephrine(self, amount: float) -> None:
        self.state.norepinephrine = float(np.clip(self.state.norepinephrine + amount, 0.0, 1.0))

    def release_serotonin(self, amount: float) -> None:
        self.state.serotonin = float(np.clip(self.state.serotonin + amount, 0.0, 1.0))

    def release_acetylcholine(self, amount: float) -> None:
        self.state.acetylcholine = float(np.clip(self.state.acetylcholine + amount, 0.0, 1.0))

    def accumulate_adenosine(self, amount: float) -> None:
        """Accumulate homeostatic sleep pressure during active waking neural activity."""
        self.state.adenosine = float(np.clip(self.state.adenosine + amount, 0.0, 1.0))

    def clear_adenosine(self, amount: float) -> None:
        """Clear adenosine during Slow-Wave Sleep."""
        self.state.adenosine = max(0.0, self.state.adenosine - amount)

    def step_clearance(self, dt: float = 1.0) -> None:
        """Passive reuptake (DAT, NET, SERT) and enzymatic degradation (MAO, COMT, AChE)."""
        rate = self.clearance_rate * dt
        self.state.dopamine += (0.5 - self.state.dopamine) * rate
        self.state.norepinephrine += (0.4 - self.state.norepinephrine) * rate
        self.state.serotonin += (0.5 - self.state.serotonin) * rate
        self.state.acetylcholine += (0.6 - self.state.acetylcholine) * rate
        self.state.histamine += (0.5 - self.state.histamine) * rate
