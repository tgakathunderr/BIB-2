"""Hypothalamic Complex: Circadian SCN, Metabolic Drives, and Neuroendocrine Control."""

from typing import Tuple
import numpy as np


class SCNClock:
    """
    Suprachiasmatic Nucleus (SCN).
    Master endogenous circadian pacemaker driven by CLOCK/BMAL1 transcription loops.
    Synchronized by retinohypothalamic tract light inputs.
    """

    def __init__(self):
        self.phase: float = 0.25  # 0.0 to 1.0 (0.25 = morning, 0.5 = noon, 0.75 = evening, 1.0 = midnight)
        self.period_hours: float = 24.0

    def step_circadian(self, dt_hours: float = 1.0) -> float:
        """Advance circadian oscillator by dt_hours."""
        self.phase = (self.phase + (dt_hours / self.period_hours)) % 1.0
        return self.phase

    def get_melatonin_drive(self) -> float:
        """Melatonin release peaks during biological night (phase > 0.75 or phase < 0.25)."""
        # Cosine peak at midnight (phase = 0.0 or 1.0)
        drive = float(np.cos(2.0 * np.pi * self.phase) * 0.5 + 0.5)
        return drive


class ArcuateNucleus:
    """
    Arcuate Nucleus.
    Primary metabolic sensor balancing:
    - Orexigenic neurons: Neuropeptide Y (NPY) and Agouti-Related Peptide (AgRP) -> hunger.
    - Anorexigenic neurons: Pro-opiomelanocortin (POMC) and CART -> satiety.
    """

    def __init__(self):
        self.npy_agrp_drive: float = 0.5
        self.pomc_cart_drive: float = 0.5

    def evaluate_metabolism(self, glucose: float, leptin: float) -> Tuple[float, float]:
        """
        Evaluate circulating metabolic signals.
        Returns: (hunger_drive, satiety_drive).
        """
        # Low glucose/leptin excites NPY/AgRP hunger
        self.npy_agrp_drive = float(np.clip(1.0 - (glucose * 0.6 + leptin * 0.4), 0.0, 1.0))
        # High glucose/leptin excites POMC/CART satiety
        self.pomc_cart_drive = float(np.clip(glucose * 0.5 + leptin * 0.5, 0.0, 1.0))
        return self.npy_agrp_drive, self.pomc_cart_drive


class HypothalamicComplex:
    """Master Hypothalamic complex coordinating homeostasis, drives, and circadian rhythms."""

    def __init__(self):
        self.scn = SCNClock()
        self.arcuate = ArcuateNucleus()
        self.orexin_level: float = 0.8  # Wakefulness stabilization from Lateral Hypothalamus
        self.body_temp_celsius: float = 37.0

    def step(self, glucose: float, leptin: float, dt_hours: float = 0.1) -> Tuple[float, float, float]:
        """Advance hypothalamic state."""
        phase = self.scn.step_circadian(dt_hours)
        hunger, satiety = self.arcuate.evaluate_metabolism(glucose, leptin)
        # Orexin is higher during waking daytime
        melatonin = self.scn.get_melatonin_drive()
        self.orexin_level = float(np.clip(1.0 - melatonin * 0.7, 0.1, 1.0))
        return phase, hunger, satiety
