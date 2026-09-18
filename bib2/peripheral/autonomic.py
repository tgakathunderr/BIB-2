"""Autonomic Nervous System (ANS) and Enteric Nervous System (ENS)."""

import numpy as np
from ..types import AutonomicState


class AutonomicNervousSystem:
    """Regulates involuntary viscero-motor and cardiovascular homeostasis."""

    def __init__(self):
        self.state = AutonomicState()
        self.baseline_hr = 70.0
        self.baseline_bp_sys = 120.0
        self.baseline_bp_dia = 80.0
        self.baseline_pupil = 3.5

    @property
    def heart_rate(self) -> float:
        return self.state.heart_rate

    @property
    def blood_pressure_sys(self) -> float:
        return self.state.blood_pressure_sys

    @property
    def blood_pressure_dia(self) -> float:
        return self.state.blood_pressure_dia

    @property
    def pupil_diameter(self) -> float:
        return self.state.pupil_diameter

    @property
    def sympathetic_tone(self) -> float:
        return self.state.sympathetic_tone

    @property
    def parasympathetic_tone(self) -> float:
        return self.state.parasympathetic_tone

    def trigger_sympathetic_surge(self, intensity: float = 0.5) -> None:
        """Trigger acute fight-or-flight catabolic surge."""
        intensity = float(np.clip(intensity, 0.0, 1.0))
        self.state.sympathetic_tone = float(np.clip(self.state.sympathetic_tone + intensity * 0.5, 0.0, 1.0))
        self.state.parasympathetic_tone = float(np.clip(self.state.parasympathetic_tone - intensity * 0.4, 0.0, 1.0))

        # Tachycardia, hypertension, mydriasis (pupil dilation)
        self.state.heart_rate += intensity * 35.0
        self.state.blood_pressure_sys += intensity * 30.0
        self.state.blood_pressure_dia += intensity * 15.0
        self.state.pupil_diameter = float(np.clip(self.state.pupil_diameter + intensity * 2.5, 1.5, 8.0))
        self.state.respiratory_rate += intensity * 8.0

    def trigger_vagal_tone(self, intensity: float = 0.5) -> None:
        """Trigger parasympathetic rest-and-digest anabolic response via the Vagus nerve."""
        intensity = float(np.clip(intensity, 0.0, 1.0))
        self.state.parasympathetic_tone = float(np.clip(self.state.parasympathetic_tone + intensity * 0.4, 0.0, 1.0))
        self.state.sympathetic_tone = float(np.clip(self.state.sympathetic_tone - intensity * 0.3, 0.0, 1.0))

        # Bradycardia, normotension, miosis (pupil constriction)
        self.state.heart_rate = max(45.0, self.state.heart_rate - intensity * 20.0)
        self.state.blood_pressure_sys = max(90.0, self.state.blood_pressure_sys - intensity * 15.0)
        self.state.pupil_diameter = float(np.clip(self.state.pupil_diameter - intensity * 1.5, 1.5, 8.0))

    def step_homeostasis(self, dt: float = 1.0) -> None:
        """Passive homeostatic relaxation back to setpoints."""
        rate = 0.1 * dt
        self.state.heart_rate += (self.baseline_hr - self.state.heart_rate) * rate
        self.state.blood_pressure_sys += (self.baseline_bp_sys - self.state.blood_pressure_sys) * rate
        self.state.blood_pressure_dia += (self.baseline_bp_dia - self.state.blood_pressure_dia) * rate
        self.state.pupil_diameter += (self.baseline_pupil - self.state.pupil_diameter) * rate
        self.state.sympathetic_tone += (0.3 - self.state.sympathetic_tone) * rate
        self.state.parasympathetic_tone += (0.7 - self.state.parasympathetic_tone) * rate


class EntericNervousSystem:
    """Autonomous gut-brain axis regulating peristalsis, satiety, and mucosal secretions."""

    def __init__(self):
        self.satiety: float = 0.7
        self.motility: float = 0.5
        self.nutrient_reserves: float = 0.8

    def ingest_nutrients(self, amount: float = 0.3) -> None:
        """Ingest metabolic substrates; raises satiety and peristaltic motility."""
        self.nutrient_reserves = float(np.clip(self.nutrient_reserves + amount, 0.0, 1.0))
        self.satiety = float(np.clip(self.satiety + amount * 0.6, 0.0, 1.0))
        self.motility = float(np.clip(self.motility + amount * 0.4, 0.0, 1.0))

    def step(self, dt: float = 1.0) -> None:
        """Metabolic decay over time."""
        decay = 0.02 * dt
        self.satiety = max(0.0, self.satiety - decay)
        self.motility = float(np.clip(self.motility + (0.5 - self.motility) * 0.05 * dt, 0.1, 1.0))
