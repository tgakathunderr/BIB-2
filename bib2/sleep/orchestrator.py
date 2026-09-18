"""Two-Process Sleep Model and Stage Transitions."""

from enum import Enum
import numpy as np


class SleepStage(Enum):
    WAKE = "Wake"
    NREM_N1_N2 = "NREM_Light_Spindles"
    NREM_SWS = "NREM_SlowWaveSleep"
    REM = "RapidEyeMovement"


class SleepOrchestrator:
    """
    Two-Process Model of Sleep Regulation (Borbely).
    - Process S: Homeostatic sleep drive driven by adenosine accumulation from daytime synaptic computation.
    - Process C: Circadian rhythm driven by the suprachiasmatic nucleus (SCN).
    Sleep gates open when homeostatic sleep pressure exceeds circadian alertness.
    """

    def __init__(self):
        self.current_stage: SleepStage = SleepStage.WAKE
        self.process_s: float = 0.1
        self.process_c: float = 0.5
        self.sleep_threshold: float = 0.65

    def update_processes(self, adenosine_pressure: float, circadian_phase: float) -> None:
        """Update Process S and Process C values."""
        self.process_s = float(np.clip(adenosine_pressure, 0.0, 1.0))
        # Circadian alertness peaks at noon (~0.5), drops at night (~0.8 to 1.0 / 0.0 to 0.2)
        alertness = float(np.cos(2.0 * np.pi * (circadian_phase - 0.5)) * 0.5 + 0.5)
        self.process_c = alertness

    def should_sleep(self) -> bool:
        """Returns True if homeostatic drive overcomes circadian alertness."""
        net_somnogenic_drive = self.process_s + (1.0 - self.process_c) * 0.4
        return net_somnogenic_drive >= self.sleep_threshold

    def transition_to(self, stage: SleepStage) -> None:
        """Explicit sleep stage transition."""
        self.current_stage = stage
