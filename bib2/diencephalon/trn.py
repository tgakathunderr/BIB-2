"""Thalamic Reticular Nucleus (TRN) Gating Shell and Sleep Spindle Generator."""

import numpy as np


class ThalamicReticularNucleus:
    """
    Thalamic Reticular Nucleus (TRN).
    Inhibitory GABAergic shell that encases the thalamus.
    - Samples thalamocortical and corticothalamic collaterals without projecting to cortex.
    - Directs selective sensory attention during wakefulness.
    - Generates 11-16 Hz sleep spindles during Stage N2 sleep, severing sensory awareness.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.attentional_mask = np.ones(dim, dtype=np.float32)
        self.spindle_phase: float = 0.0
        self.spindle_frequency_hz: float = 14.0

    def set_attentional_focus(self, focus_mask: np.ndarray) -> None:
        """Set inhibitory attentional spotlight (1.0 = pass, 0.0 = suppress)."""
        arr = np.asarray(focus_mask, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.ones(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        self.attentional_mask = np.clip(arr[:self.dim], 0.0, 1.0)

    def gate_thalamic_output(self, nucleus: str, signal: np.ndarray) -> np.ndarray:
        """Apply TRN GABAergic lateral inhibition on thalamic projection to cortical Layer IV."""
        arr = np.asarray(signal, dtype=np.float32)
        if arr.size < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.dim]

        # Element-wise attentional gating
        gated = arr * self.attentional_mask
        return gated.astype(np.float32)

    def generate_sleep_spindles(self, is_nrem2: bool, dt: float = 0.05) -> np.ndarray:
        """
        Generate burst-firing sleep spindles (11-16 Hz) during Stage N2 sleep.
        Deinactivates T-type Ca2+ channels, locking relay cells in synchronized burst mode.
        """
        if not is_nrem2:
            return np.zeros(self.dim, dtype=np.float32)

        self.spindle_phase += 2.0 * np.pi * self.spindle_frequency_hz * dt
        burst_amplitude = float(np.sin(self.spindle_phase)) * 0.5 + 0.5

        # Rhythmic burst envelope across channels
        spindle_burst = np.full(self.dim, burst_amplitude, dtype=np.float32)
        return spindle_burst
