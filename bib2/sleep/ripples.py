"""Sharp-Wave Ripples (SWRs): Compressed Replay & Memory Consolidation."""

from typing import List
import numpy as np


class SharpWaveRippleEngine:
    """
    Hippocampal Sharp-Wave Ripples (SWRs, 150-250 Hz).
    Occurs during Slow-Wave Sleep (NREM SWS).
    High-frequency ripple bursts replay daytime experiential trajectories in temporally
    compressed form, synchronizing with thalamic spindles and neocortical slow waves
    to transfer episodic traces into long-term neocortical networks.
    """

    def __init__(self, replay_frequency_hz: float = 200.0):
        self.replay_frequency_hz = float(np.clip(replay_frequency_hz, 150.0, 250.0))
        self.consolidation_gain: float = 1.25

    def replay_episodes(self, episode_history: List[np.ndarray]) -> List[np.ndarray]:
        """
        Replay experiential episodes at high-frequency ripple burst velocity.
        Applies replay gain and stabilizes memory representations.
        """
        replayed = []
        for ep in episode_history:
            arr = np.asarray(ep, dtype=np.float32)
            # Replay compression and consolidation sharpening
            sharpened = np.tanh(arr * self.consolidation_gain)
            replayed.append(sharpened.astype(np.float32))
        return replayed
