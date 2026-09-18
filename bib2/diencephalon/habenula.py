"""Lateral Habenula: Negative Reward Prediction Error and Disappointment Signaling."""

import numpy as np


class LateralHabenula:
    """
    Lateral Habenula (LHb).
    Signals negative reward prediction error, behavioral disappointment, and punishment.
    When expected positive outcomes fail to materialize, LHb fires, exciting the
    rostromedial tegmental nucleus (RMTg) to exert profound GABAergic inhibition over
    VTA and SNc dopaminergic neurons.
    """

    def __init__(self):
        self.anti_reward_signal: float = 0.0
        self.suppress_da: bool = False

    def compute_anti_reward(self, expected_reward: float, received_reward: float) -> float:
        """
        Compute disappointment / anti-reward:
        AntiReward = max(0.0, Expected - Received)
        """
        rpe = received_reward - expected_reward
        if rpe < 0.0:
            self.anti_reward_signal = float(np.clip(-rpe, 0.0, 1.0))
            self.suppress_da = True
        else:
            self.anti_reward_signal = 0.0
            self.suppress_da = False

        return self.anti_reward_signal

    def should_suppress_dopamine(self) -> bool:
        """Returns True if VTA/SNc dopaminergic release should be inhibited."""
        return self.suppress_da
