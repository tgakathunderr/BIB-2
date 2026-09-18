"""Basal Ganglia Tripartite Gating Circuits: Direct, Indirect, and Hyperdirect Pathways."""

from typing import Tuple
import numpy as np


class BasalGangliaComplex:
    """
    Basal Ganglia Action Selection Engine.
    At rest: GPi/SNr output nuclei fire tonically, inhibiting thalamocortical action channels.
    Selection:
    1. Direct Pathway (D1 MSNs): Pro-kinetic disinhibition of selected action channel.
    2. Indirect Pathway (D2 MSNs): Anti-kinetic inhibition of competing channels via GPe and STN.
    3. Hyperdirect Pathway (STN): Rapid cortical global stop brake.
    4. SNc Dopaminergic Modulation: Pro-kinetic gain control facilitating D1 and suppressing D2.
    """

    def __init__(self, n_actions: int = 8):
        self.n_actions = n_actions
        self.d1_weights = np.ones(n_actions, dtype=np.float32)
        self.d2_weights = np.ones(n_actions, dtype=np.float32)
        self.hyperdirect_active: bool = False
        self.baseline_gpi_snr_tonic: float = 1.0

    def trigger_hyperdirect_stop(self) -> None:
        """Cortical motor pyramidal axons activate STN to rapidly slam on the emergency brake."""
        self.hyperdirect_active = True

    def reset_hyperdirect_stop(self) -> None:
        """Release emergency brake."""
        self.hyperdirect_active = False

    def select_action(self, cortical_proposals: np.ndarray, dopamine_level: float = 0.5) -> Tuple[int, float]:
        """
        Evaluate competing action proposals.
        Returns: (winning_action_index, thalamic_disinhibition_amplitude).
        """
        proposals = np.asarray(cortical_proposals, dtype=np.float32)
        if proposals.size < self.n_actions:
            padded = np.zeros(self.n_actions, dtype=np.float32)
            padded[:proposals.size] = proposals.flat
            proposals = padded
        else:
            proposals = proposals[:self.n_actions]

        # Check Hyperdirect emergency brake
        if self.hyperdirect_active:
            # GPi/SNr excited to maximum across all channels; zero thalamic disinhibition
            self.hyperdirect_active = False
            best_idx = int(np.argmax(proposals))
            return best_idx, 0.0

        # Dopamine modulation factor:
        # High DA excites D1 (direct) and suppresses D2 (indirect)
        da_norm = float(np.clip(dopamine_level, 0.05, 1.0))
        d1_mod = da_norm * 1.5
        d2_mod = (1.05 - da_norm) * 1.5

        # 1. Direct pathway (Go) drive
        direct_drive = proposals * self.d1_weights * d1_mod

        # 2. Indirect pathway (NoGo) suppression
        indirect_drive = proposals * self.d2_weights * d2_mod

        # Net action preference
        net_preference = direct_drive - indirect_drive * 0.4

        # Selected winning channel
        winner_idx = int(np.argmax(net_preference))

        # Disinhibition calculation for winning channel:
        # GPi/SNr output = Baseline Tonic - Direct Inhibition + Indirect Excitation
        winner_direct = float(direct_drive[winner_idx])
        winner_indirect = float(indirect_drive[winner_idx])
        gpi_output = max(0.0, self.baseline_gpi_snr_tonic - winner_direct + winner_indirect * 0.3)

        # Thalamic disinhibition is inverse of GPi/SNr output
        disinhibition = float(np.clip(1.0 - gpi_output, 0.0, 1.0))

        return winner_idx, disinhibition

    def reinforce_action(self, action_idx: int, reward_rpe: float, lr: float = 0.1) -> None:
        """
        Dopamine-dependent corticostriatal synaptic plasticity.
        Positive RPE strengthens D1 (Go) weights; Negative RPE strengthens D2 (NoGo) weights.
        """
        if 0 <= action_idx < self.n_actions:
            if reward_rpe > 0:
                self.d1_weights[action_idx] += lr * reward_rpe
            else:
                self.d2_weights[action_idx] += lr * abs(reward_rpe)
