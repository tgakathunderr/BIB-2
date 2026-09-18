"""Autonomous Agent RL Adapter: Observation ingestion, dopamine reward / pain, and action gating."""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .base import BaseNeuralAdapter

if TYPE_CHECKING:
    from ..brain import BIB2NervousSystem


class AgentAdapter(BaseNeuralAdapter):
    """
    Adapter for autonomous reinforcement learning agents:
    - Observation vector -> Sensory dermatomes and cranial channels
    - Reward (+ / -) -> Dopamine striatal plasticity & autonomic sympathetic arousal
    - Action selection -> Basal ganglia gating & primary motor outputs
    """

    def __init__(
        self,
        brain: "BIB2NervousSystem",
        obs_dim: Optional[int] = None,
        n_actions: Optional[int] = None,
        observation_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
    ):
        super().__init__(brain)
        self.obs_dim = obs_dim or observation_dim or 16
        self.n_actions = n_actions or action_dim or 4
        self.last_action: int = 0

    def act(self, obs: np.ndarray, reward: float = 0.0) -> int:
        """
        Process reinforcement feedback from previous step, ingest observation,
        execute brain clock cycle, and select discrete action.
        """
        # 1. Process reinforcement feedback & plasticity
        if reward > 0.0:
            # Positive RPE: reinforce selected action via D1 Go pathway
            self.brain.basal_ganglia.reinforce_action(self.last_action, reward_rpe=reward)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine + reward * 0.15, 0.0, 1.0)
            )
            self.brain.peripheral.enteric.satiety = float(
                np.clip(self.brain.peripheral.enteric.satiety + reward * 0.1, 0.0, 1.0)
            )
        elif reward < 0.0:
            # Negative RPE / punishment: reinforce D2 NoGo pathway & trigger sympathetic surge
            self.brain.basal_ganglia.reinforce_action(self.last_action, reward_rpe=reward)
            self.brain.chemistry.matrix.state.dopamine = float(
                np.clip(self.brain.chemistry.matrix.state.dopamine - abs(reward) * 0.2, 0.05, 1.0)
            )
            self.brain.habenula.compute_anti_reward(expected_reward=0.5, received_reward=reward)
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=float(abs(reward)))

        # 2. Ingest observation into sensory channels
        obs_arr = np.asarray(obs, dtype=np.float32).flatten()
        # Map to spinal dermatome C5
        self.brain.peripheral.spinal.set_dermatome("C5", obs_arr)
        # Map to CN II (Optic)
        if obs_arr.size > 0:
            repeats = int(np.ceil(self.brain.dim / obs_arr.size))
            cn2_input = np.tile(obs_arr, repeats)[:self.brain.dim]
        else:
            cn2_input = np.zeros(self.brain.dim, dtype=np.float32)
        self.brain.peripheral.cranial.set_sensory("CN_II", cn2_input)

        # 3. Execute biological clock cycle
        self.brain.tick()

        # Ensure punishment sympathetic tone remains elevated if negative reward was given
        if reward < 0.0 and self.brain.peripheral.autonomic.sympathetic_tone <= 0.35:
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=float(abs(reward)))

        # 4. Action selection from Basal Ganglia and M1 Motor Cortex
        dlpfc = self.brain.neocortex.registry.get_area("DLPFC_WorkingMemory").l5_output
        action_proposals = dlpfc[:self.n_actions]
        winner_idx, _ = self.brain.basal_ganglia.select_action(
            action_proposals, dopamine_level=self.brain.chemistry.matrix.state.dopamine
        )
        action_idx = int(winner_idx % self.n_actions)
        self.last_action = action_idx

        return action_idx
