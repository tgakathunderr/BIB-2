"""Cerebellum Engine and Forward Predictive Motor Calibration."""

from typing import Tuple
import numpy as np
from .cortex import CerebellarCortex
from .nuclei import DeepCerebellarNuclei


class CerebellumEngine:
    """
    Cerebellar Internal Forward Model (Smith Predictor).
    Computes rapid sensorimotor predictions and real-time micro-corrections
    to compensate for sensorimotor conduction delays.
    """

    def __init__(self, n_mossy: int = 64, n_granule: int = 256, n_purkinje: int = 64, n_deep: int = 16, seed: int = 42):
        self.feature_dim = n_mossy
        self.cortex = CerebellarCortex(n_mossy=n_mossy, n_granule=n_granule, n_purkinje=n_purkinje, seed=seed)
        self.deep_nuclei = DeepCerebellarNuclei(n_purkinje=n_purkinje, n_deep=n_deep)

        rng = np.random.default_rng(seed)
        # Forward model expansion from deep nuclei back to motor feature space
        self.weights_deep_forward = rng.uniform(-0.4, 0.4, (n_deep, n_mossy)).astype(np.float32)

    def step(self, intended_motor_command: np.ndarray, current_sensory_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute forward predictive loop.
        Inputs: Intended cortical motor command + current peripheral sensory telemetry.
        Outputs: (predicted_future_state, micro_correction_vector).
        """
        intended = np.asarray(intended_motor_command, dtype=np.float32)
        sensory = np.asarray(current_sensory_state, dtype=np.float32)

        if intended.size < self.feature_dim:
            pad = np.zeros(self.feature_dim, dtype=np.float32)
            pad[:intended.size] = intended.flat
            intended = pad
        else:
            intended = intended[:self.feature_dim]

        if sensory.size < self.feature_dim:
            pad = np.zeros(self.feature_dim, dtype=np.float32)
            pad[:sensory.size] = sensory.flat
            sensory = pad
        else:
            sensory = sensory[:self.feature_dim]

        # 1. Mossy fibers convey motor efference copy + sensory state
        mossy_composite = (intended * 0.6 + sensory * 0.4).astype(np.float32)

        # 2. Cerebellar cortex computes Purkinje simple spikes
        purkinje_spikes = self.cortex.step_simple_spikes(mossy_composite)

        # 3. Deep cerebellar nuclei integrate Purkinje inhibition
        deep_out = self.deep_nuclei.step(purkinje_spikes)

        # 4. Synthesize predicted state and micro-correction vector
        micro_correction = np.dot(deep_out, self.weights_deep_forward).astype(np.float32)
        predicted_state = np.tanh(sensory + intended + micro_correction).astype(np.float32)

        return predicted_state, micro_correction

    def calibrate_with_error(self, mossy_input: np.ndarray, error_feedback: np.ndarray, lr: float = 0.05) -> None:
        """Calibrate internal forward model using climbing fiber error signals."""
        self.cortex.apply_climbing_ltd(mossy_input, error_feedback, lr=lr)
