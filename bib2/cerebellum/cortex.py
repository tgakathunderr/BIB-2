"""Cerebellar Cortex: Granule cells, Purkinje cells, and LTD Plasticity."""

import numpy as np


class CerebellarCortex:
    """
    Three-layered Cerebellar Cortex.
    1. Granular layer: Massive sparse expansion of mossy fibers onto parallel fibers.
    2. Purkinje cell layer: Monolayer of Purkinje neurons; sole cortical GABAergic output.
    3. Molecular layer: Parallel fibers synapsing on Purkinje dendrites; climbing fiber contacts.
    """

    def __init__(self, n_mossy: int = 64, n_granule: int = 256, n_purkinje: int = 64, seed: int = 42):
        self.n_mossy = n_mossy
        self.n_granule = n_granule
        self.n_purkinje = n_purkinje

        rng = np.random.default_rng(seed)
        # Mossy -> Granule sparse projection matrix
        self.weights_mossy_granule = (rng.uniform(-0.5, 0.5, (n_mossy, n_granule)) * (rng.uniform(0, 1, (n_mossy, n_granule)) < 0.3)).astype(np.float32)

        # Parallel -> Purkinje synaptic weights (subject to LTD)
        self.weights_parallel_purkinje = rng.uniform(0.4, 0.6, (n_granule, n_purkinje)).astype(np.float32)

        self.last_parallel_spikes = np.zeros(n_granule, dtype=np.float32)
        self.last_purkinje_spikes = np.zeros(n_purkinje, dtype=np.float32)

    def step_simple_spikes(self, mossy_input: np.ndarray) -> np.ndarray:
        """
        Compute high-frequency (10-50 Hz) simple spikes in Purkinje cells
        driven by parallel fibers with basket/stellate feedforward inhibition.
        """
        arr = np.asarray(mossy_input, dtype=np.float32)
        if arr.size < self.n_mossy:
            padded = np.zeros(self.n_mossy, dtype=np.float32)
            padded[:arr.size] = arr.flat
            arr = padded
        else:
            arr = arr[:self.n_mossy]

        # 1. Granule cell activation via mossy fibers
        granule_drive = np.dot(arr, self.weights_mossy_granule)
        # Non-linear thresholding representing high-threshold granule cells
        self.last_parallel_spikes = np.maximum(0.0, np.tanh(granule_drive)).astype(np.float32)

        # 2. Parallel fiber drive onto Purkinje dendrites
        purkinje_drive = np.dot(self.last_parallel_spikes, self.weights_parallel_purkinje)

        # 3. Basket/Stellate interneuron feedforward/lateral inhibition
        lateral_inhibition = np.mean(self.last_parallel_spikes) * 0.3
        effective_drive = np.maximum(0.0, purkinje_drive - lateral_inhibition)

        # Simple spikes (graded firing rate 10-50 Hz normalized)
        self.last_purkinje_spikes = np.clip(effective_drive * 1.5, 0.0, 1.0).astype(np.float32)
        return self.last_purkinje_spikes

    def apply_climbing_ltd(self, mossy_input: np.ndarray, climbing_error: np.ndarray, lr: float = 0.05) -> None:
        """
        Long-Term Depression (LTD) at Parallel Fiber - Purkinje Cell Synapses.
        Simultaneous activation of parallel fibers and climbing fiber complex spikes
        elevates intracellular Ca2+, activating PKC to internalize AMPA receptors.
        """
        # Ensure latest parallel activations are recorded
        self.step_simple_spikes(mossy_input)

        c_err = np.asarray(climbing_error, dtype=np.float32)
        if c_err.size < self.n_purkinje:
            padded = np.zeros(self.n_purkinje, dtype=np.float32)
            padded[:c_err.size] = c_err.flat
            c_err = padded
        else:
            c_err = c_err[:self.n_purkinje]

        # Heterosynaptic LTD update rule: delta_W = -lr * (parallel_activation outer climbing_error)
        outer_product = np.outer(self.last_parallel_spikes, np.maximum(0.0, c_err))
        self.weights_parallel_purkinje -= lr * outer_product

        # Enforce non-negative AMPA conductance bounds [0.01, 1.0]
        np.clip(self.weights_parallel_purkinje, 0.01, 1.0, out=self.weights_parallel_purkinje)
