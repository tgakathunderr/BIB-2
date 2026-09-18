"""Deep Cerebellar Nuclei: Dentate, Interposed, and Fastigial."""

import numpy as np


class DeepCerebellarNuclei:
    """
    Deep Cerebellar Nuclei (Dentate, Interposed [Emboliform/Globose], Fastigial).
    Sole output conduits of the cerebellum.
    Operate via baseline tonic firing that is modulated/sculpted by Purkinje GABAergic inhibition.
    """

    def __init__(self, n_purkinje: int = 64, n_deep: int = 16):
        self.n_purkinje = n_purkinje
        self.n_deep = n_deep
        self.baseline_tonic: float = 0.8

        # Projection matrix from Purkinje cells to Deep Nuclei
        rng = np.random.default_rng(42)
        self.weights_purkinje_deep = rng.uniform(0.3, 0.7, (n_purkinje, n_deep)).astype(np.float32)

    def step(self, purkinje_spikes: np.ndarray) -> np.ndarray:
        """
        Transform Purkinje GABAergic inhibition into calibrated deep nuclei output.
        High Purkinje activity -> suppression; Low Purkinje activity -> disinhibition.
        """
        p_spikes = np.asarray(purkinje_spikes, dtype=np.float32)
        if p_spikes.size < self.n_purkinje:
            padded = np.zeros(self.n_purkinje, dtype=np.float32)
            padded[:p_spikes.size] = p_spikes.flat
            p_spikes = padded
        else:
            p_spikes = p_spikes[:self.n_purkinje]

        # Total inhibitory drive onto deep nuclei
        gaba_inhibition = np.dot(p_spikes, self.weights_purkinje_deep) / max(1, self.n_purkinje * 0.1)

        # Output = Baseline Tonic - Purkinje Inhibition
        calibrated_output = np.maximum(0.0, self.baseline_tonic - gaba_inhibition).astype(np.float32)
        return calibrated_output
