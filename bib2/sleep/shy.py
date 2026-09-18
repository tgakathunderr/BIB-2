"""Synaptic Homeostasis Hypothesis (Tononi SHY): Synaptic Downscaling and Pruning."""

import numpy as np


class TononiSHYDownscaling:
    """
    Tononi & Cirelli Synaptic Homeostasis Hypothesis (SHY).
    During Slow-Wave Sleep (SWS), the brain executes global, proportional synaptic downscaling:
    - Protects metabolic energy expenditure and space.
    - Preserves relative synaptic weight ratios (memories remain intact).
    - Weak, sub-threshold synapses are pruned to zero, restoring signal-to-noise ratio.
    """

    def __init__(self, prune_threshold: float = 0.08):
        self.prune_threshold = prune_threshold

    def downscale(self, synaptic_weights: np.ndarray, downscale_factor: float = 0.85) -> np.ndarray:
        """
        Apply global proportional downscaling and threshold pruning.
        W_new = W_old * factor; prune if W_new < threshold.
        """
        arr = np.asarray(synaptic_weights, dtype=np.float32)
        scaled = arr * float(downscale_factor)
        # Prune sub-threshold weak synapses
        pruned = np.where(scaled < self.prune_threshold, 0.0, scaled)
        return pruned.astype(np.float32)
