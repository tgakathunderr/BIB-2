"""Glymphatic System: State-Dependent Convective Fluid Bulk Flow and Waste Clearance."""

import numpy as np


class GlymphaticEngine:
    """
    Glymphatic Convective Waste Clearance System.
    - Wakefulness: High locus coeruleus noradrenergic tone expands astrocytic volume,
      compressing extracellular interstitial space to ~14% of brain volume.
    - Slow-Wave Sleep (SWS): Noradrenergic silence expands interstitial space by ~60% (to ~23%).
    - Astrocytic Aquaporin-4 (AQP4) water channels drive convective bulk CSF-ISF exchange,
      rapidly flushing soluble metabolic waste (amyloid-beta, tau, alpha-synuclein).
    """

    def __init__(self):
        self.baseline_interstitial_wake = 0.14
        self.baseline_interstitial_sws = 0.23

    def compute_interstitial_fraction(self, ne_level: float) -> float:
        """Calculate extracellular interstitial space fraction as a function of noradrenergic tone."""
        ne = float(np.clip(ne_level, 0.0, 1.0))
        # Inverse linear relationship with NE
        fraction = self.baseline_interstitial_sws - ne * (self.baseline_interstitial_sws - self.baseline_interstitial_wake)
        return float(np.clip(fraction, 0.12, 0.25))

    def flush_parenchyma(self, current_solute_load: float, is_sws: bool, dt: float = 1.0) -> float:
        """
        Simulate AQP4-dependent convective bulk flow clearance of toxic metabolic solutes.
        Clearance rate is dramatically accelerated during Slow-Wave Sleep.
        """
        solute = max(0.0, float(current_solute_load))
        if is_sws:
            # Accelerated convective wash during SWS (~60% clearance rate)
            clearance_rate = 0.60 * dt
        else:
            # Minimal clearance during waking state (~5% clearance rate)
            clearance_rate = 0.05 * dt

        remaining_solute = max(0.0, solute * (1.0 - clearance_rate))
        return float(remaining_solute)
