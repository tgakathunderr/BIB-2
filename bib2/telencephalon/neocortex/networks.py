"""Large-Scale Tri-Network Architecture: DMN, CEN, and Salience Network."""

from typing import Dict
import numpy as np


class TriNetworkCoordinator:
    """
    Large-Scale Tri-Network Dynamics.
    Coordinates the competitive balance between:
    1. Default Mode Network (DMN): Engaged during rest, autobiographical retrieval, prospective simulation.
    2. Central Executive Network (CEN): Engaged during externally directed, demanding problem-solving.
    3. Salience Network (SN): Anterior Insula & dACC; acts as a causal dynamic switch that disengages
       the DMN and recruits the CEN upon detecting behaviorally relevant events.
    """

    def __init__(self):
        # Baseline tonic activations
        self.dmn_activity: float = 0.85
        self.cen_activity: float = 0.15
        self.salience_activity: float = 0.2

    def step(self, salience_event: bool, dt: float = 1.0) -> Dict[str, float]:
        """
        Update Tri-Network state based on salience cues.
        Returns activation levels of DMN, CEN, and SalienceNetwork.
        """
        if salience_event:
            # Salience Network fires -> Anterior Insula switch triggers:
            # Rapid suppression of DMN, robust recruitment of CEN
            self.salience_activity = 0.90
            self.dmn_activity = max(0.1, self.dmn_activity - 0.6 * dt)
            self.cen_activity = min(1.0, self.cen_activity + 0.7 * dt)
        else:
            # Relaxation back to default mode:
            # Salience decays, DMN re-emerges, CEN idles
            self.salience_activity = max(0.15, self.salience_activity - 0.2 * dt)
            self.dmn_activity = min(0.85, self.dmn_activity + 0.15 * dt)
            self.cen_activity = max(0.15, self.cen_activity - 0.2 * dt)

        return {
            "DMN": float(self.dmn_activity),
            "CEN": float(self.cen_activity),
            "SalienceNetwork": float(self.salience_activity),
        }
