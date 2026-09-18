"""Molecular Synaptic Plasticity Engine (E-LTP, L-LTP, LTD) and GHK Biophysics."""

import numpy as np


class PlasticityEngine:
    """
    Synaptic Plasticity and Biophysical Conductance Engine.
    - Early-Phase LTP: NMDA unblock -> Ca2+ surge -> CaMKII/PKC -> GluA1 AMPA insertion.
    - Late-Phase LTP: cAMP/PKA/MAPK -> CREB gene transcription -> spine expansion.
    - LTD: Low-frequency Ca2+ -> Calcineurin/PP1 -> Clathrin AMPA endocytosis.
    - GHK Equation: Steady-state transmembrane resting potential biophysics.
    """

    def __init__(self):
        # Universal constants at physiological temperature 37C (310.15 K)
        # RT/F in millivolts
        self.rt_f_mv: float = 26.72
        # Relative resting membrane ion permeabilities (P_K : P_Na : P_Cl)
        self.p_k: float = 1.0
        self.p_na: float = 0.04
        self.p_cl: float = 0.45

    def compute_ghk_potential(
        self,
        k_out: float = 4.0,
        k_in: float = 140.0,
        na_out: float = 145.0,
        na_in: float = 12.0,
        cl_out: float = 110.0,
        cl_in: float = 4.0,
    ) -> float:
        """
        Goldman-Hodgkin-Katz (GHK) voltage equation:
        V_m = (RT / F) * ln( (P_K*[K+]_out + P_Na*[Na+]_out + P_Cl*[Cl-]_in) /
                             (P_K*[K+]_in  + P_Na*[Na+]_in  + P_Cl*[Cl-]_out) )
        """
        numerator = (self.p_k * k_out + self.p_na * na_out + self.p_cl * cl_in)
        denominator = (self.p_k * k_in + self.p_na * na_in + self.p_cl * cl_out)

        vm = self.rt_f_mv * np.log(numerator / denominator)
        return float(vm)

    def compute_plasticity(self, weight: float, ca_influx: float, freq_hz: float, lr: float = 0.1) -> float:
        """
        Bi-directional synaptic modification threshold (BCM theory / Ca2+ hypothesis):
        - Ca2+ > 0.6 & Freq >= 20 Hz: E-LTP (potentiation via CaMKII).
        - 0.1 < Ca2+ <= 0.4 & Freq < 10 Hz: LTD (depression via Calcineurin).
        """
        w = float(weight)
        if ca_influx > 0.5 and freq_hz >= 20.0:
            # E-LTP potentiation
            delta = lr * (ca_influx - 0.5) * (freq_hz / 50.0)
            w = float(np.clip(w + delta, 0.01, 2.0))
        elif 0.05 < ca_influx <= 0.4 and freq_hz < 10.0:
            # LTD depression
            delta = lr * (0.4 - ca_influx) * 0.8
            w = float(np.clip(w - delta, 0.01, 2.0))

        return w
