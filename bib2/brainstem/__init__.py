"""Brainstem Complex for BIB 2 (Medulla, Pons, and Midbrain)."""

from typing import Any, Dict
import numpy as np
from .medulla import MedullaOblongata, PreBotzingerComplex, InferiorOlive
from .pons import MetencephalicPons, LocusCoeruleus, SublaterodorsalNucleus
from .midbrain import Mesencephalon, SuperiorColliculus, InferiorColliculus, PeriaqueductalGray, RedNucleus


class BrainstemComplex:
    """Master Brainstem coordinator integrating Medulla, Pons, and Midbrain."""

    def __init__(self, feature_dim: int = 64):
        self.medulla = MedullaOblongata(feature_dim=feature_dim)
        self.pons = MetencephalicPons(feature_dim=feature_dim)
        self.midbrain = Mesencephalon(feature_dim=feature_dim)

    def step(self, bp_sys: float, visual_stim: np.ndarray, is_rem: bool = False, dt: float = 1.0) -> Dict[str, Any]:
        """Execute one brainstem automation cycle."""
        resp_burst = self.medulla.pre_botzinger.step(dt=0.1 * dt)
        symp, parasymp = self.medulla.process_baroreflex(bp_sys)
        ne_level = self.pons.locus_coeruleus.step_arousal(salience=float(np.mean(visual_stim)), dt=dt)
        atonia = self.pons.sublaterodorsal.compute_motor_inhibition(is_rem)
        saccade = self.midbrain.superior_colliculus.compute_saccade(visual_stim)

        return {
            "respiratory_burst": resp_burst,
            "baroreflex": (symp, parasymp),
            "ne_level": ne_level,
            "rem_atonia": atonia,
            "saccade_target": saccade,
        }


__all__ = [
    "BrainstemComplex",
    "MedullaOblongata",
    "PreBotzingerComplex",
    "InferiorOlive",
    "MetencephalicPons",
    "LocusCoeruleus",
    "SublaterodorsalNucleus",
    "Mesencephalon",
    "SuperiorColliculus",
    "InferiorColliculus",
    "PeriaqueductalGray",
    "RedNucleus",
]
