"""Cranial Nerves Interface (CN I through CN XII)."""

from typing import Dict, Optional
import numpy as np
from ..types import CranialNerveSignal

CRANIAL_NERVE_DEFINITIONS = [
    ("CN_I", "Olfactory", "SVA", "Special Visceral Afferent - smell/chemical vectors"),
    ("CN_II", "Optic", "SSA", "Special Somatic Afferent - vision retinotopic SDR"),
    ("CN_III", "Oculomotor", "GSE", "General Somatic Efferent - eye movements / pupil constriction"),
    ("CN_IV", "Trochlear", "GSE", "General Somatic Efferent - superior oblique eye rotation"),
    ("CN_V", "Trigeminal", "GSA", "General Somatic Afferent/Efferent - facial sensation & mastication"),
    ("CN_VI", "Abducens", "GSE", "General Somatic Efferent - lateral rectus eye abduction"),
    ("CN_VII", "Facial", "SVE", "Special Visceral Efferent - facial expression & speech phonemes"),
    ("CN_VIII", "Vestibulocochlear", "SSA", "Special Somatic Afferent - tonotopic hearing & vestibular balance"),
    ("CN_IX", "Glossopharyngeal", "SVE", "Special/General Visceral - swallowing & carotid baroreceptors"),
    ("CN_X", "Vagus", "GVE", "General Visceral Efferent/Afferent - heart/gut parasympathetic conduit"),
    ("CN_XI", "Accessory", "SVE", "Special Visceral Efferent - head rotation & neck shoulder elevation"),
    ("CN_XII", "Hypoglossal", "GSE", "General Somatic Efferent - tongue movement & motor speech articulation"),
]


class CranialNervesInterface:
    """Manages bilateral cranial nerve sensory ingest and motor efference."""

    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.nerves: Dict[str, CranialNerveSignal] = {}
        self.motor_efferent: Dict[str, np.ndarray] = {}

        for cid, name, mod, desc in CRANIAL_NERVE_DEFINITIONS:
            self.nerves[cid] = CranialNerveSignal(
                nerve_id=cid,
                modality=mod,
                data=np.zeros(feature_dim, dtype=np.float32)
            )
            self.motor_efferent[cid] = np.zeros(feature_dim, dtype=np.float32)

    def set_sensory(self, nerve_id: str, data: np.ndarray) -> None:
        """Inject sensory afferent data into a cranial nerve."""
        if nerve_id not in self.nerves:
            raise KeyError(f"Unknown cranial nerve: {nerve_id}")
        data_arr = np.asarray(data, dtype=np.float32)
        if data_arr.shape[0] != self.feature_dim:
            if data_arr.size == 0:
                data_arr = np.zeros(self.feature_dim, dtype=np.float32)
            elif data_arr.size < self.feature_dim:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[:data_arr.size] = data_arr.flat
                data_arr = padded
            else:
                data_arr = data_arr.flat[:self.feature_dim]
        self.nerves[nerve_id].data = data_arr

    def get_sensory(self, nerve_id: str) -> np.ndarray:
        """Retrieve sensory afferent data from a cranial nerve."""
        if nerve_id not in self.nerves:
            raise KeyError(f"Unknown cranial nerve: {nerve_id}")
        return self.nerves[nerve_id].data

    def set_motor(self, nerve_id: str, data: np.ndarray) -> None:
        """Set motor efferent command for a cranial nerve."""
        if nerve_id not in self.motor_efferent:
            raise KeyError(f"Unknown cranial nerve: {nerve_id}")
        data_arr = np.asarray(data, dtype=np.float32)
        if data_arr.shape[0] != self.feature_dim:
            padded = np.zeros(self.feature_dim, dtype=np.float32)
            sz = min(data_arr.size, self.feature_dim)
            padded[:sz] = data_arr.flat[:sz]
            data_arr = padded
        self.motor_efferent[nerve_id] = data_arr

    def get_motor(self, nerve_id: str) -> np.ndarray:
        """Retrieve motor efferent command from a cranial nerve."""
        if nerve_id not in self.motor_efferent:
            raise KeyError(f"Unknown cranial nerve: {nerve_id}")
        return self.motor_efferent[nerve_id]

    def reset_afferents(self) -> None:
        """Clear incoming sensory buffer after ingestion cycle."""
        for cid in self.nerves:
            self.nerves[cid].data.fill(0.0)
