"""Core Data Structures & Universal Neural Bus State for BIB 2."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class CranialNerveSignal:
    """Represents afferent/efferent signals for one of the 12 cranial nerves."""
    nerve_id: str
    modality: str
    data: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))


@dataclass
class SpinalNerveSignal:
    """Represents segmental sensory (dermatome) and motor (myotome) signals."""
    segment: str
    dermatome: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=np.float32))
    myotome: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))


@dataclass
class AutonomicState:
    """Physiological visceral variables regulated by sympathetic and parasympathetic trunks."""
    heart_rate: float = 70.0
    blood_pressure_sys: float = 120.0
    blood_pressure_dia: float = 80.0
    pupil_diameter: float = 3.5
    sympathetic_tone: float = 0.3
    parasympathetic_tone: float = 0.7
    respiratory_rate: float = 14.0


@dataclass
class ChemicalState:
    """Global neuromodulator concentrations across the neuroaxis."""
    dopamine: float = 0.5
    serotonin: float = 0.5
    norepinephrine: float = 0.4
    acetylcholine: float = 0.6
    histamine: float = 0.5
    cortisol: float = 0.1
    adenosine: float = 0.0


@dataclass
class TractData:
    """Signal packet transmitted through an anatomical white-matter tract."""
    name: str
    origin: str
    destination: str
    signal: np.ndarray = field(default_factory=lambda: np.zeros(64, dtype=np.float32))
    conductance: float = 1.0


@dataclass
class NeuralBusState:
    """Complete peripheral state exchanged between external adapters and CNS."""
    cranial_nerves: Dict[str, CranialNerveSignal] = field(default_factory=dict)
    spinal_nerves: Dict[str, SpinalNerveSignal] = field(default_factory=dict)
    autonomic: AutonomicState = field(default_factory=AutonomicState)
    enteric_satiety: float = 0.7
    enteric_motility: float = 0.5

    def __post_init__(self):
        cranial_ids = [
            ("CN_I", "SVA"), ("CN_II", "SSA"), ("CN_III", "GSE"), ("CN_IV", "GSE"),
            ("CN_V", "GSA"), ("CN_VI", "GSE"), ("CN_VII", "SVE"), ("CN_VIII", "SSA"),
            ("CN_IX", "SVE"), ("CN_X", "GVE"), ("CN_XI", "SVE"), ("CN_XII", "GSE")
        ]
        for cid, mod in cranial_ids:
            if cid not in self.cranial_nerves:
                self.cranial_nerves[cid] = CranialNerveSignal(nerve_id=cid, modality=mod)

        segments = (
            [f"C{i}" for i in range(1, 9)] +
            [f"T{i}" for i in range(1, 13)] +
            [f"L{i}" for i in range(1, 6)] +
            [f"S{i}" for i in range(1, 6)] +
            ["Co1"]
        )
        for seg in segments:
            if seg not in self.spinal_nerves:
                self.spinal_nerves[seg] = SpinalNerveSignal(segment=seg)
