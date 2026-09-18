import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.types import (
    CranialNerveSignal,
    SpinalNerveSignal,
    AutonomicState,
    ChemicalState,
    TractData,
    NeuralBusState,
)

def test_cranial_nerve_signal_defaults():
    sig = CranialNerveSignal(nerve_id="CN_II", modality="SSA", data=np.ones(64, dtype=np.float32))
    assert sig.nerve_id == "CN_II"
    assert sig.modality == "SSA"
    assert sig.data.shape == (64,)
    assert np.allclose(sig.data, 1.0)

def test_spinal_nerve_signal():
    sig = SpinalNerveSignal(segment="C5", dermatome=np.zeros(16, dtype=np.float32), myotome=np.ones(8, dtype=np.float32))
    assert sig.segment == "C5"
    assert sig.dermatome.shape == (16,)
    assert sig.myotome.shape == (8,)
    assert np.allclose(sig.myotome, 1.0)

def test_autonomic_and_chemical_state():
    auto = AutonomicState(heart_rate=72.0, pupil_diameter=3.5, sympathetic_tone=0.2, parasympathetic_tone=0.8)
    chem = ChemicalState(dopamine=0.5, serotonin=0.5, norepinephrine=0.4, acetylcholine=0.6, histamine=0.5, cortisol=0.1)
    assert auto.heart_rate == 72.0
    assert auto.sympathetic_tone == pytest.approx(0.2)
    assert chem.dopamine == pytest.approx(0.5)
    assert chem.cortisol == pytest.approx(0.1)

def test_tract_data():
    tract = TractData(name="DCML", origin="DorsalHorn", destination="VPL", signal=np.ones(64, dtype=np.float32))
    assert tract.name == "DCML"
    assert tract.origin == "DorsalHorn"
    assert tract.destination == "VPL"
    assert tract.signal.shape == (64,)

def test_neural_bus_state_initialization():
    bus = NeuralBusState()
    # Exactly 12 cranial nerves
    assert len(bus.cranial_nerves) == 12
    assert "CN_I" in bus.cranial_nerves
    assert "CN_XII" in bus.cranial_nerves
    assert bus.cranial_nerves["CN_II"].modality == "SSA"
    assert bus.cranial_nerves["CN_X"].modality == "GVE"
    
    # Exactly 31 spinal segments
    assert len(bus.spinal_nerves) == 31
    assert "C1" in bus.spinal_nerves
    assert "C8" in bus.spinal_nerves
    assert "T1" in bus.spinal_nerves
    assert "T12" in bus.spinal_nerves
    assert "L1" in bus.spinal_nerves
    assert "L5" in bus.spinal_nerves
    assert "S1" in bus.spinal_nerves
    assert "S5" in bus.spinal_nerves
    
    # Autonomic & Enteric
    assert bus.autonomic.heart_rate == 70.0
    assert bus.enteric_satiety == 0.7
    assert bus.enteric_motility == 0.5
