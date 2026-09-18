import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.brain import BIB2NervousSystem
from bib2.types import NeuralBusState

def test_bib2_nervous_system_initialization():
    brain = BIB2NervousSystem(seed=42)
    assert brain.peripheral is not None
    assert brain.spinal_cord is not None
    assert brain.brainstem is not None
    assert brain.cerebellum is not None
    assert brain.thalamus is not None
    assert brain.trn is not None
    assert brain.hypothalamus is not None
    assert brain.habenula is not None
    assert brain.basal_ganglia is not None
    assert brain.limbic is not None
    assert brain.amygdala is not None
    assert brain.claustrum is not None
    assert brain.neocortex is not None
    assert brain.chemistry is not None
    assert brain.sleep is not None

def test_deterministic_19_step_tick():
    brain1 = BIB2NervousSystem(seed=42)
    brain2 = BIB2NervousSystem(seed=42)

    bus1 = NeuralBusState()
    bus1.cranial_nerves["CN_II"].data = np.ones(64, dtype=np.float32) * 0.75
    bus2 = NeuralBusState()
    bus2.cranial_nerves["CN_II"].data = np.ones(64, dtype=np.float32) * 0.75

    out1 = brain1.tick(bus1)
    out2 = brain2.tick(bus2)

    # Absolute mathematical determinism between independent instances with identical seed
    for cid in out1.cranial_nerves:
        assert np.allclose(out1.cranial_nerves[cid].data, out2.cranial_nerves[cid].data)

    for seg in ["C1", "C5", "T1", "L4", "S1"]:
        assert np.allclose(out1.spinal_nerves[seg].myotome, out2.spinal_nerves[seg].myotome)

def test_multi_tick_stability():
    brain = BIB2NervousSystem(seed=123)
    bus = NeuralBusState()
    bus.cranial_nerves["CN_II"].data = np.random.uniform(0, 1, 64).astype(np.float32)
    
    # Run 50 continuous ticks
    for _ in range(50):
        out_bus = brain.tick(bus)
        assert out_bus is not None
        assert not np.isnan(brain.chemistry.matrix.state.dopamine)
        assert 50.0 <= brain.peripheral.autonomic.heart_rate <= 160.0
