import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.peripheral.bus import UniversalNeuralBus

def test_universal_neural_bus_initialization():
    bus = UniversalNeuralBus()
    assert len(bus.cranial.nerves) == 12
    assert len(bus.spinal.segments) == 31
    assert bus.autonomic.sympathetic_tone == pytest.approx(0.3)
    assert bus.autonomic.parasympathetic_tone == pytest.approx(0.7)

def test_cranial_nerve_io():
    bus = UniversalNeuralBus()
    # CN II: Retinogeniculate / Retinotectal visual afferents
    bus.cranial.set_sensory("CN_II", np.ones(64, dtype=np.float32) * 0.85)
    sig = bus.cranial.get_sensory("CN_II")
    assert np.allclose(sig, 0.85)
    
    # CN XII: Hypoglossal motor speech efferents
    bus.cranial.set_motor("CN_XII", np.ones(64, dtype=np.float32) * 0.45)
    motor = bus.cranial.get_motor("CN_XII")
    assert np.allclose(motor, 0.45)

def test_spinal_dermatome_myotome():
    bus = UniversalNeuralBus()
    # C5 segment dermatomal sensory input
    dermatome_in = np.array([0.1, 0.2, 0.3, 0.4] * 4, dtype=np.float32)
    bus.spinal.set_dermatome("C5", dermatome_in)
    assert bus.spinal.get_dermatome("C5")[0] == pytest.approx(0.1)
    
    # C5 segment myotomal motor output
    myotome_out = np.ones(8, dtype=np.float32) * 0.95
    bus.spinal.set_myotome("C5", myotome_out)
    assert bus.spinal.get_myotome("C5")[0] == pytest.approx(0.95)

def test_autonomic_regulation_and_homeostasis():
    bus = UniversalNeuralBus()
    # Trigger acute sympathetic surge (fight-or-flight)
    bus.autonomic.trigger_sympathetic_surge(intensity=0.6)
    assert bus.autonomic.heart_rate > 75.0
    assert bus.autonomic.pupil_diameter > 3.8
    assert bus.autonomic.sympathetic_tone > 0.4
    
    # Trigger parasympathetic vagal stimulation (rest-and-digest)
    bus.autonomic.trigger_vagal_tone(intensity=0.8)
    assert bus.autonomic.parasympathetic_tone > 0.6
    
    # Step homeostatic decay towards baseline
    bus.autonomic.step_homeostasis(dt=1.0)
    assert 60.0 < bus.autonomic.heart_rate < 120.0

def test_enteric_gut_brain_axis():
    bus = UniversalNeuralBus()
    # Ingest nutrient energy
    bus.enteric.ingest_nutrients(amount=0.5)
    assert bus.enteric.satiety > 0.7
    assert bus.enteric.motility > 0.5
    
    # Step digestion
    bus.enteric.step(dt=1.0)
    assert bus.enteric.satiety <= 1.0
