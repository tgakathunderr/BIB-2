import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.telencephalon.limbic import HippocampalFormation, CircuitOfPapez, LimbicComplex
from bib2.telencephalon.amygdala import AmygdaloidComplex
from bib2.telencephalon.claustrum import ClaustrumSynchronizer

def test_dentate_gyrus_pattern_separation():
    hippo = HippocampalFormation(dim=64, dg_dim=256)
    sdr_a = np.ones(64, dtype=np.float32)
    sdr_b = sdr_a.copy()
    sdr_b[0] = 0.0 # Highly overlapping input (98.4% similar)

    sep_a = hippo.dentate_gyrus.separate(sdr_a)
    sep_b = hippo.dentate_gyrus.separate(sdr_b)

    # Cosine distance between separated outputs must be larger than input distance (lower cosine similarity)
    cos_in = np.dot(sdr_a, sdr_b) / (np.linalg.norm(sdr_a) * np.linalg.norm(sdr_b))
    cos_out = np.dot(sep_a, sep_b) / (np.linalg.norm(sep_a) * np.linalg.norm(sep_b) + 1e-8)
    assert cos_out < cos_in

def test_ca3_pattern_completion():
    hippo = HippocampalFormation(dim=64, dg_dim=256)
    stored_pattern = np.random.uniform(0, 1, 64).astype(np.float32)
    hippo.ca3.store(stored_pattern)

    # Degrade pattern by zeroing out 50% of the vector
    degraded = stored_pattern.copy()
    degraded[:32] = 0.0

    completed = hippo.ca3.complete(degraded)
    overlap_degraded = np.dot(degraded, stored_pattern)
    overlap_completed = np.dot(completed, stored_pattern)
    assert overlap_completed > overlap_degraded

def test_circuit_of_papez_loop():
    papez = CircuitOfPapez(dim=64)
    episode = np.ones(64, dtype=np.float32) * 0.7
    recirculated = papez.cycle(episode)
    assert recirculated.shape == (64,)
    assert np.all(recirculated > 0.0)

def test_amygdala_fear_and_valence():
    amygdala = AmygdaloidComplex(dim=64)
    threat_stimulus = np.ones(64, dtype=np.float32) * 0.8
    # Condition fear response
    amygdala.condition_threat(threat_stimulus, unconditioned_pain=0.95)
    salience, fear_output = amygdala.evaluate_salience(threat_stimulus)
    assert salience > 0.7
    assert fear_output > 0.7

    # Neutral stimulus has low fear
    neutral_stim = np.zeros(64, dtype=np.float32)
    salience_neu, fear_neu = amygdala.evaluate_salience(neutral_stim)
    assert fear_neu < 0.2

def test_claustrum_synchronizer():
    claustrum = ClaustrumSynchronizer(dim=64)
    modality_a = np.ones(64, dtype=np.float32) * 0.5
    modality_b = np.ones(64, dtype=np.float32) * 0.8
    bound_sdr = claustrum.synchronize([modality_a, modality_b])
    assert bound_sdr.shape == (64,)
    assert np.mean(bound_sdr) > 0.4
