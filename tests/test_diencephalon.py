import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.diencephalon.thalamus import ThalamusComplex
from bib2.diencephalon.trn import ThalamicReticularNucleus
from bib2.diencephalon.hypothalamus import HypothalamicComplex
from bib2.diencephalon.habenula import LateralHabenula

def test_thalamus_relay_nuclei():
    thal = ThalamusComplex(dim=64)
    assert "LGN" in thal.relay_nuclei
    assert "MGN" in thal.relay_nuclei
    assert "VPL" in thal.relay_nuclei
    assert "VPM" in thal.relay_nuclei
    assert "VA_VL" in thal.relay_nuclei
    assert "Anterior" in thal.relay_nuclei
    assert "Pulvinar" in thal.relay_nuclei
    assert "DM" in thal.relay_nuclei

    # Retinogeniculate routing
    vis_in = np.ones(64, dtype=np.float32) * 0.8
    lgn_out = thal.relay_sensory("LGN", vis_in)
    assert lgn_out.shape == (64,)
    assert np.allclose(lgn_out, 0.8)

def test_trn_attentional_gating_and_spindles():
    trn = ThalamicReticularNucleus(dim=64)
    # Attentional spotlight focusing on channel 32 to 64, suppressing 0 to 32
    focus_mask = np.ones(64, dtype=np.float32)
    focus_mask[:32] = 0.0
    trn.set_attentional_focus(focus_mask)

    signal = np.ones(64, dtype=np.float32)
    gated_signal = trn.gate_thalamic_output("LGN", signal)
    assert np.all(gated_signal[:32] == 0.0)
    assert np.all(gated_signal[32:] > 0.0)

    # Sleep spindle generation during NREM Stage 2
    spindles = trn.generate_sleep_spindles(is_nrem2=True)
    assert spindles.shape == (64,)
    assert np.max(spindles) > 0.0
    wake_spindles = trn.generate_sleep_spindles(is_nrem2=False)
    assert np.max(wake_spindles) == 0.0

def test_hypothalamus_scn_circadian_and_metabolism():
    hypo = HypothalamicComplex()
    # 24-hour cycle progression
    phase_noon = hypo.scn.step_circadian(dt_hours=12.0)
    assert 0.0 <= phase_noon <= 1.0

    # Low glucose, low leptin -> orexigenic NPY/AgRP hunger dominance
    hunger, satiety = hypo.arcuate.evaluate_metabolism(glucose=0.25, leptin=0.15)
    assert hunger > satiety

    # High glucose, high leptin -> anorexigenic POMC/CART satiety dominance
    hunger_fed, satiety_fed = hypo.arcuate.evaluate_metabolism(glucose=0.85, leptin=0.8)
    assert satiety_fed > hunger_fed

def test_lateral_habenula_anti_reward():
    lhb = LateralHabenula()
    # High expectation, zero outcome -> massive disappointment
    anti_reward = lhb.compute_anti_reward(expected_reward=0.9, received_reward=0.0)
    assert anti_reward > 0.8
    assert lhb.should_suppress_dopamine() is True

    # Outcome meets expectation -> zero anti-reward
    neutral = lhb.compute_anti_reward(expected_reward=0.7, received_reward=0.7)
    assert neutral == 0.0
    assert lhb.should_suppress_dopamine() is False
