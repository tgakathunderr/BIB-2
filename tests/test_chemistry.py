import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.chemistry.matrix import ChemicalMatrix
from bib2.chemistry.hpa import HPAAxis
from bib2.chemistry.plasticity import PlasticityEngine

def test_chemical_matrix_updates():
    matrix = ChemicalMatrix()
    matrix.release_dopamine(amount=0.4)
    assert matrix.state.dopamine > 0.5
    matrix.release_norepinephrine(amount=0.3)
    assert matrix.state.norepinephrine > 0.4
    matrix.step_clearance(dt=1.0)
    assert matrix.state.dopamine < 0.95
    assert matrix.state.adenosine >= 0.0

def test_hpa_stress_axis_negative_feedback():
    hpa = HPAAxis()
    hpa.trigger_stress(stress_level=0.85)
    assert hpa.crh > 0.6
    assert hpa.acth > 0.6
    assert hpa.cortisol > 0.5

    # Negative feedback dampens CRH and ACTH over subsequent ticks
    for _ in range(5):
        hpa.step(dt=1.0)
    assert hpa.crh < 0.6
    assert hpa.acth < 0.6

def test_e_ltp_and_ltd_plasticity():
    pe = PlasticityEngine()
    w_initial = 0.5

    # High frequency / high calcium surge -> E-LTP (CaMKII/PKC phosphorylation)
    w_potentiated = pe.compute_plasticity(w_initial, ca_influx=0.9, freq_hz=50.0)
    assert w_potentiated > w_initial

    # Low frequency / low calcium -> LTD (calcineurin/PP1 dephosphorylation)
    w_depressed = pe.compute_plasticity(w_initial, ca_influx=0.15, freq_hz=2.0)
    assert w_depressed < w_initial

def test_ghk_resting_potential():
    pe = PlasticityEngine()
    # Baseline physiological ion concentrations (mM)
    vm = pe.compute_ghk_potential(k_out=4.0, k_in=140.0, na_out=145.0, na_in=12.0, cl_out=110.0, cl_in=4.0)
    # Typical resting membrane potential is between -65 mV and -75 mV
    assert -75.0 <= vm <= -60.0
