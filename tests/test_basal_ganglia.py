import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.telencephalon.basal_ganglia import BasalGangliaComplex

def test_basal_ganglia_action_selection():
    bg = BasalGangliaComplex(n_actions=8)
    cortical_proposals = np.array([0.2, 0.85, 0.3, 0.1, 0.05, 0.4, 0.1, 0.2], dtype=np.float32)
    
    selected_idx, disinhibition = bg.select_action(cortical_proposals, dopamine_level=0.7)
    assert selected_idx == 1 # Action 1 had highest drive
    assert disinhibition > 0.0 # Thalamus disinhibited for action 1

def test_hyperdirect_emergency_brake():
    bg = BasalGangliaComplex(n_actions=8)
    cortical_proposals = np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Trigger STN hyperdirect global brake
    bg.trigger_hyperdirect_stop()
    selected_idx, disinhibition = bg.select_action(cortical_proposals, dopamine_level=0.7)
    assert disinhibition == 0.0 # Global brake prevents any action release

def test_dopamine_modulation_go_nogo():
    bg = BasalGangliaComplex(n_actions=4)
    proposals = np.array([0.5, 0.4, 0.3, 0.2], dtype=np.float32)
    
    # Low dopamine favours NoGo (indirect), high dopamine favours Go (direct)
    _, disinhibition_low_da = bg.select_action(proposals, dopamine_level=0.1)
    _, disinhibition_high_da = bg.select_action(proposals, dopamine_level=0.9)
    assert disinhibition_high_da > disinhibition_low_da

def test_striatal_reinforcement_plasticity():
    bg = BasalGangliaComplex(n_actions=4)
    proposals = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    initial_weights = bg.d1_weights.copy()
    
    # Positive reward prediction error reinforces chosen action 2
    bg.reinforce_action(action_idx=2, reward_rpe=0.8, lr=0.1)
    assert bg.d1_weights[2] > initial_weights[2]
