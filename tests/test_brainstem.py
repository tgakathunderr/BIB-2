import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.brainstem.medulla import PreBotzingerComplex, MedullaOblongata
from bib2.brainstem.pons import MetencephalicPons
from bib2.brainstem.midbrain import Mesencephalon
from bib2.brainstem import BrainstemComplex

def test_pre_botzinger_rhythmogenesis_and_opioids():
    pre_bot = PreBotzingerComplex()
    # Step 50 ticks to observe rhythmic eupneic burst pattern
    bursts = [pre_bot.step(dt=0.1) for _ in range(50)]
    assert sum(bursts) > 0 # Rhythmic inspiration bursts generated
    
    # Apply mu-opioid receptor agonist -> suppresses respiratory burst frequency
    pre_bot.apply_opioid_agonism(agonism=0.95)
    suppressed_bursts = [pre_bot.step(dt=0.1) for _ in range(50)]
    assert sum(suppressed_bursts) <= sum(bursts)

def test_medulla_baroreflex_regulation():
    medulla = MedullaOblongata()
    # High blood pressure (hypertension: 165 mmHg) into NTS triggers baroreflex
    symp_tone, parasymp_tone = medulla.process_baroreflex(bp_sys=165.0)
    # Must inhibit sympathetic tone (via CVLM -> RVLM) and elevate vagal parasympathetic tone
    assert symp_tone < 0.3
    assert parasymp_tone > 0.7

def test_inferior_olive_climbing_spikes():
    medulla = MedullaOblongata()
    # High performance error between cortical motor plan and sensory feedback
    error_vec = np.array([0.9, -0.6, 0.4, -0.2] * 16, dtype=np.float32)
    climbing_spikes = medulla.inferior_olive.generate_climbing_spikes(error_vec)
    assert climbing_spikes.shape == (64,)
    assert np.max(climbing_spikes) > 0.0

def test_pons_locus_coeruleus_and_rem_atonia():
    pons = MetencephalicPons()
    # Salience trigger excites locus coeruleus noradrenaline release
    ne_surge = pons.locus_coeruleus.step_arousal(salience=0.85)
    assert ne_surge > 0.5
    
    # Sublaterodorsal nucleus REM motor atonia activation
    atonia = pons.sublaterodorsal.compute_motor_inhibition(is_rem_sleep=True)
    assert atonia > 0.8
    wake_atonia = pons.sublaterodorsal.compute_motor_inhibition(is_rem_sleep=False)
    assert wake_atonia == 0.0

def test_midbrain_colliculi_and_pag():
    midbrain = Mesencephalon()
    visual_stim = np.zeros(64, dtype=np.float32)
    visual_stim[15] = 0.95 # Salient peripheral visual cue
    saccade_vec = midbrain.superior_colliculus.compute_saccade(visual_stim)
    assert saccade_vec.shape == (2,)
    assert np.linalg.norm(saccade_vec) > 0.0
    
    # Periaqueductal Gray (PAG) defensive response to severe threat
    pag_defensive = midbrain.pag.process_threat_arousal(threat_level=0.9)
    assert pag_defensive["freeze"] is True or pag_defensive["fight_flight"] is True

def test_brainstem_complex_integrated():
    brainstem = BrainstemComplex()
    assert brainstem.medulla is not None
    assert brainstem.pons is not None
    assert brainstem.midbrain is not None
    
    step_out = brainstem.step(bp_sys=120.0, visual_stim=np.ones(64), is_rem=False)
    assert "respiratory_burst" in step_out
    assert "ne_level" in step_out
    assert "baroreflex" in step_out
