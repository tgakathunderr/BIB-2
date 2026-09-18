import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.spinal.cord import SpinalCordEngine

def test_spinal_cord_laminae_initialization():
    cord = SpinalCordEngine(lamina_dim=64)
    assert cord.laminae.shape == (10, 64) # Laminae I through X
    assert cord.lamina_names[0] == "Lamina_I_Marginal"
    assert cord.lamina_names[1] == "Lamina_II_SubstantiaGelatinosa"
    assert cord.lamina_names[8] == "Lamina_IX_AlphaMotorPool"

def test_monosynaptic_stretch_reflex():
    cord = SpinalCordEngine(lamina_dim=64)
    # Muscle stretch input on Ia afferent (channel 0)
    stretch_in = np.zeros(16, dtype=np.float32)
    stretch_in[0] = 0.9 # High Ia stretch
    
    # Process reflex arc at C5
    efferent, antagonist_inhibition = cord.reflexes.process_stretch_reflex("C5", stretch_in)
    assert efferent[0] > 0.6 # Homonymous alpha motor contraction
    assert antagonist_inhibition > 0.5 # Reciprocal inhibition active

def test_crossed_extensor_reflex():
    cord = SpinalCordEngine(lamina_dim=64)
    # Pain stimulus at right leg (L4 dermatome)
    nociceptive_in = np.zeros(16, dtype=np.float32)
    nociceptive_in[4] = 0.85 # High pain in C-fibers
    
    ipsi_flexion, contra_extension = cord.reflexes.process_crossed_extensor("L4", nociceptive_in)
    assert ipsi_flexion > 0.5 # Limb pulls away
    assert contra_extension > 0.5 # Opposite limb extends to bear weight

def test_ascending_tracts_conduction():
    cord = SpinalCordEngine(lamina_dim=64)
    touch_data = np.ones(16, dtype=np.float32) * 0.7
    pain_data = np.ones(16, dtype=np.float32) * 0.85
    
    dcml_out, stt_out = cord.conduct_ascending(touch_data, pain_data)
    assert dcml_out.shape == (64,)
    assert stt_out.shape == (64,)
    assert np.mean(dcml_out) > 0.2
    assert np.mean(stt_out) > 0.2

def test_descending_corticospinal_decussation():
    cord = SpinalCordEngine(lamina_dim=64)
    cortical_drive = np.ones(64, dtype=np.float32) * 0.6
    
    lateral_motor, anterior_motor = cord.conduct_descending(cortical_drive)
    # Lateral tract contains 85-90% of decussated distal drive; anterior has 10-15% axial drive
    assert np.sum(lateral_motor) > np.sum(anterior_motor)
    assert np.isclose(np.sum(lateral_motor) / (np.sum(lateral_motor) + np.sum(anterior_motor)), 0.85, atol=0.05)
