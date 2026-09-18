import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.telencephalon.neocortex.layer import CanonicalMicrocircuit
from bib2.telencephalon.neocortex.areas import CorticalAreaRegistry, CerebralNeocortex
from bib2.telencephalon.neocortex.networks import TriNetworkCoordinator

def test_canonical_microcircuit_laminar_flow():
    cm = CanonicalMicrocircuit(dim=64)
    thalamic_input = np.ones(64, dtype=np.float32) * 0.75
    # Canonical laminar circuit: Thalamus -> L4 -> L2/3 -> L5 -> L6
    l4, l23, l5, l6 = cm.forward(thalamic_input)
    assert l4.shape == (64,)
    assert l23.shape == (64,)
    assert l5.shape == (64,) # Subcortical motor output
    assert l6.shape == (64,) # Corticothalamic feedback
    assert np.all(l4 > 0.0)
    assert np.all(l5 > 0.0)

def test_14_brodmann_areas_initialized():
    reg = CorticalAreaRegistry(dim=64)
    assert len(reg.areas) >= 14
    assert "M1_PrimaryMotor" in reg.areas
    assert "S1_Somatosensory" in reg.areas
    assert "V1_Visual" in reg.areas
    assert "A1_PrimaryAuditory" in reg.areas
    assert "Broca_SpeechProduction" in reg.areas
    assert "Wernicke_SpeechComprehension" in reg.areas
    assert "DLPFC_WorkingMemory" in reg.areas
    assert "OFC_Valuation" in reg.areas
    assert "VMPFC_AffectiveDecision" in reg.areas
    assert "SPL_SpatialIntegration" in reg.areas
    assert "IPL_MultimodalHub" in reg.areas
    assert "IT_ObjectRecognition" in reg.areas
    assert "FFA_FaceArea" in reg.areas
    assert "Insula_Interoception" in reg.areas

def test_visual_stream_bifurcation():
    cortex = CerebralNeocortex(dim=64)
    v1_input = np.ones(64, dtype=np.float32) * 0.8
    dorsal_where, ventral_what = cortex.process_visual_bifurcation(v1_input)
    assert dorsal_where.shape == (64,)
    assert ventral_what.shape == (64,)
    assert np.max(dorsal_where) > 0.0
    assert np.max(ventral_what) > 0.0

def test_tri_network_salience_switch():
    tri = TriNetworkCoordinator()
    # Baseline: DMN active (introspection), CEN inactive
    state_idle = tri.step(salience_event=False)
    assert state_idle["DMN"] > state_idle["CEN"]

    # Salience event detected: Anterior Insula switches DMN OFF and CEN ON
    state_task = tri.step(salience_event=True)
    assert state_task["CEN"] > state_task["DMN"]
    assert state_task["SalienceNetwork"] > 0.7
