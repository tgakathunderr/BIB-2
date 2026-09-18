import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.sleep.orchestrator import SleepOrchestrator, SleepStage
from bib2.sleep.ripples import SharpWaveRippleEngine
from bib2.sleep.shy import TononiSHYDownscaling
from bib2.sleep.glymphatic import GlymphaticEngine

def test_two_process_sleep_trigger():
    sleep = SleepOrchestrator()
    assert sleep.current_stage == SleepStage.WAKE
    
    # High adenosine pressure (Process S) + circadian nadir (Process C) triggers sleep
    sleep.update_processes(adenosine_pressure=0.9, circadian_phase=0.8)
    assert sleep.should_sleep() is True
    
    sleep.transition_to(SleepStage.NREM_SWS)
    assert sleep.current_stage == SleepStage.NREM_SWS

def test_sharp_wave_ripples_memory_replay():
    swr = SharpWaveRippleEngine(replay_frequency_hz=200.0)
    trajectories = [np.ones(64, dtype=np.float32) * (i + 1) * 0.2 for i in range(5)]
    replayed = swr.replay_episodes(trajectories)
    assert len(replayed) == 5
    assert swr.replay_frequency_hz >= 150.0 # 150-250 Hz SWR ripple burst
    assert np.all(replayed[0] > 0.0)

def test_tononi_shy_downscaling():
    shy = TononiSHYDownscaling(prune_threshold=0.08)
    synapses = np.array([0.95, 0.45, 0.12, 0.05], dtype=np.float32)
    downscaled = shy.downscale(synapses, downscale_factor=0.85)
    
    assert downscaled[0] < 0.95 # Proportional reduction
    assert downscaled[0] == pytest.approx(0.95 * 0.85, rel=1e-3)
    assert downscaled[3] == 0.0 # Sub-threshold (0.05 * 0.85 = 0.0425 < 0.08) pruned to 0

def test_glymphatic_waste_clearance():
    glym = GlymphaticEngine()
    # Wake (high NE): interstitial space ~14%
    assert glym.compute_interstitial_fraction(ne_level=0.8) == pytest.approx(0.14, abs=0.02)
    # Slow-Wave Sleep (zero NE): interstitial space expands by 60% to ~23%
    assert glym.compute_interstitial_fraction(ne_level=0.0) == pytest.approx(0.23, abs=0.02)

    solute_load = 100.0
    cleared = glym.flush_parenchyma(solute_load, is_sws=True)
    assert cleared < solute_load # Metabolic solutes flushed out
    
    # Waste clearance during wake is much lower
    cleared_wake = glym.flush_parenchyma(solute_load, is_sws=False)
    assert cleared < cleared_wake
