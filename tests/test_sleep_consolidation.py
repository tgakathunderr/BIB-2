"""Test episodic memory retention and Tononi SHY downscaling across Slow-Wave Sleep consolidation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.brain import BIB2NervousSystem
from bib2.sleep.orchestrator import SleepStage


def test_memory_retention_after_sws_consolidation():
    brain = BIB2NervousSystem(seed=42)

    # 1. Daytime learning of 3 episodes
    for i in range(3):
        bus = brain.get_bus()
        bus.cranial_nerves["CN_II"].data = np.ones(64, dtype=np.float32) * (i + 1) * 0.3
        brain.tick(bus)

    # Store recall accuracy before sleep
    pre_sleep_weights = brain.neocortex.get_total_synaptic_weight()

    # 2. Transition to Slow-Wave Sleep and execute SWR replay and SHY downscaling
    brain.sleep.transition_to(SleepStage.NREM_SWS)
    for _ in range(10):
        brain.tick_sleep()

    # 3. Verify that sleep downscaled total synaptic weight (Tononi SHY efficiency)
    post_sleep_weights = brain.neocortex.get_total_synaptic_weight()
    assert post_sleep_weights < pre_sleep_weights

    # 4. Verify memories are preserved in neocortex
    recalled = brain.recall_episode(index=0)
    assert recalled is not None
    assert np.all(np.isfinite(recalled))
