"""Comprehensive End-to-End System Tests: Continuous multi-modal execution and state stability."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.brain import BIB2NervousSystem
from bib2.sleep.orchestrator import SleepStage
from bib2.adapters.robotics import RoboticsAdapter
from bib2.adapters.language import LanguageAdapter
from bib2.adapters.agent import AgentAdapter


def test_continuous_100_ticks_biological_stability():
    """Verify 100 continuous ticks under varying sensory inputs maintain physiological homeostasis."""
    brain = BIB2NervousSystem(seed=123)

    for step in range(100):
        bus = brain.get_bus()
        # Dynamic sensory stimuli
        bus.cranial_nerves["CN_II"].data = np.sin(np.linspace(0, 3.14, 64) + step * 0.1).astype(np.float32)
        bus.cranial_nerves["CN_VIII"].data = np.cos(np.linspace(0, 3.14, 64) + step * 0.05).astype(np.float32)
        bus.spinal_nerves["C5"].dermatome = np.full(16, fill_value=0.2 + 0.1 * np.sin(step), dtype=np.float32)

        out_bus = brain.tick(bus)

        # Confirm all outputs are finite numbers
        assert np.all(np.isfinite(out_bus.cranial_nerves["CN_III"].data))
        assert np.all(np.isfinite(out_bus.cranial_nerves["CN_XII"].data))

        # Physiological bounds check
        ans = brain.peripheral.autonomic
        assert 40.0 <= ans.heart_rate <= 180.0
        assert 80.0 <= ans.blood_pressure_sys <= 200.0
        assert 0.0 <= ans.sympathetic_tone <= 1.0
        assert 0.0 <= ans.parasympathetic_tone <= 1.0

        # Neurotransmitter matrix bounds check
        chem = brain.chemistry.matrix.state
        assert 0.0 <= chem.dopamine <= 1.0
        assert 0.0 <= chem.serotonin <= 1.0
        assert 0.0 <= chem.norepinephrine <= 1.0
        assert 0.0 <= chem.acetylcholine <= 1.0


def test_sleep_wake_cycle_full_transition():
    """Verify state transitions across WAKE -> NREM -> SWS -> REM -> WAKE."""
    brain = BIB2NervousSystem(seed=42)

    # 1. Awake
    assert brain.sleep.current_stage == SleepStage.AWAKE

    # 2. Transition to Light NREM
    brain.sleep.transition_to(SleepStage.NREM_LIGHT)
    assert brain.sleep.current_stage == SleepStage.NREM_LIGHT

    # 3. Transition to SWS (Slow-Wave Sleep)
    brain.sleep.transition_to(SleepStage.NREM_SWS)
    assert brain.sleep.current_stage == SleepStage.NREM_SWS
    brain.tick_sleep()

    # 4. Transition to REM
    brain.sleep.transition_to(SleepStage.REM)
    assert brain.sleep.current_stage == SleepStage.REM

    # 5. Return to Awake
    brain.sleep.transition_to(SleepStage.AWAKE)
    assert brain.sleep.current_stage == SleepStage.AWAKE


def test_all_domain_adapters_concurrent_operation():
    """Verify that Robotics, Language, and Agent adapters can run against the same brain instance."""
    brain = BIB2NervousSystem(seed=999)
    robot = RoboticsAdapter(brain, num_joints=6)
    lang = LanguageAdapter(brain)
    agent = AgentAdapter(brain, obs_dim=16, n_actions=4)

    # Run robotic step
    cam = np.random.uniform(0, 1, (32, 32)).astype(np.float32)
    imu = np.array([0, 0, 9.8, 0, 0, 0], dtype=np.float32)
    torques = robot.step(cam, imu)
    assert torques.shape == (6,)

    # Run language step
    reply = lang.process_text("system active")
    assert len(reply) > 0

    # Run agent RL step
    action = agent.act(np.ones(16, dtype=np.float32) * 0.5, reward=0.5)
    assert 0 <= action < 4
