import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from bib2.brain import BIB2NervousSystem
from bib2.adapters.robotics import RoboticsAdapter
from bib2.adapters.language import LanguageAdapter
from bib2.adapters.agent import AgentAdapter

def test_robotics_adapter_camera_and_torques():
    brain = BIB2NervousSystem(seed=42)
    robot = RoboticsAdapter(brain, num_joints=6)

    camera_image = np.random.uniform(0, 1, (32, 32)).astype(np.float32)
    imu_6dof = np.array([0.0, 0.0, 9.81, 0.05, -0.02, 0.01], dtype=np.float32)
    tactile_touch = np.array([0.2, 0.5, 0.1, 0.0], dtype=np.float32)

    joint_torques = robot.step(camera_image, imu_6dof, tactile_touch)
    assert joint_torques.shape == (6,)
    assert np.all(np.isfinite(joint_torques))

def test_language_adapter_comprehension_and_generation():
    brain = BIB2NervousSystem(seed=42)
    lang = LanguageAdapter(brain)

    prompt = "hello biological intelligence"
    response = lang.process_text(prompt)
    assert isinstance(response, str)
    assert len(response) > 0

def test_agent_adapter_rl_loop():
    brain = BIB2NervousSystem(seed=42)
    agent = AgentAdapter(brain, obs_dim=16, n_actions=4)

    obs = np.ones(16, dtype=np.float32) * 0.5
    action_idx = agent.act(obs, reward=1.0)
    assert 0 <= action_idx < 4

    # Negative reward / pain
    action_punished = agent.act(obs, reward=-0.8)
    assert 0 <= action_punished < 4
    assert brain.peripheral.autonomic.sympathetic_tone > 0.35
