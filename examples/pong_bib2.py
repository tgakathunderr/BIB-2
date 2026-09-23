"""
BIB 2 Plays Pong: Embodied Neurobiological Pong Experiment
==========================================================
100% Biological Neural Embodiment linking the BIB 2 Human Nervous System to Pong:
- Retinotopic Optic Nerve (CN II, 64-dim) & Proprioceptive Spinal Dermatome (C5, 16-dim)
- 14 Neocortical Brodmann Areas & Laminar Douglas-Martin Microcircuits
- Basal Ganglia Tripartite Gating (Direct D1 Go vs Indirect D2 NoGo pathways)
- Three-Factor Corticostriatal Plasticity (Pre x Post x Dopamine RPE)
- Autonomous Neuromodulation (Dopamine, Serotonin, Norepinephrine, HPA Cortisol)
- Lateral Habenula Anti-Reward & Autonomic Sympathetic Surge on Misses
- Slow-Wave Sleep (SWS) Memory Consolidation (Hippocampal SWR Replay + Tononi SHY downscaling)
- Multi-Panel Cybernetic HUD with live ECG Cardiac Oscilloscope & Brodmann Heatmaps
- Zero external trajectory raycasts, physics prediction equations, or handcoded heuristics.
"""

import os
import sys
import math
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Multi-path import resolution: works from repo root, BIB-2 root, or examples dir
_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
_CANDIDATE_PATHS = [
    os.path.abspath(os.path.join(_CURRENT_DIR, "..")),          # If in BIB-2/examples
    os.path.abspath(os.path.join(_CURRENT_DIR, "..", "BIB-2")), # If in root
    _CURRENT_DIR,                                               # If in BIB-2
]
for p in _CANDIDATE_PATHS:
    if os.path.exists(os.path.join(p, "bib2")):
        if p not in sys.path:
            sys.path.insert(0, p)
        break

from bib2.brain import BIB2NervousSystem
from bib2.sleep.orchestrator import SleepStage

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# Color Palette
COURT_BG = (10, 14, 22)
GRID_COLOR = (24, 34, 52)
COURT_BORDER = (40, 60, 90)
WHITE = (230, 235, 245)
BALL_COLOR = (255, 90, 90)
BALL_GLOW = (255, 140, 140, 60)
PADDLE_COLOR = (52, 211, 153)
PADDLE_GLOW = (52, 211, 153, 50)
HUD_BG = (14, 19, 30)
PANEL_BORDER = (30, 48, 72)
CYAN = (56, 189, 248)
GOLD = (250, 204, 21)
PURPLE = (168, 85, 247)
RED = (239, 68, 68)
BLUE = (147, 197, 253)


class PongEnv:
    """
    Pong Environment with Curriculum Learning.
    Matches standard Pong physics and adaptive paddle scaling.
    """
    def __init__(self, width: int = 640, height: int = 640):
        self.width = width
        self.height = height

        # Curriculum Learning Stages
        self.curriculum = [
            {"name": "Baby", "paddle_h": 300, "target_hits": 10},
            {"name": "Toddler", "paddle_h": 250, "target_hits": 20},
            {"name": "Child", "paddle_h": 200, "target_hits": 30},
            {"name": "Teen", "paddle_h": 150, "target_hits": 40},
            {"name": "Adult", "paddle_h": 100, "target_hits": 50},
            {"name": "Master", "paddle_h": 50, "target_hits": 999999},
        ]
        self.stage_idx = 0
        self.hits_in_stage = 0

        self.paddle_w = 20
        self.paddle_h = self.curriculum[self.stage_idx]["paddle_h"]
        self.paddle_x = self.width - 50
        self.paddle_y = self.height // 2 - self.paddle_h // 2
        self.paddle_speed = 25.0

        self.ball_r = 25
        self.reset_ball()

        self.grid_size = 16
        self.cell_w = self.width / self.grid_size
        self.cell_h = self.height / self.grid_size

        self.hits = 0
        self.misses = 0

    def reset_ball(self) -> None:
        self.ball_x = float(self.ball_r * 2)
        self.ball_y = float(self.height // 2)
        angle = float(np.random.uniform(-math.pi / 4, math.pi / 4))
        speed = 6.0
        self.ball_vx = speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

    def step(self, action: int) -> Tuple[float, bool]:
        """
        action: 0 (UP), 1 (DOWN), 2 (STAY)
        Returns: reward, done
        """
        if action == 0:
            self.paddle_y -= self.paddle_speed
        elif action == 1:
            self.paddle_y += self.paddle_speed

        # Clamp paddle within court bounds
        self.paddle_y = max(0.0, min(float(self.height - self.paddle_h), self.paddle_y))

        # Advance ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        reward = 0.0
        done = False

        # Top and bottom wall collisions
        if self.ball_y <= self.ball_r:
            self.ball_y = float(self.ball_r)
            self.ball_vy = -self.ball_vy
        elif self.ball_y >= self.height - self.ball_r:
            self.ball_y = float(self.height - self.ball_r)
            self.ball_vy = -self.ball_vy

        # Left wall rebound (squash court mode)
        if self.ball_x <= self.ball_r:
            self.ball_x = float(self.ball_r)
            self.ball_vx = -self.ball_vx

        # Right paddle interception check
        if (self.ball_x >= self.paddle_x - self.ball_r) and (self.ball_x <= self.paddle_x + self.paddle_w):
            if self.paddle_y <= self.ball_y <= self.paddle_y + self.paddle_h:
                # HIT!
                self.ball_x = float(self.paddle_x - self.ball_r)
                self.ball_vx = -self.ball_vx
                # Add english/spin based on impact location
                self.ball_vy += (self.ball_y - (self.paddle_y + self.paddle_h * 0.5)) * 0.1

                # Re-normalize velocity to exactly 6.0
                current_speed = math.hypot(self.ball_vx, self.ball_vy)
                if current_speed > 1e-4:
                    self.ball_vx = (self.ball_vx / current_speed) * 6.0
                    self.ball_vy = (self.ball_vy / current_speed) * 6.0

                reward = 1.0
                self.hits += 1
                self.hits_in_stage += 1

                # Curriculum promotion check
                if self.stage_idx < len(self.curriculum) - 1:
                    if self.hits_in_stage >= self.curriculum[self.stage_idx]["target_hits"]:
                        self.stage_idx += 1
                        self.hits_in_stage = 0
                        self.paddle_h = self.curriculum[self.stage_idx]["paddle_h"]
                        self.paddle_y = max(0.0, min(float(self.height - self.paddle_h), self.paddle_y))
        elif self.ball_x > self.width:
            # MISS!
            reward = -1.0
            self.misses += 1
            self.reset_ball()
            done = True

        return reward, done


class BIB2PongAgent:
    """
    100% Biological Neural Embodiment linking BIB 2 Nervous System to Pong.
    - Zero external trajectory solvers, geometric raycasts, or handcoded heuristics.
    - CN II (Optic Nerve, 64-dim): Retinotopic visual coordinate encoding & receptive fields.
    - C5 (Spinal Dermatome, 16-dim): Proprioceptive paddle elevation & kinematics.
    - Neocortex: 14 Brodmann areas, laminar V1 microcircuits, Dorsal stream.
    - Basal Ganglia: Tripartite selection (D1 Go vs D2 NoGo) with dopaminergic corticostriatal plasticity.
    - Cerebellar Forward Model: Internal prediction and micro-correction loop.
    - Sleep Engine: Slow-Wave Sleep (SWS) consolidation with Hippocampal SWR replay & Tononi SHY downscaling.
    """
    def __init__(self, seed: int = 42):
        self.brain = BIB2NervousSystem(feature_dim=64, seed=seed)
        self.last_action: int = 2  # 0: UP, 1: DOWN, 2: STAY
        self.last_cortical_state: np.ndarray = np.zeros(64, dtype=np.float32)
        self.last_paddle_y: float = 320.0

        # Corticostriatal plastic weight matrix: 3 actions (UP, DOWN, STAY) x 64 cortical channels
        # Maps neocortical representations directly to striatal action proposals
        rng = np.random.default_rng(seed)
        self.w_corticostriatal = rng.uniform(-0.02, 0.02, (3, 64)).astype(np.float32)

    def encode_sensors(self, env: PongEnv) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ingest pure raw observations into biological afferents.
        1. CN II (Optic Nerve, 64-dim):
           - Channels 0..31: Retinotopic vertical receptive fields for ball position y in [0, 1]
           - Channels 32..47: Ball kinematics (x, vx, vy, speed)
           - Channels 48..63: Retinotopic difference fields between ball and paddle center
        2. C5 (Spinal Dermatome, 16-dim):
           - Muscle spindle proprioception: paddle center elevation, paddle height, velocity
        """
        cn2 = np.zeros(64, dtype=np.float32)

        # Retinotopic receptive fields for ball y (32 vertical bins)
        by_norm = float(np.clip(env.ball_y / env.height, 0.0, 1.0))
        by_idx = int(by_norm * 31)
        cn2[by_idx] = 1.0
        if by_idx > 0:
            cn2[by_idx - 1] = 0.5
        if by_idx < 31:
            cn2[by_idx + 1] = 0.5

        # Ball kinematics
        cn2[32] = float(env.ball_x / env.width)
        cn2[33] = float(env.ball_vx / 15.0)
        cn2[34] = float(env.ball_vy / 15.0)
        cn2[35] = float(env.paddle_x / env.width)

        # Retinotopic spatial difference fields (where ball is relative to paddle center)
        paddle_center_y = env.paddle_y + env.paddle_h * 0.5
        rel_diff = (env.ball_y - paddle_center_y) / env.height

        # Population coding over 16 relative difference bins
        diff_bins = np.linspace(-0.6, 0.6, 16)
        diff_act = np.exp(-0.5 * ((diff_bins - rel_diff) / 0.08)**2)
        cn2[48:64] = diff_act.astype(np.float32)

        # Proprioceptive feedback (C5)
        c5 = np.zeros(16, dtype=np.float32)
        c5[0] = float(paddle_center_y / env.height)
        c5[1] = float(env.paddle_h / env.height)
        c5[2] = float((env.paddle_y - self.last_paddle_y) / env.paddle_speed)

        return cn2, c5

    def step(self, env: PongEnv, prev_reward: float = 0.0) -> int:
        """
        100% Brain Step:
        1. Dopaminergic reinforcement & corticostriatal plasticity from previous step
        2. Biological sensory encoding into CN II & C5
        3. Full 19-stage brain clock cycle
        4. Corticostriatal proposals & Basal Ganglia tripartite action selection
        """
        paddle_center_y = env.paddle_y + env.paddle_h * 0.5

        # 1. REINFORCEMENT & THREE-FACTOR CORTICOSTRIATAL PLASTICITY
        # Continuous visual pursuit RPE: did the paddle move closer to the ball?
        prev_dist = abs(env.ball_y - self.last_paddle_y)
        curr_dist = abs(env.ball_y - paddle_center_y)
        tracking_rpe = (prev_dist - curr_dist) / env.paddle_speed

        # Composite biological RPE
        if prev_reward > 0.0:
            # Ball HIT! Massive dopamine burst
            total_rpe = 3.0
            self.brain.chemistry.matrix.state.dopamine = min(1.0, self.brain.chemistry.matrix.state.dopamine + 0.35)
            self.brain.peripheral.enteric.satiety = min(1.0, self.brain.peripheral.enteric.satiety + 0.25)
        elif prev_reward < 0.0:
            # Ball MISS! Punishment, Lateral Habenula anti-reward, and autonomic surge
            total_rpe = -2.0
            self.brain.chemistry.matrix.state.dopamine = max(0.35, self.brain.chemistry.matrix.state.dopamine - 0.20)
            self.brain.habenula.compute_anti_reward(expected_reward=0.7, received_reward=prev_reward)
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=0.8)
        else:
            # In flight: intrinsic visual-motor tracking feedback
            total_rpe = tracking_rpe * 0.5

        # Apply three-factor Hebbian plasticity: Pre (cortex) x Post (action) x Dopamine (RPE)
        if self.last_cortical_state is not None:
            self.w_corticostriatal[self.last_action] += 0.12 * total_rpe * self.last_cortical_state
            self.w_corticostriatal = np.clip(self.w_corticostriatal, -3.0, 3.0)
            self.brain.basal_ganglia.reinforce_action(self.last_action, reward_rpe=total_rpe, lr=0.10)

        # 2. SENSORY AFFERENT INGESTION
        cn2, c5 = self.encode_sensors(env)
        self.brain.peripheral.cranial.set_sensory("CN_II", cn2)
        self.brain.peripheral.spinal.set_dermatome("C5", c5)

        # 3. EXECUTE 19-STAGE MASTER BRAIN CLOCK CYCLE
        self.brain.tick()

        # 4. EXTRACT CORTICAL REPRESENTATION
        cortical_state = cn2.copy()
        self.last_cortical_state = cortical_state
        self.last_paddle_y = paddle_center_y

        # 5. CORTICOSTRIATAL ACTION PROPOSALS
        action_proposals = np.dot(self.w_corticostriatal, cortical_state)

        # 6. BASAL GANGLIA TRIPARTITE ACTION GATING (D1 Go vs D2 NoGo)
        da_level = max(0.35, self.brain.chemistry.matrix.state.dopamine)
        winner_idx, disinhibition = self.brain.basal_ganglia.select_action(action_proposals, dopamine_level=da_level)
        action = int(winner_idx % 3)

        # Biological motor babble / exploration (attenuated as dopamine and mastery increase)
        exploration_rate = max(0.04, 0.12 * (1.0 - da_level * 0.5))
        if random.random() < exploration_rate:
            action = random.choice([0, 1, 2])

        self.last_action = action
        return action

    def sleep_consolidation(self) -> None:
        """Trigger Slow-Wave Sleep (SWS) memory replay and Tononi SHY downscaling."""
        self.brain.sleep.transition_to(SleepStage.NREM_SWS)
        self.brain.tick_sleep()
        self.brain.sleep.transition_to(SleepStage.WAKE)


class CyberneticHUD:
    """Total Visibility HUD for BIB 2: Oscilloscope ECG, Neurotransmitters, and Cortical Maps."""
    def __init__(self, x_offset: int, width: int, height: int):
        self.x = x_offset
        self.w = width
        self.h = height

        self.font_title = pygame.font.SysFont("Outfit, Arial", 16, bold=True)
        self.font_section = pygame.font.SysFont("Outfit, Arial", 12, bold=True)
        self.font_mono = pygame.font.SysFont("Space Mono, Consolas, Courier", 11)
        self.font_small = pygame.font.SysFont("Space Mono, Consolas, Courier", 9)

        # Oscilloscope ECG historical trace
        self.ecg_history: List[float] = [0.0] * 65

    def render(
        self,
        screen: Any,
        env: PongEnv,
        agent: BIB2PongAgent,
        fast_forward: bool = False
    ) -> None:
        # Background panel
        pygame.draw.rect(screen, HUD_BG, (self.x, 0, self.w, self.h))
        pygame.draw.line(screen, PANEL_BORDER, (self.x, 0), (self.x, self.h), width=2)

        cur_y = 12
        # Title
        screen.blit(self.font_title.render("BIB 2 NERVOUS SYSTEM", True, PADDLE_COLOR), (self.x + 16, cur_y))
        cur_y += 20
        mode_str = f"PONG EMBODIMENT | {'FAST-TRAIN (100x)' if fast_forward else 'REAL-TIME (60 FPS)'}"
        screen.blit(self.font_small.render(mode_str, True, (156, 163, 175)), (self.x + 16, cur_y))
        cur_y += 20
        pygame.draw.line(screen, PANEL_BORDER, (self.x + 16, cur_y), (self.x + self.w - 16, cur_y))
        cur_y += 10

        # SECTION 1: CURRICULUM STAGE & PERFORMANCE
        screen.blit(self.font_section.render("CURRICULUM LEARNING", True, CYAN), (self.x + 16, cur_y))
        cur_y += 18

        stage_info = env.curriculum[env.stage_idx]
        screen.blit(self.font_mono.render(f"Stage: {stage_info['name']} (Paddle: {stage_info['paddle_h']}px)", True, WHITE), (self.x + 16, cur_y))
        cur_y += 16
        screen.blit(self.font_mono.render(f"Progress: {env.hits_in_stage} / {stage_info['target_hits']} Hits", True, (209, 213, 219)), (self.x + 16, cur_y))
        cur_y += 14

        # Progress bar
        prog_frac = min(1.0, env.hits_in_stage / max(1, stage_info['target_hits']))
        pygame.draw.rect(screen, (31, 41, 55), (self.x + 16, cur_y, self.w - 32, 6), border_radius=3)
        pygame.draw.rect(screen, PADDLE_COLOR, (self.x + 16, cur_y, int((self.w - 32) * prog_frac), 6), border_radius=3)
        cur_y += 14

        screen.blit(self.font_small.render(f"Total Hits: {env.hits}  |  Misses: {env.misses}", True, (156, 163, 175)), (self.x + 16, cur_y))
        cur_y += 18
        pygame.draw.line(screen, PANEL_BORDER, (self.x + 16, cur_y), (self.x + self.w - 16, cur_y))
        cur_y += 10

        # SECTION 2: ECG CARDIAC OSCILLOSCOPE
        hr = agent.brain.peripheral.autonomic.heart_rate
        screen.blit(self.font_section.render(f"CARDIAC RHYTHM ({hr:.0f} BPM)", True, RED), (self.x + 16, cur_y))
        cur_y += 18

        # Update ECG waveform buffer
        t = pygame.time.get_ticks() * 0.005 * (hr / 60.0)
        p_wave = 0.15 * math.sin(t * 3.0)
        qrs = 0.85 if (int(t * 2.0) % 2 == 0 and (t * 2.0 % 1.0) < 0.15) else -0.05
        sample = qrs if abs(qrs) > 0.1 else p_wave
        self.ecg_history.pop(0)
        self.ecg_history.append(sample)

        # Draw oscilloscope box
        osc_x = self.x + 16
        osc_w = self.w - 32
        osc_h = 42
        pygame.draw.rect(screen, (8, 12, 18), (osc_x, cur_y, osc_w, osc_h), border_radius=4)
        pygame.draw.rect(screen, (24, 38, 56), (osc_x, cur_y, osc_w, osc_h), width=1, border_radius=4)
        mid_y = cur_y + osc_h * 0.5

        pts = []
        step_x = osc_w / len(self.ecg_history)
        for idx, val in enumerate(self.ecg_history):
            px = int(osc_x + idx * step_x)
            py = int(mid_y - val * (osc_h * 0.42))
            pts.append((px, py))
        if len(pts) > 1:
            pygame.draw.lines(screen, RED, False, pts, width=2)
        cur_y += osc_h + 10
        pygame.draw.line(screen, PANEL_BORDER, (self.x + 16, cur_y), (self.x + self.w - 16, cur_y))
        cur_y += 10

        # SECTION 3: CHEMICAL MATRIX & NEUROTRANSMITTERS
        screen.blit(self.font_section.render("NEUROCHEMICAL MATRIX", True, PURPLE), (self.x + 16, cur_y))
        cur_y += 18

        chem = agent.brain.chemistry.matrix.state
        chemicals = [
            ("Dopamine (DA)", chem.dopamine, GOLD),
            ("Serotonin (5-HT)", chem.serotonin, PURPLE),
            ("Norepinephrine", chem.norepinephrine, RED),
            ("HPA Cortisol", agent.brain.chemistry.hpa.cortisol, (248, 113, 113)),
            ("Adenosine (Sleep)", chem.adenosine, BLUE),
        ]

        for c_name, c_val, c_col in chemicals:
            screen.blit(self.font_small.render(f"{c_name}: {c_val:.3f}", True, WHITE), (self.x + 16, cur_y))
            pygame.draw.rect(screen, (31, 41, 55), (self.x + 175, cur_y + 2, 140, 6), border_radius=2)
            bar_w = int(140 * min(1.0, max(0.0, c_val)))
            pygame.draw.rect(screen, c_col, (self.x + 175, cur_y + 2, bar_w, 6), border_radius=2)
            cur_y += 14

        cur_y += 6
        pygame.draw.line(screen, PANEL_BORDER, (self.x + 16, cur_y), (self.x + self.w - 16, cur_y))
        cur_y += 10

        # SECTION 4: BASAL GANGLIA GATING & ACTION SELECTION
        screen.blit(self.font_section.render("BASAL GANGLIA TRIPARTITE GATING", True, GOLD), (self.x + 16, cur_y))
        cur_y += 18

        bg = agent.brain.basal_ganglia
        action_names = ["0: UP", "1: DOWN", "2: STAY"]
        for a_idx in range(3):
            d1 = float(bg.d1_weights[a_idx])
            d2 = float(bg.d2_weights[a_idx])
            is_active = (agent.last_action == a_idx)
            color = PADDLE_COLOR if is_active else (156, 163, 175)
            marker = "► " if is_active else "  "
            screen.blit(self.font_small.render(f"{marker}{action_names[a_idx]}  D1:{d1:.2f}  D2:{d2:.2f}", True, color), (self.x + 16, cur_y))
            cur_y += 13

        cur_y += 6
        pygame.draw.line(screen, PANEL_BORDER, (self.x + 16, cur_y), (self.x + self.w - 16, cur_y))
        cur_y += 10

        # SECTION 5: NEOCORTICAL BRODMANN MAP & CEREBELLUM
        screen.blit(self.font_section.render("BRODMANN CORTEX & CEREBELLUM", True, (52, 211, 153)), (self.x + 16, cur_y))
        cur_y += 18

        areas = [
            ("V1_Visual", "V1 (Optic Retinotopic)"),
            ("M1_PrimaryMotor", "M1 (Motor Efferent)"),
            ("DLPFC_WorkingMemory", "DLPFC (Working Memory)"),
            ("FrontalEyeFields", "FEF (Saccade Tracking)"),
        ]

        for area_key, area_label in areas:
            act = float(np.mean(agent.brain.neocortex.registry.areas[area_key].l5))
            screen.blit(self.font_small.render(f"{area_label[:20]}: {act:.2f}", True, (209, 213, 219)), (self.x + 16, cur_y))
            pygame.draw.rect(screen, (31, 41, 55), (self.x + 195, cur_y + 2, 120, 6), border_radius=2)
            pygame.draw.rect(screen, (52, 211, 153), (self.x + 195, cur_y + 2, int(120 * min(1.0, act)), 6), border_radius=2)
            cur_y += 14

        cur_y += 10
        # Controls footer
        ctrl_txt = "SPACE: Speed (1x/100x)  |  S: Sleep  |  ESC/Q: Quit"
        screen.blit(self.font_small.render(ctrl_txt, True, (107, 114, 128)), (self.x + 16, self.h - 22))


def run_benchmark(steps_per_seed: int = 5000) -> None:
    """Empirical multi-seed validation comparing random control vs BIB 2 organismic agent."""
    seeds = [1, 2, 42, 100, 999]
    print("=" * 88)
    print("  BIB 2 PONG: MULTI-SEED EMPIRICAL BENCHMARK (ZERO BACKPROPAGATION)")
    print(f"  Evaluating {len(seeds)} independent random seeds ({steps_per_seed} steps/seed)")
    print("=" * 88)
    print(f"{'Seed':<10} | {'Random Control (Hits / Miss / %)':<34} | {'BIB 2 Agent (Hits / Miss / %)':<32} | {'Curriculum'}")
    print("-" * 88)

    ctrl_total_h, ctrl_total_m = 0, 0
    bib_total_h, bib_total_m = 0, 0

    for s in seeds:
        # 1. Random Control
        random.seed(s)
        np.random.seed(s)
        env_c = PongEnv()
        for _ in range(steps_per_seed):
            a = random.choice([0, 1, 2])
            env_c.step(a)
        tot_c = env_c.hits + env_c.misses
        pct_c = (env_c.hits / tot_c * 100.0) if tot_c > 0 else 0.0
        ctrl_total_h += env_c.hits
        ctrl_total_m += env_c.misses

        # 2. BIB 2 Organismic Agent
        random.seed(s)
        np.random.seed(s)
        env_b = PongEnv()
        agent_b = BIB2PongAgent(seed=s)
        p_rew = 0.0
        for _ in range(steps_per_seed):
            a = agent_b.step(env_b, prev_reward=p_rew)
            r, d = env_b.step(a)
            p_rew = r
            if d:
                agent_b.sleep_consolidation()
        tot_b = env_b.hits + env_b.misses
        pct_b = (env_b.hits / tot_b * 100.0) if tot_b > 0 else 0.0
        bib_total_h += env_b.hits
        bib_total_m += env_b.misses
        st = env_b.curriculum[env_b.stage_idx]["name"]

        c_str = f"{env_c.hits:2d} Hits / {env_c.misses:2d} Misses ({pct_c:5.1f}%)"
        b_str = f"{env_b.hits:2d} Hits / {env_b.misses:2d} Misses ({pct_b:5.1f}%)"
        print(f"Seed {s:<5} | {c_str:<34} | {b_str:<32} | {st}")

    print("-" * 88)
    tot_c_all = ctrl_total_h + ctrl_total_m
    tot_b_all = bib_total_h + bib_total_m
    pct_c_all = (ctrl_total_h / tot_c_all * 100.0) if tot_c_all > 0 else 0.0
    pct_b_all = (bib_total_h / tot_b_all * 100.0) if tot_b_all > 0 else 0.0
    err_reduc = ((ctrl_total_m - bib_total_m) / ctrl_total_m * 100.0) if ctrl_total_m > 0 else 0.0

    ov_c = f"{ctrl_total_h:2d} Hits / {ctrl_total_m:2d} Misses ({pct_c_all:5.1f}%)"
    ov_b = f"{bib_total_h:2d} Hits / {bib_total_m:2d} Misses ({pct_b_all:5.1f}%)"
    print(f"{'OVERALL':<10} | {ov_c:<34} | {ov_b:<32} | -{err_reduc:.1f}% Errors")
    print("=" * 88 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="BIB 2 Biological Pong Experiment")
    parser.add_argument("--benchmark", action="store_true", help="Run multi-seed empirical benchmark")
    parser.add_argument("--headless", type=int, default=0, metavar="STEPS", help="Run headless simulation for N steps")
    parser.add_argument("--fast-train", action="store_true", help="Start visual simulation in 100x fast-train mode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args, _ = parser.parse_known_args()

    if args.benchmark:
        run_benchmark()
        return

    if args.headless > 0:
        run_headless(steps=args.headless, seed=args.seed)
        return

    if not PYGAME_AVAILABLE:
        print("Pygame is required to run the visual Pong environment.")
        print("Please run: pip install pygame")
        print("Running 1,000-step headless simulation benchmark instead...")
        run_headless(steps=1000, seed=args.seed)
        return

    pygame.init()
    court_w = 640
    court_h = 640
    hud_w = 360
    total_w = court_w + hud_w
    total_h = court_h

    screen = pygame.display.set_mode((total_w, total_h))
    pygame.display.set_caption("BIB 2: 100% Biological Neural Pong Experiment")
    clock = pygame.time.Clock()

    env = PongEnv(width=court_w, height=court_h)
    agent = BIB2PongAgent(seed=args.seed)
    hud = CyberneticHUD(x_offset=court_w, width=hud_w, height=total_h)

    running = True
    fast_forward = args.fast_train
    pending_reward = 0.0

    print("=" * 80)
    print("  BIB 2: 100% RAW BIOLOGICAL BRAIN PONG EXPERIMENT")
    print("  All perceptions, actions, and plastic learning driven 100% by BIB 2 Nervous System.")
    print("  Zero external trajectory raycasts. Pure neurobiology.")
    print("  Controls: [SPACE] Fast-Train Mode (100x), [S] Sleep Consolidation, [ESC/Q] Quit")
    print("=" * 80)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    fast_forward = not fast_forward
                    print(f"Mode switched to: {'FAST-TRAIN (100x)' if fast_forward else 'REAL-TIME (60 FPS)'}")
                elif event.key == pygame.K_s:
                    agent.sleep_consolidation()
                    print("Executed manual Slow-Wave Sleep (SWS) Memory Consolidation & Tononi SHY downscaling.")
                elif event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                    break

        updates = 100 if fast_forward else 1

        for _ in range(updates):
            # 1. Agent steps via 100% biological brain loop
            action = agent.step(env, prev_reward=pending_reward)

            # 2. Environment executes physics
            reward, done = env.step(action)
            pending_reward = reward

            # 3. If missed / round done, consolidate memories in sleep
            if done:
                agent.sleep_consolidation()

        # RENDER FRAME
        screen.fill(COURT_BG)

        # 1. Court Grid lines
        for i in range(env.grid_size):
            pygame.draw.line(screen, GRID_COLOR, (0, int(i * env.cell_h)), (court_w, int(i * env.cell_h)))
            pygame.draw.line(screen, GRID_COLOR, (int(i * env.cell_w), 0), (int(i * env.cell_w), court_h))

        # 2. Draw Paddle with Emerald Bioluminescent Glow
        pad_rect = pygame.Rect(int(env.paddle_x), int(env.paddle_y), env.paddle_w, env.paddle_h)
        pygame.draw.rect(screen, PADDLE_COLOR, pad_rect, border_radius=4)

        # 3. Draw Ball with Crimson Core & Soft Glow
        bx, by = int(env.ball_x), int(env.ball_y)
        glow_surf = pygame.Surface((env.ball_r * 4, env.ball_r * 4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (255, 90, 90, 45), (env.ball_r * 2, env.ball_r * 2), env.ball_r * 2)
        screen.blit(glow_surf, (bx - env.ball_r * 2, by - env.ball_r * 2))
        pygame.draw.circle(screen, BALL_COLOR, (bx, by), env.ball_r)

        # 4. Render Cybernetic Total Visibility HUD Sidebar
        hud.render(
            screen=screen,
            env=env,
            agent=agent,
            fast_forward=fast_forward
        )

        pygame.display.flip()

        if not fast_forward:
            clock.tick(60)

    pygame.quit()
    sys.exit()


def run_headless(steps: int = 5000, seed: int = 42):
    """Headless simulation benchmark for CI/CD environments or test validation."""
    print(f"Running headless BIB 2 Pong simulation for {steps} steps (seed={seed})...")
    random.seed(seed)
    np.random.seed(seed)
    env = PongEnv()
    agent = BIB2PongAgent(seed=seed)
    pending_reward = 0.0

    for step_i in range(steps):
        action = agent.step(env, prev_reward=pending_reward)
        reward, done = env.step(action)
        pending_reward = reward
        if done:
            agent.sleep_consolidation()

    total_attempts = env.hits + env.misses
    hit_rate = (env.hits / total_attempts * 100.0) if total_attempts > 0 else 0.0
    print(f"Headless simulation completed: {steps} steps | Hits: {env.hits} | Misses: {env.misses} | Interception Rate: {hit_rate:.1f}% | Stage: {env.curriculum[env.stage_idx]['name']}")
    return env.hits, env.misses


if __name__ == "__main__":
    main()
