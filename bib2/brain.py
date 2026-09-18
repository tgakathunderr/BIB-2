"""BIB 2: Biologically Inspired Brain 2 - Master Human Nervous System Container."""

from typing import Any, Dict, List, Optional
import numpy as np

from .types import NeuralBusState, CranialNerveSignal, SpinalNerveSignal, ChemicalState, AutonomicState
from .peripheral.bus import UniversalNeuralBus
from .spinal.cord import SpinalCordEngine
from .brainstem import BrainstemComplex
from .cerebellum.forward_model import CerebellumEngine
from .diencephalon.thalamus import ThalamusComplex
from .diencephalon.trn import ThalamicReticularNucleus
from .diencephalon.hypothalamus import HypothalamicComplex
from .diencephalon.habenula import LateralHabenula
from .telencephalon.basal_ganglia import BasalGangliaComplex
from .telencephalon.limbic import LimbicComplex
from .telencephalon.amygdala import AmygdaloidComplex
from .telencephalon.claustrum import ClaustrumSynchronizer
from .telencephalon.neocortex.areas import CerebralNeocortex
from .telencephalon.neocortex.networks import TriNetworkCoordinator
from .chemistry.matrix import ChemicalMatrix
from .chemistry.hpa import HPAAxis
from .chemistry.plasticity import PlasticityEngine
from .sleep.orchestrator import SleepOrchestrator, SleepStage
from .sleep.ripples import SharpWaveRippleEngine
from .sleep.shy import TononiSHYDownscaling
from .sleep.glymphatic import GlymphaticEngine


class ChemistrySystem:
    """Convenience container for chemistry and plasticity systems."""
    def __init__(self):
        self.matrix = ChemicalMatrix()
        self.hpa = HPAAxis()
        self.plasticity = PlasticityEngine()


class SleepSystem:
    """Convenience container for sleep and consolidation systems."""
    def __init__(self):
        self.orchestrator = SleepOrchestrator()
        self.ripples = SharpWaveRippleEngine()
        self.shy = TononiSHYDownscaling()
        self.glymphatic = GlymphaticEngine()

    @property
    def current_stage(self) -> SleepStage:
        return self.orchestrator.current_stage

    def transition_to(self, stage: SleepStage) -> None:
        self.orchestrator.transition_to(stage)


class BIB2NervousSystem:
    """
    BIB 2: Biologically Inspired Brain 2.
    Complete 1:1 macro-anatomical and functional circuit replica of the Human Nervous System:
    - Peripheral Nervous System (CN I-XII, 31 Spinal Segments, Sympathetic, Parasympathetic Vagus, Enteric)
    - Spinal Cord Engine (Rexed Laminae I-X, Ia Stretch & Crossed-Extensor Reflexes, Ascending & Descending Tracts)
    - Brainstem Complex (Pre-Bötzinger, Medulla Baroreflex, Inferior Olive, Pons, Midbrain Colliculi/PAG)
    - Cerebellum Engine (Forward Predictive Smith Model, Purkinje LTD Plasticity, Deep Cerebellar Nuclei)
    - Diencephalon (Thalamus Relay Nuclei, TRN Gating Shell, SCN Circadian Clock, Hypothalamus, Habenula)
    - Telencephalon (Basal Ganglia 3 pathways, Hippocampal Trisynaptic Circuit & Papez Loop, Amygdala, Claustrum)
    - Cerebral Neocortex (6-layer canonical microcircuit across 14 Brodmann areas, Visual streams, Tri-Network)
    - Chemical Matrix & HPA Axis (DA, 5-HT, NE, ACh, Histamine, Cortisol)
    - 3-Stage Sleep Engine & Glymphatic Clearance (NREM -> SWS SWR Ripples -> REM, Tononi SHY, AQP4 flush)
    """

    def __init__(self, feature_dim: int = 64, seed: int = 42):
        self.dim = feature_dim
        self.seed = seed
        self.tick_count: int = 0

        # 1. Peripheral Interface
        self.peripheral = UniversalNeuralBus(cranial_dim=feature_dim)

        # 2. Spinal Cord
        self.spinal_cord = SpinalCordEngine(lamina_dim=feature_dim)

        # 3. Brainstem
        self.brainstem = BrainstemComplex(feature_dim=feature_dim)

        # 4. Cerebellum
        self.cerebellum = CerebellumEngine(n_mossy=feature_dim, n_granule=256, n_purkinje=feature_dim, seed=seed)

        # 5. Diencephalon
        self.thalamus = ThalamusComplex(dim=feature_dim)
        self.trn = ThalamicReticularNucleus(dim=feature_dim)
        self.hypothalamus = HypothalamicComplex()
        self.habenula = LateralHabenula()

        # 6. Telencephalon
        self.basal_ganglia = BasalGangliaComplex(n_actions=8)
        self.limbic = LimbicComplex(dim=feature_dim)
        self.amygdala = AmygdaloidComplex(dim=feature_dim)
        self.claustrum = ClaustrumSynchronizer(dim=feature_dim)
        self.neocortex = CerebralNeocortex(dim=feature_dim)
        self.tri_network = TriNetworkCoordinator()

        # 7. Chemistry & Plasticity
        self.chemistry = ChemistrySystem()

        # 8. Sleep & Waste Clearance
        self.sleep = SleepSystem()

        # Memory episodic trajectory buffer
        self.episodic_memory: List[np.ndarray] = []

    def get_bus(self) -> NeuralBusState:
        """Return snapshot of current peripheral bus state."""
        return self.peripheral.snapshot()

    def tick(self, sensory_bus: Optional[NeuralBusState] = None) -> NeuralBusState:
        """
        Execute deterministic 19-stage biological clock cycle (Delta t).
        """
        self.tick_count += 1
        dt = 0.1  # 100 ms simulation step

        # If incoming sensory bus state provided, ingest into peripheral interfaces
        if sensory_bus is not None:
            for cid, sig in sensory_bus.cranial_nerves.items():
                if cid in self.peripheral.cranial.nerves:
                    self.peripheral.cranial.set_sensory(cid, sig.data)
            for seg, sig in sensory_bus.spinal_nerves.items():
                if seg in self.peripheral.spinal.segments:
                    self.peripheral.spinal.set_dermatome(seg, sig.dermatome)

        # 1. SENSORY INGESTION
        cn2_visual = self.peripheral.cranial.get_sensory("CN_II")
        cn8_audio = self.peripheral.cranial.get_sensory("CN_VIII")
        spinal_touch = self.peripheral.spinal.get_dermatome("C5")

        # 2. SPINAL CORD REFLEXES & ASCENDING TRACTS
        # Local monosynaptic reflex at C5
        reflex_myotome = self.spinal_cord.process_segment_reflex("C5", spinal_touch)
        self.peripheral.spinal.set_myotome("C5", reflex_myotome)

        # Ascending conduction into DCML (touch) and Spinothalamic (pain/temp)
        dcml_out, stt_out = self.spinal_cord.conduct_ascending(spinal_touch, spinal_touch * 0.2)

        # 3. BRAINSTEM REFLEXES & VEGETATIVE LIFE SUPPORT
        bs_out = self.brainstem.step(
            bp_sys=self.peripheral.autonomic.blood_pressure_sys,
            visual_stim=cn2_visual,
            is_rem=(self.sleep.current_stage == SleepStage.REM),
            dt=dt
        )
        # Saccadic eye tracking via CN III / IV / VI
        saccade = bs_out["saccade_target"]
        self.peripheral.cranial.set_motor("CN_III", np.tile(saccade, 32)[:self.dim])

        # 4. THALAMIC RELAY & TRN GATING
        # Feed sensory inputs into specific thalamic nuclei
        self.thalamus.relay_sensory("LGN", cn2_visual)
        self.thalamus.relay_sensory("MGN", cn8_audio)
        self.thalamus.relay_sensory("VPL", dcml_out)

        # TRN applies attentional gating spotlight
        gated_lgn = self.trn.gate_thalamic_output("LGN", self.thalamus.read_nucleus("LGN"))
        gated_mgn = self.trn.gate_thalamic_output("MGN", self.thalamus.read_nucleus("MGN"))
        gated_vpl = self.trn.gate_thalamic_output("VPL", self.thalamus.read_nucleus("VPL"))

        # 5. THALAMOCORTICAL PROJECTION TO NEOCORTEX LAYER IV
        # 6. NEOCORTEX CANONICAL MICROCIRCUIT
        v1_out = self.neocortex.registry.forward_area("V1_Visual", gated_lgn)
        a1_out = self.neocortex.registry.forward_area("A1_PrimaryAuditory", gated_mgn)
        s1_out = self.neocortex.registry.forward_area("S1_Somatosensory", gated_vpl)

        # 7. TRANS-MODAL BIFURCATION & CLAUSTRUM SYNCHRONIZATION
        dorsal_where, ventral_what = self.neocortex.process_visual_bifurcation(v1_out)
        bound_sdr = self.claustrum.synchronize([v1_out, a1_out, s1_out, ventral_what])

        # 8. INSULAR CORTEX & SALIENCE NETWORK SWITCH
        salience_stim = float(np.mean(bound_sdr) > 0.4)
        tri_state = self.tri_network.step(salience_event=bool(salience_stim), dt=dt)
        self.neocortex.registry.forward_area("Insula_Interoception", bound_sdr * 0.5)

        # 9. PREFRONTAL CORTEX (DLPFC) WORKING MEMORY
        dlpfc_out = self.neocortex.registry.forward_area("DLPFC_WorkingMemory", bound_sdr * tri_state["CEN"])

        # 10. HIPPOCAMPUS & CIRCUIT OF PAPEZ
        # DG separation -> CA3 completion -> Subiculum -> Papez loop
        subiculum_out = self.limbic.hippocampus.process(bound_sdr)
        papez_out = self.limbic.papez.cycle(subiculum_out)
        self.episodic_memory.append(subiculum_out.copy())
        if len(self.episodic_memory) > 100:
            self.episodic_memory.pop(0)

        # 11. AMYGDALA SALIENCE & THREAT EVALUATION
        salience_val, fear_drive = self.amygdala.evaluate_salience(bound_sdr)
        if fear_drive > 0.5:
            self.peripheral.autonomic.trigger_sympathetic_surge(fear_drive)

        # 12. LATERAL HABENULA DISAPPOINTMENT CHECK
        expected_r = float(np.mean(dlpfc_out))
        received_r = float(self.peripheral.enteric.satiety)
        anti_reward = self.habenula.compute_anti_reward(expected_r, received_r)
        if anti_reward > 0.5:
            # Depress dopamine in chemical matrix
            self.chemistry.matrix.state.dopamine = max(0.1, self.chemistry.matrix.state.dopamine - 0.2)

        # 13. BASAL GANGLIA TRIPARTITE ACTION SELECTION
        # Action proposals generated by Premotor / SMA & DLPFC
        proposals = dlpfc_out[:8]
        da_tone = self.chemistry.matrix.state.dopamine
        winner_idx, disinhibition = self.basal_ganglia.select_action(proposals, dopamine_level=da_tone)

        # 14. WINNING ACTION DISINHIBITION INTO M1
        m1_motor_command = np.zeros(self.dim, dtype=np.float32)
        if disinhibition > 0.1:
            m1_motor_command[winner_idx * 8:(winner_idx + 1) * 8] = disinhibition
            self.thalamus.set_nucleus("VA_VL", m1_motor_command)
            m1_out = self.neocortex.registry.forward_area("M1_PrimaryMotor", m1_motor_command)
        else:
            m1_out = np.zeros(self.dim, dtype=np.float32)

        # 15. CEREBELLAR FORWARD MODEL CALIBRATION
        predicted_state, micro_correction = self.cerebellum.step(
            intended_motor_command=m1_out,
            current_sensory_state=dcml_out
        )
        calibrated_motor = np.clip(m1_out + micro_correction * 0.5, 0.0, 1.0)

        # 16. CORTICOSPINAL DESCENDING MOTOR CONDUCTION
        lateral_motor, anterior_motor = self.spinal_cord.conduct_descending(calibrated_motor)

        # 17. SPINAL MOTOR POOL ACTUATION & CRANIAL SPEECH
        # Update spinal myotomes across segments
        self.peripheral.spinal.set_all_efferents(np.tile(lateral_motor, 4)[:248])
        # Language articulation via Broca -> CN XII
        broca_speech = self.neocortex.process_language_repetition(a1_out)
        self.peripheral.cranial.set_motor("CN_XII", broca_speech)

        # 18. AUTONOMIC & ENTERIC HOMEOSTASIS
        self.peripheral.step_vegetative(dt=dt)

        # 19. CHEMICAL MATRIX & CIRCADIAN SCN UPDATE
        self.chemistry.matrix.step_clearance(dt=dt)
        self.chemistry.matrix.accumulate_adenosine(amount=0.005)
        self.hypothalamus.step(glucose=self.peripheral.enteric.satiety, leptin=0.5, dt_hours=0.01)

        return self.peripheral.snapshot()

    def tick_sleep(self) -> None:
        """
        Execute one sleep cycle iteration:
        - Sharp-Wave Ripples (SWRs) memory replay from Hippocampus to Neocortex
        - Tononi Synaptic Homeostasis (SHY) downscaling
        - Glymphatic convective waste clearance
        """
        if self.sleep.current_stage == SleepStage.NREM_SWS:
            # 1. SWR Replay
            if self.episodic_memory:
                replayed = self.sleep.ripples.replay_episodes(self.episodic_memory[-5:])
                # Transfer consolidated trace into Neocortex association areas
                for rep in replayed:
                    self.neocortex.registry.forward_area("IPL_MultimodalHub", rep)

            # 2. Tononi SHY downscaling across all cortical areas
            for area in self.neocortex.registry.areas.values():
                area.w_thal_l4 = self.sleep.shy.downscale(area.w_thal_l4, downscale_factor=0.95)
                area.w_l4_l23  = self.sleep.shy.downscale(area.w_l4_l23, downscale_factor=0.95)
                area.w_l23_l5  = self.sleep.shy.downscale(area.w_l23_l5, downscale_factor=0.95)

            # 3. Glymphatic convective wash
            self.sleep.glymphatic.flush_parenchyma(current_solute_load=50.0, is_sws=True)
            self.chemistry.matrix.clear_adenosine(amount=0.05)

    def recall_episode(self, index: int = -1) -> Optional[np.ndarray]:
        """Recall consolidated episodic memory trace from hippocampal CA3."""
        if not self.episodic_memory:
            return None
        idx = max(0, min(len(self.episodic_memory) - 1, index if index >= 0 else len(self.episodic_memory) + index))
        stored = self.episodic_memory[idx]
        return self.limbic.hippocampus.ca3.complete(stored)
