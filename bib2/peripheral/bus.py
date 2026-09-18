"""Universal Neural Bus coordinating peripheral nervous trunks."""

import numpy as np
from .cranial import CranialNervesInterface
from .spinal_nerves import SpinalNervesInterface
from .autonomic import AutonomicNervousSystem, EntericNervousSystem
from ..types import NeuralBusState


class UniversalNeuralBus:
    """Standardized peripheral nervous bus connecting domain adapters to the CNS."""

    def __init__(self, cranial_dim: int = 64, dermatome_dim: int = 16, myotome_dim: int = 8):
        self.cranial = CranialNervesInterface(feature_dim=cranial_dim)
        self.spinal = SpinalNervesInterface(dermatome_dim=dermatome_dim, myotome_dim=myotome_dim)
        self.autonomic = AutonomicNervousSystem()
        self.enteric = EntericNervousSystem()

    def snapshot(self) -> NeuralBusState:
        """Capture the current state of all peripheral nerve channels."""
        state = NeuralBusState(
            cranial_nerves={cid: self.cranial.nerves[cid] for cid in self.cranial.nerves},
            spinal_nerves={seg: self.spinal.segments[seg] for seg in self.spinal.segments},
            autonomic=self.autonomic.state,
            enteric_satiety=self.enteric.satiety,
            enteric_motility=self.enteric.motility,
        )
        return state

    def step_vegetative(self, dt: float = 1.0) -> None:
        """Step autonomic and enteric homeostatic decay."""
        self.autonomic.step_homeostasis(dt=dt)
        self.enteric.step(dt=dt)
