"""Peripheral Nervous System interfaces for BIB 2."""

from .cranial import CranialNervesInterface
from .spinal_nerves import SpinalNervesInterface
from .autonomic import AutonomicNervousSystem, EntericNervousSystem
from .bus import UniversalNeuralBus

__all__ = [
    "CranialNervesInterface",
    "SpinalNervesInterface",
    "AutonomicNervousSystem",
    "EntericNervousSystem",
    "UniversalNeuralBus",
]
