"""BIB 2: Biologically Inspired Brain 2.

A 1:1 functional macro-anatomical and circuit replica of the complete Human Nervous System.
"""

__version__ = "2.0.0"

from .brain import BIB2NervousSystem
from .types import (
    NeuralBusState,
    CranialNerveSignal,
    SpinalNerveSignal,
    AutonomicState,
    ChemicalState,
    TractData,
)

__all__ = [
    "BIB2NervousSystem",
    "NeuralBusState",
    "CranialNerveSignal",
    "SpinalNerveSignal",
    "AutonomicState",
    "ChemicalState",
    "TractData",
    "__version__",
]
