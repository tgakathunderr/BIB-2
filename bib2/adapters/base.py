"""Base adapter class connecting domain-specific tasks to BIB 2 Nervous System."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional
import numpy as np

if TYPE_CHECKING:
    from ..brain import BIB2NervousSystem


class BaseNeuralAdapter(ABC):
    """Abstract base adapter interfacing external domains with the universal neural bus."""

    def __init__(self, brain: "BIB2NervousSystem"):
        self.brain = brain
