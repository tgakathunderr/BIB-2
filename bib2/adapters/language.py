"""Language Domain Adapter: Text encoding, Wernicke-Broca loop, and motor speech decoding."""

from typing import TYPE_CHECKING, List, Optional
import numpy as np
from .base import BaseNeuralAdapter

if TYPE_CHECKING:
    from ..brain import BIB2NervousSystem


DEFAULT_VOCAB = [
    "neural", "synapse", "cortex", "thalamus", "circuit", "action",
    "signal", "reflex", "memory", "plasticity", "biological", "intelligence",
    "cognitive", "active", "processing", "conscious", "perception", "response",
    "pathway", "motor", "sensory", "pattern", "adaptation", "resonance"
]


class LanguageAdapter(BaseNeuralAdapter):
    """
    Adapter for natural language understanding and generation:
    - Text prompt -> phonemic/orthographic encoding -> CN II (reading) & CN VIII (auditory language)
    - Wernicke's area (BA22) -> Arcuate fasciculus -> Broca's area (BA44/45)
    - CN XII (Hypoglossal) motor speech articulation -> Decoded linguistic response
    """

    def __init__(self, brain: "BIB2NervousSystem", vocab: Optional[List[str]] = None):
        super().__init__(brain)
        self.vocab = vocab if vocab is not None else DEFAULT_VOCAB
        self._vocab_embeddings = self._build_vocab_embeddings()

    def _build_vocab_embeddings(self) -> np.ndarray:
        """Precompute normalized semantic embeddings for vocabulary items."""
        embeds = np.zeros((len(self.vocab), self.brain.dim), dtype=np.float32)
        for i, word in enumerate(self.vocab):
            embeds[i] = self.encode_text(word)
        return embeds

    def encode_text(self, text: str) -> np.ndarray:
        """Deterministically encode a string into a cortical feature vector."""
        vec = np.zeros(self.brain.dim, dtype=np.float32)
        if not text:
            return vec
        for i, char in enumerate(text):
            idx = (ord(char) * 17 + i * 31) % self.brain.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-6:
            vec = vec / norm
        return vec

    def decode_speech(self, speech_vec: np.ndarray, top_k: int = 3) -> str:
        """Decode motor speech activation vector into coherent linguistic tokens."""
        s_arr = np.asarray(speech_vec, dtype=np.float32)
        norm = np.linalg.norm(s_arr)
        if norm < 1e-6:
            return "neural response active"

        normalized = s_arr / norm
        scores = np.dot(self._vocab_embeddings, normalized)
        top_indices = np.argsort(scores)[::-1][:top_k]
        selected_tokens = [self.vocab[idx] for idx in top_indices]
        return " ".join(selected_tokens)

    def process_text(self, text: str) -> str:
        """Ingest text string, execute brain clock tick, and return generated speech."""
        encoded = self.encode_text(text)
        # Ingest into Auditory (CN VIII) and Reading (CN II)
        self.brain.peripheral.cranial.set_sensory("CN_VIII", encoded)
        self.brain.peripheral.cranial.set_sensory("CN_II", encoded)

        # Run biological clock cycle
        self.brain.tick()

        # Read articulated motor speech from CN XII (Hypoglossal)
        speech_motor = self.brain.peripheral.cranial.get_motor("CN_XII")
        if np.linalg.norm(speech_motor) < 1e-6:
            # Fallback to Broca L5 output directly
            speech_motor = self.brain.neocortex.registry.get_area("Broca_SpeechProduction").l5_output

        response = self.decode_speech(speech_motor)
        return response

    def process_input(self, text: str) -> str:
        """Alias for process_text."""
        return self.process_text(text)
