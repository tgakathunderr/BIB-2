"""Hypothalamic-Pituitary-Adrenal (HPA) Endocrine Stress Axis."""

import numpy as np


class HPAAxis:
    """
    Hypothalamic-Pituitary-Adrenal (HPA) Axis.
    Stress cascade:
    1. Hypothalamic PVN releases Corticotropin-Releasing Hormone (CRH).
    2. Anterior pituitary releases Adrenocorticotropic Hormone (ACTH) into portal system.
    3. Adrenal cortex mobilizes Cortisol to adaptively elevate vigilance and glucose availability.
    4. Negative feedback: Circulating cortisol inhibits hypothalamic CRH and pituitary ACTH.
    """

    def __init__(self):
        self.crh: float = 0.1
        self.acth: float = 0.1
        self.cortisol: float = 0.1

    def trigger_stress(self, stress_level: float) -> None:
        """Acute stress triggers CRH release."""
        stim = float(np.clip(stress_level, 0.0, 1.0))
        self.crh = float(np.clip(self.crh + stim * 0.8, 0.0, 1.0))
        self.acth = float(np.clip(self.acth + self.crh * 0.75, 0.0, 1.0))
        self.cortisol = float(np.clip(self.cortisol + self.acth * 0.7, 0.0, 1.0))

    def step(self, dt: float = 1.0) -> None:
        """
        Advance endocrine cascade and apply glucocorticoid negative feedback.
        """
        # Negative feedback: Cortisol suppresses upstream CRH and ACTH
        feedback = self.cortisol * 0.25 * dt
        self.crh = max(0.1, self.crh - feedback)
        self.acth = max(0.1, self.acth - feedback * 0.8)

        # Cortisol slow metabolic clearance
        clearance = 0.05 * dt
        self.cortisol = max(0.1, self.cortisol - clearance)
