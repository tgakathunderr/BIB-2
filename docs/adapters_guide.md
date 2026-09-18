# Custom Domain Adapters Developer Guide

**BIB 2** was engineered from day one to be multi-domain and embodiment-agnostic. The nervous system communicates with the external world strictly through its standardized **Universal Neural Bus** (`bib2.peripheral.bus.UniversalNeuralBus`).

This guide demonstrates how to build custom adapters to connect BIB 2 to any physical robot, virtual environment, sensor suite, or game engine.

---

## 1. The `BaseNeuralAdapter` Pattern

All domain adapters inherit from `BaseNeuralAdapter` defined in [`bib2/adapters/base.py`](../bib2/adapters/base.py):

```python
from bib2.brain import BIB2NervousSystem
from bib2.adapters.base import BaseNeuralAdapter

class MyCustomAdapter(BaseNeuralAdapter):
    def __init__(self, brain: BIB2NervousSystem):
        super().__init__(brain)
        # Custom initialization...
```

The adapter is responsible for:
1. **Translating Domain Inputs $\to$ Biological Afferents**: Ingesting external camera frames, audio streams, force sensors, or feature vectors into appropriate Cranial Nerves (`CN_I`–`CN_XII`) or Spinal Dermatomes (`C1`–`Co1`).
2. **Advancing the Clock Tick**: Calling `self.brain.tick()`.
3. **Translating Biological Efferents $\to$ Domain Outputs**: Reading motor commands from Cranial Nerves (e.g. `CN_XII` for speech, `CN_III` for saccades) or Spinal Myotomes (e.g. `C5`–`C8` for arm torques, `L2`–`S1` for leg locomotion).

---

## 2. Standard Biological Mapping Conventions

| Domain Signal | Biological Conduit | Nervous System Mechanism |
|---|---|---|
| **Visual images / cameras** | `CN_II` (Optic Nerve) | Projects to Thalamus LGN $\to$ V1 $\to$ Dorsal/Ventral streams |
| **Audio / microphone** | `CN_VIII` (Cochlear) | Projects to Thalamus MGN $\to$ A1 $\to$ Wernicke's area |
| **IMU / Balance / Gyro** | `CN_VIII` (Vestibular) | Projects to Vestibular nuclei $\to$ Cerebellum for balance |
| **Tactile touch / Pressure** | Spinal Dermatomes (`C5`, `L2`) | Projects to Spinal dorsal horn $\to$ DCML lemniscal tract $\to$ S1 |
| **Pain / Thermal warning** | Spinal Dermatomes (high amplitude) | Projects to Spinothalamic tract $\to$ Amygdala threat response |
| **Reward / Success (+)** | Dopamine / Enteric Satiety | Stimulates SNc dopamine release $\to$ Basal Ganglia D1 Go pathway |
| **Punishment / Collision (-)** | Lateral Habenula / Sympathetic | Activates LHb anti-reward $\to$ Depresses DA, surges sympathetic tone |
| **Upper body joint torques** | Cervical Myotomes (`C5`–`C8`) | Decussated corticospinal motor commands calibrated by Cerebellum |
| **Lower limb locomotion** | Lumbar/Sacral Myotomes (`L2`–`S2`)| Corticospinal motor pool driving locomotor central pattern generators |
| **Eye tracking / Gaze** | `CN_III`, `CN_IV`, `CN_VI` | Superior colliculus & Frontal Eye Fields conjugate saccadic target |
| **Speech / Text output** | `CN_XII` (Hypoglossal) / Broca | Motor articulation sequence from Broca's area (BA 44/45) |

---

## 3. Step-by-Step Example: Building an Autonomous Driving Adapter

Below is a complete, working example implementing a self-driving vehicle adapter:

```python
import numpy as np
from typing import Tuple, Optional
from bib2.brain import BIB2NervousSystem
from bib2.adapters.base import BaseNeuralAdapter

class AutonomousVehicleAdapter(BaseNeuralAdapter):
    """
    Adapter interfacing a simulated vehicle or real car with BIB 2:
    - Forward camera -> CN II (Optic)
    - LiDAR obstacle distances -> Spinal dermatome C5 (Tactile/Distance)
    - Vehicle speed & steering angle -> CN VIII (Vestibular/IMU)
    - Steering torque & brake/throttle -> Spinal myotomes
    """

    def __init__(self, brain: BIB2NervousSystem):
        super().__init__(brain)

    def control_step(
        self,
        camera_frame: np.ndarray,      # e.g. (64, 64) grayscale road view
        lidar_ranges: np.ndarray,       # e.g. (16,) distance beam array
        imu_velocity: float,            # current speed in m/s
        collision_detected: bool = False
    ) -> Tuple[float, float, float]:
        """
        Returns: (steering_angle, throttle, brake)
        """
        # 1. Ingest camera into CN II
        cam_flat = camera_frame.flatten()
        indices = np.linspace(0, cam_flat.size - 1, self.brain.dim, dtype=int)
        self.brain.peripheral.cranial.set_sensory("CN_II", cam_flat[indices])

        # 2. Ingest LiDAR into C5 dermatome
        self.brain.peripheral.spinal.set_dermatome("C5", lidar_ranges[:16])

        # 3. Ingest velocity into CN VIII
        speed_vec = np.full(self.brain.dim, fill_value=float(imu_velocity) / 30.0, dtype=np.float32)
        self.brain.peripheral.cranial.set_sensory("CN_VIII", speed_vec)

        # 4. Handle collision punishment
        if collision_detected:
            # Trigger sympathetic alarm & lateral habenula negative reward
            self.brain.peripheral.autonomic.trigger_sympathetic_surge(intensity=1.0)
            self.brain.habenula.compute_anti_reward(expected_reward=0.8, received_reward=0.0)

        # 5. Execute biological clock tick
        self.brain.tick()

        # 6. Extract vehicle actuation commands from spinal motor pools
        c5_myotome = self.brain.peripheral.spinal.get_myotome("C5") # 8-dim motor vector
        
        # Steering mapped from differential motor channels 0 and 1
        steering = float(np.tanh(c5_myotome[0] - c5_myotome[1]))
        # Throttle from forward motor channel 2
        throttle = float(np.clip(c5_myotome[2], 0.0, 1.0))
        # Brake activated if sympathetic alarm or collision fear is elevated
        brake = float(np.clip(self.brain.peripheral.autonomic.sympathetic_tone - 0.4, 0.0, 1.0))

        return steering, throttle, brake
```

---

## 4. Testing Your Custom Adapter

You can test custom adapters in Python without spinning up external robotics simulators:

```python
def test_vehicle_adapter():
    brain = BIB2NervousSystem(seed=42)
    adapter = AutonomousVehicleAdapter(brain)

    fake_cam = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
    fake_lidar = np.ones(16, dtype=np.float32) * 5.0
    
    steer, throttle, brake = adapter.control_step(fake_cam, fake_lidar, imu_velocity=12.5)
    
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0
```
