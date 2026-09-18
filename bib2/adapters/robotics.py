"""Robotics Domain Adapter: Visual, IMU, tactile sensing and joint torque control."""

from typing import TYPE_CHECKING, Optional
import numpy as np
from .base import BaseNeuralAdapter

if TYPE_CHECKING:
    from ..brain import BIB2NervousSystem


class RoboticsAdapter(BaseNeuralAdapter):
    """
    Adapter for robotic embodiment:
    - Vision (camera frame) -> CN II (Optic Nerve)
    - IMU telemetry (6-DOF accel/gyro) -> CN VIII (Vestibulocochlear Nerve)
    - Tactile arrays -> Spinal dermatomes (C5)
    - Motor outputs -> Spinal myotomes -> Joint torques
    """

    def __init__(self, brain: "BIB2NervousSystem", num_joints: int = 6):
        super().__init__(brain)
        self.num_joints = num_joints

    def step(
        self,
        camera_frame: np.ndarray,
        imu_telemetry: np.ndarray,
        tactile_touch: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Ingest sensor telemetry, execute 1 brain tick, and return joint torques.
        """
        # 1. Process visual input -> CN II
        cam_arr = np.asarray(camera_frame, dtype=np.float32)
        if cam_arr.ndim > 1:
            flat_cam = cam_arr.flatten()
            if flat_cam.size >= self.brain.dim:
                indices = np.linspace(0, flat_cam.size - 1, self.brain.dim, dtype=int)
                vis_vec = flat_cam[indices]
            else:
                vis_vec = np.pad(flat_cam, (0, self.brain.dim - flat_cam.size))
        else:
            vis_vec = cam_arr
        self.brain.peripheral.cranial.set_sensory("CN_II", vis_vec)

        # 2. Process IMU (6-DOF) -> CN VIII (Vestibulocochlear)
        imu_arr = np.asarray(imu_telemetry, dtype=np.float32).flatten()
        if imu_arr.size > 0:
            repeats = int(np.ceil(self.brain.dim / imu_arr.size))
            vestibular_vec = np.tile(imu_arr, repeats)[:self.brain.dim]
        else:
            vestibular_vec = np.zeros(self.brain.dim, dtype=np.float32)
        self.brain.peripheral.cranial.set_sensory("CN_VIII", vestibular_vec)

        # 3. Process tactile touch -> C5 dermatome
        if tactile_touch is not None:
            self.brain.peripheral.spinal.set_dermatome("C5", tactile_touch)

        # 4. Execute brain clock tick
        self.brain.tick()

        # 5. Extract joint torques from spinal efferent myotomes
        myo_c5 = self.brain.peripheral.spinal.get_myotome("C5")
        myo_c6 = self.brain.peripheral.spinal.get_myotome("C6")
        combined_myo = np.concatenate([myo_c5, myo_c6])
        if combined_myo.size >= self.num_joints:
            torques = combined_myo[:self.num_joints].copy()
        else:
            torques = np.tile(combined_myo, int(np.ceil(self.num_joints / max(1, combined_myo.size))))[:self.num_joints].copy()

        # Ensure finite values
        torques = np.nan_to_num(torques, nan=0.0, posinf=1.0, neginf=-1.0)
        return torques.astype(np.float32)
