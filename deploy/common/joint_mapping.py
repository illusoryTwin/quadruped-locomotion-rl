"""
Joint order mapping between Isaac Lab and MuJoCo/Unitree SDK.

Isaac Lab order (by joint type):
    FL_hip, FR_hip, RL_hip, RR_hip,
    FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    FL_calf, FR_calf, RL_calf, RR_calf

MuJoCo/Unitree SDK order (by leg):
    FR_hip, FR_thigh, FR_calf,
    FL_hip, FL_thigh, FL_calf,
    RR_hip, RR_thigh, RR_calf,
    RL_hip, RL_thigh, RL_calf
"""

import numpy as np

# Isaac Lab index -> MuJoCo index mapping
# mujoco_action[i] = isaac_action[ISAAC_TO_MUJOCO[i]]
ISAAC_TO_MUJOCO = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10], dtype=np.int32)

# MuJoCo index -> Isaac Lab index mapping
# isaac_obs[i] = mujoco_obs[MUJOCO_TO_ISAAC[i]]
MUJOCO_TO_ISAAC = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.int32)


def isaac_to_mujoco(isaac_array: np.ndarray) -> np.ndarray:
    """Convert joint array from Isaac Lab order to MuJoCo order."""
    return isaac_array[ISAAC_TO_MUJOCO]


def mujoco_to_isaac(mujoco_array: np.ndarray) -> np.ndarray:
    """Convert joint array from MuJoCo order to Isaac Lab order."""
    return mujoco_array[MUJOCO_TO_ISAAC]
