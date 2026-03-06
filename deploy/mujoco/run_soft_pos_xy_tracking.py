#!/usr/bin/env python3
"""
Deploy go2_soft_pos_xy_tracking policy with unitree_mujoco.

This policy was trained with position commands (body-frame velocity from
position error P-controller) and stiffness commands (compliance kp).

Usage:
    1. Start unitree_mujoco:
       cd ~/Workspace/Projects/unitree_robotics/unitree_mujoco/simulate_python
       python unitree_mujoco.py

    2. Run this script (velocity command mode - default):
       python -m deploy.mujoco.run_soft_pos_xy_tracking --vx 0.5 --vy 0.0 --kp 20.0

    3. Or with a specific checkpoint:
       python -m deploy.mujoco.run_soft_pos_xy_tracking \
           --policy logs/rsl_rl/unitree_go2_walk/2026-03-01_18-46-12/exported/policy.pt \
           --vx 0.3 --vy 0.0 --kp 15.0
"""

import argparse
import time
import numpy as np
import torch

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Default policy path (2026-03-01_18-46-12 training run)
DEFAULT_POLICY = (
    "logs/rsl_rl/unitree_go2_walk/2026-03-01_18-46-12/exported/policy.pt"
)

# =========================================================================
# Joint ordering
# =========================================================================
# Isaac Lab order (by joint type - how the policy was trained):
#   FL_hip, FR_hip, RL_hip, RR_hip,
#   FL_thigh, FR_thigh, RL_thigh, RR_thigh,
#   FL_calf, FR_calf, RL_calf, RR_calf
#
# MuJoCo/SDK order (by leg - what the simulator uses):
#   FR_hip, FR_thigh, FR_calf,
#   FL_hip, FL_thigh, FL_calf,
#   RR_hip, RR_thigh, RR_calf,
#   RL_hip, RL_thigh, RL_calf
# =========================================================================

# mujoco_array = isaac_array[ISAAC_TO_MUJOCO]
ISAAC_TO_MUJOCO = np.array([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10], dtype=np.int32)

# isaac_array = mujoco_array[MUJOCO_TO_ISAAC]
MUJOCO_TO_ISAAC = np.array([3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8], dtype=np.int32)

# Default joint positions in Isaac Lab order
# FL_hip, FR_hip, RL_hip, RR_hip = 0.1, -0.1, 0.1, -0.1
# FL_thigh, FR_thigh, RL_thigh, RR_thigh = 0.8, 0.8, 1.0, 1.0
# FL_calf, FR_calf, RL_calf, RR_calf = -1.5, -1.5, -1.5, -1.5
DEFAULT_JOINT_POS_ISAAC = np.array([
    0.1, -0.1, 0.1, -0.1,       # hips:   FL, FR, RL, RR
    0.8,  0.8, 1.0,  1.0,       # thighs: FL, FR, RL, RR
    -1.5, -1.5, -1.5, -1.5,     # calves: FL, FR, RL, RR
], dtype=np.float32)

# =========================================================================
# Observation structure (per frame, Isaac Lab order)
#   base_ang_vel:        3
#   projected_gravity:   3
#   joint_pos (rel):    12
#   joint_vel:          12
#   actions (last):     12
#   position_commands:   2  (body-frame vx, vy from position P-controller)
#   stiffness_commands:  1  (compliance kp)
#   --------------------------
#   Single frame:       45
#   x10 history:       450
#
# History is flattened per-term (Isaac Lab flatten_history_dim=True):
#   [ang_vel_t0..t9, gravity_t0..t9, jpos_t0..t9, ..., stiffness_t0..t9]
# =========================================================================

OBS_TERMS = [
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("position_commands", 2),
    ("stiffness_commands", 1),
]
HISTORY_LENGTH = 10
SINGLE_OBS_DIM = sum(dim for _, dim in OBS_TERMS)  # 45
TOTAL_OBS_DIM = SINGLE_OBS_DIM * HISTORY_LENGTH     # 450

# Control parameters (from training config)
CONTROL_DT = 0.02      # sim.dt (0.005) * decimation (4) = 50 Hz
ACTION_SCALE = 0.5
KP = 25.0               # actuator stiffness
KD = 0.5                # actuator damping


def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion. q is (w, x, y, z)."""
    w, x, y, z = q
    # Conjugate for inverse rotation
    qw, qx, qy, qz = w, -x, -y, -z
    vx, vy, vz = v
    # Quaternion-vector rotation
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return np.array([
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ], dtype=np.float32)


class SoftPosXYTrackingRunner:
    """Deploy go2_soft_pos_xy_tracking policy in MuJoCo."""

    def __init__(self, policy_path: str, device: str = "cpu"):
        # Load JIT policy
        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()
        self.device = device

        # Per-term observation history buffers (None = not yet initialized)
        self.obs_history = {name: None for name, _ in OBS_TERMS}

        # Action buffer
        self.last_action = np.zeros(12, dtype=np.float32)

        # Commands (set externally)
        self.position_cmd = np.zeros(2, dtype=np.float32)   # body-frame [vx, vy]
        self.stiffness_cmd = np.float32(20.0)                # kp value

        # State from simulator
        self.joint_pos_mujoco = np.zeros(12, dtype=np.float32)
        self.joint_vel_mujoco = np.zeros(12, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0, 0, -1], dtype=np.float32)
        self.imu_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.state_received = False

        # DDS communication
        ChannelFactoryInitialize(1, "lo")
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_callback, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.crc = CRC()

    def _state_callback(self, msg: LowState_):
        """Receive robot state from simulator (MuJoCo/SDK order)."""
        self.joint_pos_mujoco = np.array(
            [msg.motor_state[i].q for i in range(12)], dtype=np.float32
        )
        self.joint_vel_mujoco = np.array(
            [msg.motor_state[i].dq for i in range(12)], dtype=np.float32
        )
        self.imu_quat = np.array(msg.imu_state.quaternion, dtype=np.float32)
        self.base_ang_vel = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        self.projected_gravity = quat_rotate_inverse(
            self.imu_quat, np.array([0, 0, -1], dtype=np.float32)
        )
        self.state_received = True

    def _update_history(self, name: str, value: torch.Tensor, dim: int) -> torch.Tensor:
        """Update per-term history buffer and return flattened history."""
        if self.obs_history[name] is None:
            # Initialize: repeat current observation for all history frames
            self.obs_history[name] = value.repeat(HISTORY_LENGTH)
        else:
            # Shift: drop oldest frame, append newest
            self.obs_history[name] = torch.cat([
                self.obs_history[name][dim:],
                value,
            ])
        return self.obs_history[name]

    def build_observation(self) -> torch.Tensor:
        """Build full observation vector with per-term history."""
        # Convert MuJoCo-order joints to Isaac Lab order
        joint_pos_isaac = self.joint_pos_mujoco[MUJOCO_TO_ISAAC]
        joint_vel_isaac = self.joint_vel_mujoco[MUJOCO_TO_ISAAC]

        # Compute per-term current values
        term_values = {
            "base_ang_vel": torch.tensor(
                self.base_ang_vel, dtype=torch.float32, device=self.device
            ),
            "projected_gravity": torch.tensor(
                self.projected_gravity, dtype=torch.float32, device=self.device
            ),
            "joint_pos": torch.tensor(
                joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC,
                dtype=torch.float32, device=self.device,
            ),
            "joint_vel": torch.tensor(
                joint_vel_isaac, dtype=torch.float32, device=self.device
            ),
            "actions": torch.tensor(
                self.last_action, dtype=torch.float32, device=self.device
            ),
            "position_commands": torch.tensor(
                self.position_cmd, dtype=torch.float32, device=self.device
            ),
            "stiffness_commands": torch.tensor(
                [self.stiffness_cmd], dtype=torch.float32, device=self.device
            ),
        }

        # Build per-term flattened histories and concatenate
        obs_parts = []
        for name, dim in OBS_TERMS:
            history = self._update_history(name, term_values[name], dim)
            obs_parts.append(history)

        return torch.cat(obs_parts, dim=-1)

    def send_command(self, target_isaac: np.ndarray):
        """Send motor command. target_isaac is in Isaac Lab joint order."""
        # Convert to MuJoCo order
        target_mujoco = target_isaac[ISAAC_TO_MUJOCO]

        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        for i in range(12):
            cmd.motor_cmd[i].q = float(target_mujoco[i])
            cmd.motor_cmd[i].kp = float(KP)
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = float(KD)
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    def wait_for_state(self, timeout: float = 10.0) -> bool:
        start = time.time()
        while not self.state_received:
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        return True

    def go_to_default_pose(self, duration_s: float = 3.0):
        """Smoothly interpolate from current pose to default standing pose."""
        # Current pose in Isaac Lab order
        start_pos = self.joint_pos_mujoco[MUJOCO_TO_ISAAC].copy()
        n_steps = int(duration_s / CONTROL_DT)

        print("[INFO] Going to initial pose...")
        for i in range(n_steps):
            t = (i + 1) / n_steps
            target = start_pos + t * (DEFAULT_JOINT_POS_ISAAC - start_pos)
            self.send_command(target)
            time.sleep(CONTROL_DT)

        # Hold default pose briefly
        print("[INFO] Holding default pose...")
        for _ in range(int(2.0 / CONTROL_DT)):
            self.send_command(DEFAULT_JOINT_POS_ISAAC)
            time.sleep(CONTROL_DT)

    def run(self, duration: float = 60.0):
        """Main control loop."""
        print("[INFO] Waiting for simulator...")
        if not self.wait_for_state():
            print("[ERROR] Timeout waiting for robot state. Is unitree_mujoco running?")
            return

        print("[INFO] Connected to simulator!")
        self.go_to_default_pose()

        print("=" * 60)
        print(f"[INFO] Starting RL control")
        print(f"[INFO] Control: {1.0/CONTROL_DT:.0f} Hz, action_scale={ACTION_SCALE}")
        print(f"[INFO] Position cmd (body vx, vy): {self.position_cmd}")
        print(f"[INFO] Stiffness cmd (kp): {self.stiffness_cmd}")
        print(f"[INFO] Obs dim: {SINGLE_OBS_DIM} x {HISTORY_LENGTH} = {TOTAL_OBS_DIM}")
        print(f"[INFO] Duration: {duration:.0f}s")
        print("=" * 60)
        print("[INFO] Press Ctrl+C to stop\n")

        step = 0
        try:
            while step * CONTROL_DT < duration:
                t_start = time.perf_counter()

                # Build observation and run policy
                obs = self.build_observation()
                with torch.inference_mode():
                    action = self.policy(obs.unsqueeze(0)).squeeze().cpu().numpy()

                # Store action in Isaac Lab order for next observation
                self.last_action = action.astype(np.float32)

                # Compute target joint positions (Isaac Lab order)
                target_isaac = action * ACTION_SCALE + DEFAULT_JOINT_POS_ISAAC

                # Send to simulator
                self.send_command(target_isaac)

                step += 1
                if step % 50 == 0:
                    elapsed = step * CONTROL_DT
                    print(
                        f"[Step {step:5d}] t={elapsed:5.1f}s  "
                        f"gravity=[{self.projected_gravity[0]:+.2f}, "
                        f"{self.projected_gravity[1]:+.2f}, "
                        f"{self.projected_gravity[2]:+.2f}]  "
                        f"cmd=[{self.position_cmd[0]:+.2f}, "
                        f"{self.position_cmd[1]:+.2f}]  "
                        f"kp={self.stiffness_cmd:.1f}"
                    )

                # Maintain control frequency
                elapsed = time.perf_counter() - t_start
                if elapsed < CONTROL_DT:
                    time.sleep(CONTROL_DT - elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        print(f"[INFO] Finished: {step} steps, {step * CONTROL_DT:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy go2_soft_pos_xy_tracking policy in MuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Walk forward with medium stiffness
  python -m deploy.mujoco.run_soft_pos_xy_tracking --vx 0.5 --kp 20.0

  # Walk sideways with low stiffness (more compliant)
  python -m deploy.mujoco.run_soft_pos_xy_tracking --vx 0.0 --vy 0.3 --kp 10.0

  # Stand still with high stiffness (stiff)
  python -m deploy.mujoco.run_soft_pos_xy_tracking --vx 0.0 --vy 0.0 --kp 30.0

  # Use a different checkpoint
  python -m deploy.mujoco.run_soft_pos_xy_tracking \\
      --policy logs/rsl_rl/unitree_go2_walk/2026-03-01_18-46-12/exported/policy.pt
        """,
    )

    parser.add_argument(
        "--policy", type=str, default=DEFAULT_POLICY,
        help="Path to exported policy.pt",
    )
    parser.add_argument(
        "--vx", type=float, default=0.5,
        help="Body-frame forward velocity command (m/s). Clipped to [-1.5, 1.5].",
    )
    parser.add_argument(
        "--vy", type=float, default=0.0,
        help="Body-frame lateral velocity command (m/s). Clipped to [-1.5, 1.5].",
    )
    parser.add_argument(
        "--kp", type=float, default=20.0,
        help="Stiffness command (compliance kp). Training range: [10, 30].",
    )
    parser.add_argument(
        "--duration", type=float, default=60.0,
        help="Run duration in seconds.",
    )

    args = parser.parse_args()

    # Clip velocity commands to training range
    vx = np.clip(args.vx, -1.5, 1.5)
    vy = np.clip(args.vy, -1.5, 1.5)
    kp = np.clip(args.kp, 10.0, 30.0)

    runner = SoftPosXYTrackingRunner(policy_path=args.policy)
    runner.position_cmd = np.array([vx, vy], dtype=np.float32)
    runner.stiffness_cmd = np.float32(kp)
    runner.run(duration=args.duration)


if __name__ == "__main__":
    main()
