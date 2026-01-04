#!/usr/bin/env python3
"""
MuJoCo deployment script for trained Isaac Lab policies.

Usage:
    1. Start the MuJoCo simulator:
       cd ~/Workspace/Projects/unitree_robotics/unitree_mujoco/simulate_python
       python unitree_mujoco.py

    2. Run this controller:
       cd ~/Workspace/Projects/quadruped-locomotion-rl
       python -m deploy.mujoco.run_policy --config deploy/configs/go2_flat.yaml

    Or with command line overrides:
       python -m deploy.mujoco.run_policy --config deploy/configs/go2_flat.yaml --vx 1.0 --wz 0.5
"""

import argparse
import time
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.common.policy_loader import load_policy
from deploy.common.joint_mapping import ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC

# Import unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class PolicyController:
    """Controller that runs trained policy and sends commands to robot."""

    def __init__(self, config: dict, device: str = "cpu"):
        self.config = config
        self.device = device

        # Load policy
        checkpoint_path = REPO_ROOT / config["checkpoint"]
        self.policy = load_policy(str(checkpoint_path), device)

        # Get observation dimensions
        self.obs_dim = self.policy.input_dim
        self.base_obs_dim = config.get("base_obs_dim", 45)
        self.height_scan_dim = max(0, self.obs_dim - self.base_obs_dim)

        print(f"[INFO] Policy expects {self.obs_dim} observation dims")
        if self.height_scan_dim > 0:
            print(f"[INFO] Adding {self.height_scan_dim} dims for height scan (zeros)")

        # Control parameters
        self.action_scale = config.get("action_scale", 0.5)
        self.kp = config.get("kp", 25.0)
        self.kd = config.get("kd", 0.5)
        self.control_dt = config.get("control_dt", 0.02)

        # Default joint positions
        self.default_joint_pos = np.array(
            config.get("default_joint_pos", [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]),
            dtype=np.float32
        )

        # State variables
        self.last_action = np.zeros(12, dtype=np.float32)
        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.height_scan = np.zeros(self.height_scan_dim, dtype=np.float32)

        # Velocity commands
        default_vel = config.get("default_velocity", {})
        self.velocity_commands = np.array([
            default_vel.get("vx", 0.5),
            default_vel.get("vy", 0.0),
            default_vel.get("wz", 0.0),
        ], dtype=np.float32)

    def update_state_from_lowstate(self, msg: LowState_):
        """Update internal state from LowState message."""
        # Get joint positions and velocities (MuJoCo order)
        mujoco_pos = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
        mujoco_vel = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)

        # Convert to Isaac Lab order
        self.joint_pos = mujoco_pos[MUJOCO_TO_ISAAC]
        self.joint_vel = mujoco_vel[MUJOCO_TO_ISAAC]

        # Get IMU data
        self.imu_quat = np.array([
            msg.imu_state.quaternion[0],
            msg.imu_state.quaternion[1],
            msg.imu_state.quaternion[2],
            msg.imu_state.quaternion[3],
        ], dtype=np.float32)

        self.base_ang_vel = np.array([
            msg.imu_state.gyroscope[0],
            msg.imu_state.gyroscope[1],
            msg.imu_state.gyroscope[2],
        ], dtype=np.float32)

        # Compute projected gravity from quaternion
        self.projected_gravity = self._quat_rotate_inverse(
            self.imu_quat, np.array([0.0, 0.0, -1.0])
        )

    def _quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by the inverse of quaternion q (w, x, y, z format)."""
        w, x, y, z = q
        qw, qx, qy, qz = w, -x, -y, -z
        vx, vy, vz = v
        tx = 2.0 * (qy * vz - qz * vy)
        ty = 2.0 * (qz * vx - qx * vz)
        tz = 2.0 * (qx * vy - qy * vx)
        return np.array([
            vx + qw * tx + qy * tz - qz * ty,
            vy + qw * ty + qz * tx - qx * tz,
            vz + qw * tz + qx * ty - qy * tx,
        ], dtype=np.float32)

    def get_observation(self) -> torch.Tensor:
        """Construct observation vector matching Isaac Lab format."""
        joint_pos_rel = self.joint_pos - self.default_joint_pos

        obs_parts = [
            self.base_ang_vel,
            self.projected_gravity,
            self.velocity_commands,
            joint_pos_rel,
            self.joint_vel,
            self.last_action,
        ]

        if self.height_scan_dim > 0:
            obs_parts.append(self.height_scan)

        obs = np.concatenate(obs_parts)
        return torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

    def get_action(self) -> np.ndarray:
        """Run policy inference and return joint position targets."""
        obs = self.get_observation()
        action = self.policy(obs)
        action = action.squeeze(0).cpu().numpy()

        self.last_action = action.copy()
        target_pos = action * self.action_scale + self.default_joint_pos

        return target_pos

    def set_velocity_command(self, vx: float, vy: float, wz: float):
        """Set velocity command for the robot."""
        self.velocity_commands = np.array([vx, vy, wz], dtype=np.float32)


class RobotInterface:
    """Interface for communicating with the robot via unitree_sdk2."""

    def __init__(self, policy_controller: PolicyController, config: dict):
        self.policy = policy_controller
        self.config = config
        self.low_state = None
        self.state_received = False
        self.crc = CRC()

    def lowstate_handler(self, msg: LowState_):
        """Callback for LowState messages."""
        self.low_state = msg
        self.state_received = True
        self.policy.update_state_from_lowstate(msg)

    def run(self):
        """Main control loop."""
        # Initialize DDS
        domain_id = self.config.get("domain_id", 1)
        interface = self.config.get("interface", "lo")
        print(f"[INFO] Initializing DDS: domain_id={domain_id}, interface={interface}")
        ChannelFactoryInitialize(domain_id, interface)

        # Create publisher and subscriber
        pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        pub.Init()

        sub = ChannelSubscriber("rt/lowstate", LowState_)
        sub.Init(self.lowstate_handler, 10)

        # Initialize command message
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

        # Wait for robot state
        print("[INFO] Waiting for robot state...")
        while not self.state_received:
            time.sleep(0.01)
        print("[INFO] Robot state received.")

        input("Press Enter to start the policy...")

        # Warmup phase
        print("[INFO] Warming up...")
        warmup_time = 2.0
        warmup_start = time.time()
        kp = self.policy.kp
        kd = self.policy.kd
        control_dt = self.policy.control_dt

        while time.time() - warmup_start < warmup_time:
            step_start = time.perf_counter()

            target_pos_isaac = self.policy.get_action()
            target_pos_mujoco = target_pos_isaac[ISAAC_TO_MUJOCO]

            alpha = min(1.0, (time.time() - warmup_start) / warmup_time)
            current_kp = alpha * kp
            current_kd = alpha * kd

            for i in range(12):
                cmd.motor_cmd[i].q = float(target_pos_mujoco[i])
                cmd.motor_cmd[i].kp = current_kp
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = current_kd
                cmd.motor_cmd[i].tau = 0.0

            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)

            elapsed = time.perf_counter() - step_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

        # Main control loop
        print("[INFO] Running policy... (Ctrl+C to stop)")
        try:
            while True:
                step_start = time.perf_counter()

                target_pos_isaac = self.policy.get_action()
                target_pos_mujoco = target_pos_isaac[ISAAC_TO_MUJOCO]

                for i in range(12):
                    cmd.motor_cmd[i].q = float(target_pos_mujoco[i])
                    cmd.motor_cmd[i].kp = kp
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd
                    cmd.motor_cmd[i].tau = 0.0

                cmd.crc = self.crc.Crc(cmd)
                pub.Write(cmd)

                elapsed = time.perf_counter() - step_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping controller...")
            for i in range(12):
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0
            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)
            print("[INFO] Done.")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run trained policy in MuJoCo")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--vx", type=float, default=None, help="Forward velocity override")
    parser.add_argument("--vy", type=float, default=None, help="Lateral velocity override")
    parser.add_argument("--wz", type=float, default=None, help="Angular velocity override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    if args.vx is not None:
        config.setdefault("default_velocity", {})["vx"] = args.vx
    if args.vy is not None:
        config.setdefault("default_velocity", {})["vy"] = args.vy
    if args.wz is not None:
        config.setdefault("default_velocity", {})["wz"] = args.wz

    # Create controller and run
    policy_ctrl = PolicyController(config, args.device)

    if args.vx is not None or args.vy is not None or args.wz is not None:
        vx = args.vx if args.vx is not None else config.get("default_velocity", {}).get("vx", 0.5)
        vy = args.vy if args.vy is not None else config.get("default_velocity", {}).get("vy", 0.0)
        wz = args.wz if args.wz is not None else config.get("default_velocity", {}).get("wz", 0.0)
        policy_ctrl.set_velocity_command(vx, vy, wz)

    robot = RobotInterface(policy_ctrl, config)
    robot.run()


if __name__ == "__main__":
    main()
