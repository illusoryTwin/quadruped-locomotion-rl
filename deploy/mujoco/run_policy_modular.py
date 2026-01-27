#!/usr/bin/env python3
"""
MuJoCo deployment script using modular RLController.

Usage:
    1. Start the MuJoCo simulator:
       cd ~/Workspace/Projects/unitree_robotics/unitree_mujoco/simulate_python
       python unitree_mujoco.py

    2. Run this controller:
       cd ~/Workspace/Projects/quadruped-locomotion/quadruped-locomotion-rl
       python -m deploy.mujoco.run_policy_modular --config deploy/configs/go2_flat_modular.yaml

    Or with command line overrides:
       python -m deploy.mujoco.run_policy_modular --config deploy/configs/go2_flat_modular.yaml --vx 1.0 --wz 0.5
"""

import argparse
import time
import sys
from pathlib import Path

import numpy as np
import yaml

# Add parent directory to path for imports
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deploy.common import RLController, ISAAC_TO_MUJOCO, MUJOCO_TO_ISAAC

# Import unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


class RobotState:
    """Container for robot state from sensors."""

    def __init__(self, num_joints: int = 12):
        self.joint_pos = np.zeros(num_joints, dtype=np.float32)
        self.joint_vel = np.zeros(num_joints, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    def update_from_lowstate(self, msg: LowState_):
        """Update state from LowState message."""
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

    @staticmethod
    def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
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


class RobotInterface:
    """Interface for communicating with the robot via unitree_sdk2."""

    def __init__(self, controller: RLController, config: dict):
        self.controller = controller
        self.config = config
        self.robot_state = RobotState()
        self.state_received = False
        self.crc = CRC()

        # Height scan placeholder (for flat terrain, use constant value)
        obs_cfg = config.get("observation", {})
        height_scan_dim = obs_cfg.get("dims", {}).get("height_scan", 187)
        self.height_scan = np.full(height_scan_dim, 0.5, dtype=np.float32)

    def lowstate_handler(self, msg: LowState_):
        """Callback for LowState messages."""
        self.robot_state.update_from_lowstate(msg)
        self.state_received = True

    def get_observation_dict(self) -> dict:
        """Build observation dictionary from robot state."""
        return {
            "base_ang_vel": self.robot_state.base_ang_vel,
            "projected_gravity": self.robot_state.projected_gravity,
            "dof_pos": self.robot_state.joint_pos,
            "dof_vel": self.robot_state.joint_vel,
            "height_scan": self.height_scan,
        }

    def run(self):
        """Main control loop."""
        # Initialize DDS
        dds_cfg = self.config.get("dds", {})
        domain_id = dds_cfg.get("domain_id", 1)
        interface = dds_cfg.get("interface", "lo")
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

        # Get control parameters
        control_cfg = self.config.get("control", {})
        control_dt = control_cfg.get("control_dt", 0.02)
        kp_gains, kd_gains = self.controller.get_gains()

        # Warmup phase
        print("[INFO] Warming up...")
        warmup_time = 2.0
        warmup_start = time.time()

        while time.time() - warmup_start < warmup_time:
            step_start = time.perf_counter()

            obs_dict = self.get_observation_dict()
            target_pos_isaac = self.controller(obs_dict)
            target_pos_mujoco = target_pos_isaac[ISAAC_TO_MUJOCO]

            # Ramp up gains during warmup
            alpha = min(1.0, (time.time() - warmup_start) / warmup_time)

            for i in range(12):
                cmd.motor_cmd[i].q = float(target_pos_mujoco[i])
                cmd.motor_cmd[i].kp = alpha * kp_gains[ISAAC_TO_MUJOCO[i]]
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = alpha * kd_gains[ISAAC_TO_MUJOCO[i]]
                cmd.motor_cmd[i].tau = 0.0

            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)

            elapsed = time.perf_counter() - step_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

        # Main control loop
        print("[INFO] Running policy... (Ctrl+C to stop)")
        step_count = 0

        try:
            while True:
                step_start = time.perf_counter()

                obs_dict = self.get_observation_dict()
                target_pos_isaac = self.controller(obs_dict)
                target_pos_mujoco = target_pos_isaac[ISAAC_TO_MUJOCO]

                # Debug output every 50 steps (~1 second)
                if step_count % 50 == 0:
                    cmds = self.controller.commander.get_cmd_unscaled()
                    print(f"[DEBUG] Step {step_count}")
                    print(f"  vel_cmd: vx={cmds[0]:.2f}, vy={cmds[1]:.2f}, wz={cmds[2]:.2f}")
                    print(f"  joint_pos: {self.robot_state.joint_pos[:4]}...")
                    print(f"  target_pos: {target_pos_isaac[:4]}...")
                step_count += 1

                for i in range(12):
                    cmd.motor_cmd[i].q = float(target_pos_mujoco[i])
                    cmd.motor_cmd[i].kp = kp_gains[ISAAC_TO_MUJOCO[i]]
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd_gains[ISAAC_TO_MUJOCO[i]]
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
    parser = argparse.ArgumentParser(description="Run trained policy in MuJoCo (modular)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--vx", type=float, default=None, help="Forward velocity override")
    parser.add_argument("--vy", type=float, default=None, help="Lateral velocity override")
    parser.add_argument("--wz", type=float, default=None, help="Angular velocity override")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint override")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Resolve checkpoint path
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    checkpoint = config["checkpoint"]
    if not Path(checkpoint).is_absolute():
        config["checkpoint"] = str(REPO_ROOT / checkpoint)

    # Create controller
    print(f"[INFO] Loading policy from: {config['checkpoint']}")
    controller = RLController(config, device=args.device)
    print(f"[INFO] Observation dim: {controller.obs_dim}")

    # Apply velocity overrides
    if args.vx is not None:
        controller.set_cmd("vx", args.vx)
    if args.vy is not None:
        controller.set_cmd("vy", args.vy)
    if args.wz is not None:
        controller.set_cmd("wz", args.wz)

    # Run
    robot = RobotInterface(controller, config)
    robot.run()


if __name__ == "__main__":
    main()
