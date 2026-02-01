"""
Deploy trained policy to Go2 robot using unitree_sdk2_python.

This script connects to unitree_mujoco simulator (or real robot) via DDS,
loads your trained policy, and runs the control loop.

Usage:
    # Simulation
    python scripts/deploy_unitree_sdk2.py \
        --policy logs/rsl_rl/unitree_go2_walk/2025-12-30_10-07-43/exported/policy.pt \
        --interface lo --domain 1

    # Real robot
    python scripts/deploy_unitree_sdk2.py \
        --policy logs/rsl_rl/unitree_go2_walk/2025-12-30_10-07-43/exported/policy.pt \
        --interface eth0 --domain 0
"""

import sys
import os
import time
import argparse
import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.utils.crc import CRC


class Go2PolicyDeployer:
    """
    Deploy RL policy to Unitree Go2 robot.

    Handles communication via unitree_sdk2_python (DDS) and runs
    policy inference at specified control frequency.
    """

    # =========================================================================
    # JOINT ORDERING
    # =========================================================================
    # SDK/MuJoCo motor indices (this is the order for LowCmd/LowState):
    #   0: FR_hip,   1: FR_thigh,  2: FR_calf
    #   3: FL_hip,   4: FL_thigh,  5: FL_calf
    #   6: RR_hip,   7: RR_thigh,  8: RR_calf
    #   9: RL_hip,  10: RL_thigh, 11: RL_calf
    #
    # Isaac Lab joint order (check your training config if different):
    # Usually matches SDK order for Go2, but verify with your env.
    # =========================================================================

    SDK_JOINT_NAMES = [
        "FR_hip", "FR_thigh", "FR_calf",   # 0, 1, 2
        "FL_hip", "FL_thigh", "FL_calf",   # 3, 4, 5
        "RR_hip", "RR_thigh", "RR_calf",   # 6, 7, 8
        "RL_hip", "RL_thigh", "RL_calf",   # 9, 10, 11
    ]

    def __init__(
        self,
        policy_path: str,
        interface: str = "lo",
        domain_id: int = 1,
        use_height_scan: bool = False,
    ):
        """
        Initialize the policy deployer.

        Args:
            policy_path: Path to exported policy.pt file
            interface: Network interface ("lo" for sim, "eth0" for real)
            domain_id: DDS domain ID (1 for sim, 0 for real)
            use_height_scan: Whether policy expects height scan observation
        """
        self.use_height_scan = use_height_scan

        # Initialize DDS communication
        print(f"[INFO] Initializing DDS: interface={interface}, domain_id={domain_id}")
        ChannelFactoryInitialize(domain_id, interface)

        # Load policy
        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        # CRC calculator for command validation
        self.crc = CRC()

        # =====================================================================
        # CONTROL PARAMETERS - MATCH THESE TO YOUR TRAINING CONFIG!
        # From your flat_walk_env_cfg.py:
        # =====================================================================

        # Timing: control_dt = sim.dt * decimation = 0.005 * 4 = 0.02s (50Hz)
        self.control_dt = 0.02

        # Action scaling (from actions.joint_pos.scale)
        self.action_scale = 0.5

        # PD gains (from actuators.base_legs)
        self.kp = np.full(12, 25.0, dtype=np.float32)   # stiffness
        self.kd = np.full(12, 0.5, dtype=np.float32)    # damping

        # Default joint positions (from init_state.joint_pos)
        # Order: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
        #        RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
        self.default_joint_pos = np.array([
            -0.1, 0.8, -1.5,   # FR: R_hip=-0.1, F_thigh=0.8, calf=-1.5
            0.1, 0.8, -1.5,    # FL: L_hip=0.1, F_thigh=0.8, calf=-1.5
            -0.1, 1.0, -1.5,   # RR: R_hip=-0.1, R_thigh=1.0, calf=-1.5
            0.1, 1.0, -1.5,    # RL: L_hip=0.1, R_thigh=1.0, calf=-1.5
        ], dtype=np.float32)

        # =====================================================================
        # STATE BUFFERS
        # =====================================================================
        self.state: LowState_ = None
        self.state_received = False
        self.last_action = np.zeros(12, dtype=np.float32)

        # Velocity command [vx, vy, wz]
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # =====================================================================
        # SETUP DDS CHANNELS
        # =====================================================================

        # Subscribe to robot state
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._state_callback, 10)

        # Publisher for motor commands
        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        # Wait for first state message
        print("[INFO] Waiting for robot state...")
        timeout = 10.0
        start = time.time()
        while not self.state_received:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for robot state. Is unitree_mujoco running?")
            time.sleep(0.01)
        print("[INFO] Robot state received!")

    def _state_callback(self, msg: LowState_):
        """Callback for robot state updates."""
        self.state = msg
        self.state_received = True

    def _get_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector projected into body frame.

        Args:
            quat: Quaternion [w, x, y, z]

        Returns:
            Projected gravity vector [gx, gy, gz]
        """
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Rotate world gravity [0, 0, -1] to body frame
        gx = 2.0 * (-z * x + w * y)
        gy = -2.0 * (z * y + w * x)
        gz = 1.0 - 2.0 * (w * w + z * z)

        return np.array([gx, gy, gz], dtype=np.float32)

    def build_observation(self) -> np.ndarray:
        """
        Build observation vector matching Isaac Lab training.

        Observation space (from flat_walk_env_cfg.py):
            - base_ang_vel: 3
            - projected_gravity: 3
            - velocity_commands: 3
            - joint_pos (relative to default): 12
            - joint_vel: 12
            - actions (last action): 12
            - height_scan: 187 (optional, requires ray casting)

        Total: 45 (without height_scan) or 232 (with height_scan)
        """
        if self.state is None:
            return None

        imu = self.state.imu_state
        motors = self.state.motor_state

        # -----------------------------------------------------------------
        # base_ang_vel (3) - angular velocity from gyroscope
        # -----------------------------------------------------------------
        ang_vel = np.array([
            imu.gyroscope[0],
            imu.gyroscope[1],
            imu.gyroscope[2],
        ], dtype=np.float32)

        # -----------------------------------------------------------------
        # projected_gravity (3) - gravity in body frame
        # -----------------------------------------------------------------
        quat = np.array(imu.quaternion, dtype=np.float32)  # [w, x, y, z]
        projected_gravity = self._get_projected_gravity(quat)

        # -----------------------------------------------------------------
        # velocity_commands (3) - commanded velocity [vx, vy, wz]
        # -----------------------------------------------------------------
        velocity_commands = self.cmd.copy()

        # -----------------------------------------------------------------
        # joint_pos (12) - relative to default positions
        # -----------------------------------------------------------------
        joint_pos = np.array(
            [motors[i].q for i in range(12)], dtype=np.float32
        )
        joint_pos_rel = joint_pos - self.default_joint_pos

        # -----------------------------------------------------------------
        # joint_vel (12) - joint velocities
        # -----------------------------------------------------------------
        joint_vel = np.array(
            [motors[i].dq for i in range(12)], dtype=np.float32
        )

        # -----------------------------------------------------------------
        # actions (12) - last action taken
        # -----------------------------------------------------------------
        actions = self.last_action.copy()

        # -----------------------------------------------------------------
        # Concatenate observation
        # -----------------------------------------------------------------
        obs = np.concatenate([
            ang_vel,            # 3
            projected_gravity,  # 3
            velocity_commands,  # 3
            joint_pos_rel,      # 12
            joint_vel,          # 12
            actions,            # 12
        ])  # Total: 45

        # -----------------------------------------------------------------
        # height_scan (187) - optional
        # WARNING: Your policy was trained with height_scan!
        # Without proper ray casting, the policy may not work correctly.
        # -----------------------------------------------------------------
        if self.use_height_scan:
            # Placeholder: zeros (policy won't work well!)
            # TODO: Implement proper ray casting if needed
            height_scan = np.zeros(187, dtype=np.float32)
            obs = np.concatenate([obs, height_scan])

        return obs.astype(np.float32)

    def step(self) -> None:
        """Execute one control step: observe -> infer -> command."""
        obs = self.build_observation()
        if obs is None:
            return

        # Policy inference
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self.policy(obs_tensor).numpy().squeeze()

        # Store action for next observation
        self.last_action = action.astype(np.float32)

        # Convert action to target joint positions
        target_pos = action * self.action_scale + self.default_joint_pos

        # Build LowCmd message
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        # Initialize all motors
        for i in range(20):
            cmd.motor_cmd[i].mode = 0x01  # PMSM mode
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].kp = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 0.0
            cmd.motor_cmd[i].tau = 0.0

        # Set target positions for leg motors (0-11)
        for i in range(12):
            cmd.motor_cmd[i].q = float(target_pos[i])
            cmd.motor_cmd[i].kp = float(self.kp[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = float(self.kd[i])
            cmd.motor_cmd[i].tau = 0.0

        # Compute CRC and publish
        cmd.crc = self.crc.Crc(cmd)
        self.cmd_pub.Write(cmd)

    def run(self, duration: float = 60.0) -> None:
        """
        Run the control loop.

        Args:
            duration: How long to run in seconds
        """
        print("=" * 60)
        print(f"[INFO] Starting policy deployment")
        print(f"[INFO] Control frequency: {1.0/self.control_dt:.1f} Hz")
        print(f"[INFO] Duration: {duration:.1f} s")
        print(f"[INFO] Command: vx={self.cmd[0]:.2f}, vy={self.cmd[1]:.2f}, wz={self.cmd[2]:.2f}")
        print(f"[INFO] Height scan: {'enabled' if self.use_height_scan else 'disabled'}")
        print("=" * 60)
        print("[INFO] Press Ctrl+C to stop")
        print()

        start_time = time.time()
        step_count = 0

        try:
            while (time.time() - start_time) < duration:
                step_start = time.perf_counter()

                self.step()
                step_count += 1

                # Maintain control frequency
                elapsed = time.perf_counter() - step_start
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Print status every second
                if step_count % int(1.0 / self.control_dt) == 0:
                    elapsed_total = time.time() - start_time
                    print(f"[INFO] Running... {elapsed_total:.1f}s / {duration:.1f}s", end="\r")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        print(f"\n[INFO] Finished after {time.time() - start_time:.1f}s, {step_count} steps")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy trained policy to Go2 robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulation (with unitree_mujoco running)
  python scripts/deploy_unitree_sdk2.py --policy path/to/policy.pt --interface lo --domain 1

  # Real robot
  python scripts/deploy_unitree_sdk2.py --policy path/to/policy.pt --interface eth0 --domain 0
        """
    )

    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to exported policy.pt file"
    )
    parser.add_argument(
        "--interface",
        type=str,
        default="lo",
        help="Network interface: 'lo' for simulation, 'eth0' (or similar) for real robot"
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=1,
        help="DDS domain ID: 1 for simulation, 0 for real robot"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Run duration in seconds"
    )
    parser.add_argument(
        "--vx",
        type=float,
        default=0.5,
        help="Forward velocity command (m/s)"
    )
    parser.add_argument(
        "--vy",
        type=float,
        default=0.0,
        help="Lateral velocity command (m/s)"
    )
    parser.add_argument(
        "--wz",
        type=float,
        default=0.0,
        help="Yaw rate command (rad/s)"
    )
    parser.add_argument(
        "--height-scan",
        action="store_true",
        help="Enable height scan observation (requires policy trained with it)"
    )

    args = parser.parse_args()

    # Validate policy path
    if not os.path.exists(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)

    # Create deployer
    deployer = Go2PolicyDeployer(
        policy_path=args.policy,
        interface=args.interface,
        domain_id=args.domain,
        use_height_scan=args.height_scan,
    )

    # Set velocity command
    deployer.cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)

    # Run
    deployer.run(duration=args.duration)


if __name__ == "__main__":
    main()
