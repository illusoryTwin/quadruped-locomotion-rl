"""
Deploy compliant stance policy to Go2 robot using unitree_sdk2_python.

This script connects to unitree_mujoco simulator (or real robot) via DDS,
loads the trained compliant stance policy, and runs the control loop.

Usage:
    # Simulation (with unitree_mujoco running)
    python deploy/deploy_compliant_stance.py \
        --policy logs/rsl_rl/unitree_go2_walk_soft/2026-03-05_11-35-27/exported/policy.pt \
        --interface lo --domain 1

    # Real robot
    python deploy/deploy_compliant_stance.py \
        --policy logs/rsl_rl/unitree_go2_walk_soft/2026-03-05_11-35-27/exported/policy.pt \
        --interface eth0 --domain 0
"""

import sys
import os
import time
import argparse
from collections import deque
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unitree_sdk2py.core.channel import (
    ChannelPublisher,
    ChannelSubscriber,
    ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_
from unitree_sdk2py.utils.crc import CRC


class Go2CompliantStanceDeployer:
    """
    Deploy compliant stance RL policy to Unitree Go2.

    Observation space (per step = 46 dims, x10 history = 460 total):
        joint_pos_rel (12), joint_vel (12), last_action (12),
        projected_gravity (3), position_commands (3),
        stiffness_commands (1), base_ang_vel (3)

    Joint ordering:
        Isaac Lab: FL_hip, FR_hip, RL_hip, RR_hip,
                   FL_thigh, FR_thigh, RL_thigh, RR_thigh,
                   FL_calf, FR_calf, RL_calf, RR_calf
        SDK:       FR_hip, FR_thigh, FR_calf,
                   FL_hip, FL_thigh, FL_calf,
                   RR_hip, RR_thigh, RR_calf,
                   RL_hip, RL_thigh, RL_calf
    """

    # SDK index -> Isaac Lab index
    SDK_TO_ISAAC = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]
    # Isaac Lab index -> SDK index
    ISAAC_TO_SDK = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]

    # Default joint positions in Isaac Lab order
    DEFAULT_POS_ISAAC = np.array([
        0.1, -0.1, 0.1, -0.1,       # FL_hip, FR_hip, RL_hip, RR_hip
        0.8, 0.8, 1.0, 1.0,         # FL_thigh, FR_thigh, RL_thigh, RR_thigh
        -1.5, -1.5, -1.5, -1.5,     # FL_calf, FR_calf, RL_calf, RR_calf
    ], dtype=np.float32)

    # Default joint positions in SDK order
    DEFAULT_POS_SDK = np.array([
        -0.1, 0.8, -1.5,   # FR
        0.1, 0.8, -1.5,    # FL
        -0.1, 1.0, -1.5,   # RR
        0.1, 1.0, -1.5,    # RL
    ], dtype=np.float32)

    HISTORY_LENGTH = 10

    # Per-term observation dimensions (must match training config order)
    OBS_TERM_DIMS = {
        "joint_pos": 12,
        "joint_vel": 12,
        "actions": 12,
        "projected_gravity": 3,
        "position_commands": 3,
        "stiffness_commands": 1,
        "base_ang_vel": 3,
    }

    def __init__(
        self,
        policy_path: str,
        interface: str = "lo",
        domain_id: int = 1,
        target_height: float = 0.3,
        stiffness_kp: float = 40.0,
    ):
        self.target_height = target_height
        self.stiffness_kp = stiffness_kp

        # Initialize DDS
        print(f"[INFO] Initializing DDS: interface={interface}, domain_id={domain_id}")
        ChannelFactoryInitialize(domain_id, interface)

        # Load policy
        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.crc = CRC()

        # Control parameters (from training config)
        self.control_dt = 0.02      # decimation(4) * sim_dt(0.005) = 0.02s (50Hz)
        self.action_scale = 0.5
        self.kp = np.full(12, 25.0, dtype=np.float32)
        self.kd = np.full(12, 0.5, dtype=np.float32)

        # State buffers
        self.state: LowState_ = None
        self.state_received = False
        self.last_action = np.zeros(12, dtype=np.float32)  # in Isaac Lab order

        # Per-term history buffers - initialized AFTER first state received
        # (Isaac Lab fills all history slots with the first real observation)
        self.term_histories = None

        # Commands
        self.position_cmd = np.array([0.0, 0.0, self.target_height], dtype=np.float32)
        self.stiffness_cmd = np.array([self.stiffness_kp], dtype=np.float32)

        # DDS channels
        self.state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.state_sub.Init(self._state_callback, 10)

        self.cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.cmd_pub.Init()

        # Wait for state
        print("[INFO] Waiting for robot state...")
        timeout = 10.0
        start = time.time()
        while not self.state_received:
            if time.time() - start > timeout:
                raise RuntimeError("Timeout waiting for robot state. Is unitree_mujoco running?")
            time.sleep(0.01)
        print("[INFO] Robot state received!")

        # Initialize history buffers with the FIRST real observation
        # (matches Isaac Lab's CircularBuffer behavior)
        self._init_history_from_state()

    def _state_callback(self, msg: LowState_):
        self.state = msg
        self.state_received = True

    def _read_current_terms(self) -> dict:
        """Read current observation term values from robot state."""
        imu = self.state.imu_state
        motors = self.state.motor_state

        # Joint positions (SDK -> Isaac Lab -> relative)
        joint_pos_sdk = np.array([motors[i].q for i in range(12)], dtype=np.float32)
        joint_pos_isaac = self._sdk_to_isaac(joint_pos_sdk)
        joint_pos_rel = joint_pos_isaac - self.DEFAULT_POS_ISAAC

        # Joint velocities (SDK -> Isaac Lab)
        joint_vel_sdk = np.array([motors[i].dq for i in range(12)], dtype=np.float32)
        joint_vel_isaac = self._sdk_to_isaac(joint_vel_sdk)

        # Last action (already in Isaac Lab order)
        actions = self.last_action.copy()

        # Projected gravity
        quat = np.array(imu.quaternion, dtype=np.float32)  # [w, x, y, z]
        projected_gravity = self._get_projected_gravity(quat)

        # Angular velocity
        ang_vel = np.array([
            imu.gyroscope[0],
            imu.gyroscope[1],
            imu.gyroscope[2],
        ], dtype=np.float32)

        return {
            "joint_pos": joint_pos_rel,
            "joint_vel": joint_vel_isaac,
            "actions": actions,
            "projected_gravity": projected_gravity,
            "position_commands": self.position_cmd.copy(),
            "stiffness_commands": self.stiffness_cmd.copy(),
            "base_ang_vel": ang_vel,
        }

    def _init_history_from_state(self):
        """Initialize history buffers by repeating the first real observation.

        This matches Isaac Lab's CircularBuffer behavior which fills all
        history slots with the first observation on the first push.
        """
        first_obs = self._read_current_terms()
        self.term_histories = {}
        for name, dim in self.OBS_TERM_DIMS.items():
            buf = deque(maxlen=self.HISTORY_LENGTH)
            for _ in range(self.HISTORY_LENGTH):
                buf.append(first_obs[name].copy())
            self.term_histories[name] = buf
        print("[INFO] History buffers initialized from first robot state")

    def _sdk_to_isaac(self, sdk_array: np.ndarray) -> np.ndarray:
        """Reorder 12-dim array from SDK order to Isaac Lab order.

        result[isaac_idx] = sdk_array[ISAAC_TO_SDK[isaac_idx]]
        i.e., for each Isaac Lab position, pick from the corresponding SDK index.
        """
        return sdk_array[self.ISAAC_TO_SDK]

    def _isaac_to_sdk(self, isaac_array: np.ndarray) -> np.ndarray:
        """Reorder 12-dim array from Isaac Lab order to SDK order.

        result[sdk_idx] = isaac_array[SDK_TO_ISAAC[sdk_idx]]
        i.e., for each SDK position, pick from the corresponding Isaac Lab index.
        """
        return isaac_array[self.SDK_TO_ISAAC]

    def _get_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        """Compute gravity vector in body frame from quaternion [w, x, y, z]."""
        w, x, y, z = quat
        gx = 2.0 * (-z * x + w * y)
        gy = -2.0 * (z * y + w * x)
        gz = 1.0 - 2.0 * (w * w + z * z)
        return np.array([gx, gy, gz], dtype=np.float32)

    def build_observation(self) -> np.ndarray:
        """
        Build 460-dim observation with per-term history.

        Isaac Lab applies history_length=10 to EACH term individually,
        so the layout is:
            [joint_pos_t-9, ..., joint_pos_t,        # 12*10 = 120
             joint_vel_t-9, ..., joint_vel_t,         # 12*10 = 120
             actions_t-9, ..., actions_t,              # 12*10 = 120
             proj_grav_t-9, ..., proj_grav_t,          #  3*10 = 30
             pos_cmd_t-9, ..., pos_cmd_t,              #  3*10 = 30
             stiff_cmd_t-9, ..., stiff_cmd_t,          #  1*10 = 10
             ang_vel_t-9, ..., ang_vel_t]              #  3*10 = 30
                                                  Total = 460
        """
        if self.state is None:
            return None

        # Read current observation terms
        current_terms = self._read_current_terms()

        # Append current values to each term's history buffer
        for name, value in current_terms.items():
            self.term_histories[name].append(value)

        # Flatten: for each term, concatenate its history (oldest to newest)
        obs_parts = []
        for name in self.OBS_TERM_DIMS:
            term_hist = np.concatenate(list(self.term_histories[name]))
            obs_parts.append(term_hist)

        return np.concatenate(obs_parts).astype(np.float32)

    def step(self) -> None:
        """Execute one control step: observe -> infer -> command."""
        obs = self.build_observation()
        if obs is None:
            return

        # Policy inference
        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self.policy(obs_tensor).numpy().squeeze()

        # Store action for next observation (in Isaac Lab order)
        self.last_action = action.astype(np.float32)

        # Convert action to target joint positions (Isaac Lab order)
        target_pos_isaac = action * self.action_scale + self.DEFAULT_POS_ISAAC

        # Reorder to SDK order for sending commands
        target_pos_sdk = self._isaac_to_sdk(target_pos_isaac)

        # Build LowCmd message
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
            cmd.motor_cmd[i].q = float(target_pos_sdk[i])
            cmd.motor_cmd[i].kp = float(self.kp[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = float(self.kd[i])
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.cmd_pub.Write(cmd)

    def run(self, duration: float = 60.0) -> None:
        """Run the control loop."""
        print("=" * 60)
        print(f"[INFO] Deploying compliant stance policy")
        print(f"[INFO] Control frequency: {1.0/self.control_dt:.1f} Hz")
        print(f"[INFO] Duration: {duration:.1f} s")
        print(f"[INFO] Target height: {self.target_height:.2f} m")
        print(f"[INFO] Stiffness kp: {self.stiffness_kp:.1f}")
        total_obs = sum(d * self.HISTORY_LENGTH for d in self.OBS_TERM_DIMS.values())
        print(f"[INFO] Observation: per-term history x {self.HISTORY_LENGTH} = {total_obs} dims")
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

                elapsed = time.perf_counter() - step_start
                sleep_time = self.control_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if step_count % int(1.0 / self.control_dt) == 0:
                    elapsed_total = time.time() - start_time
                    print(f"[INFO] Running... {elapsed_total:.1f}s / {duration:.1f}s", end="\r")

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user")

        print(f"\n[INFO] Finished after {time.time() - start_time:.1f}s, {step_count} steps")


def main():
    parser = argparse.ArgumentParser(
        description="Deploy compliant stance policy to Go2 robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simulation (with unitree_mujoco running)
  python deploy/deploy_compliant_stance.py \\
      --policy logs/rsl_rl/unitree_go2_walk_soft/2026-03-05_11-35-27/exported/policy.pt

  # Real robot
  python deploy/deploy_compliant_stance.py \\
      --policy logs/rsl_rl/unitree_go2_walk_soft/2026-03-05_11-35-27/exported/policy.pt \\
      --interface eth0 --domain 0
        """
    )

    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--interface", type=str, default="lo", help="Network interface ('lo' for sim)")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain ID (1 for sim, 0 for real)")
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration in seconds")
    parser.add_argument("--height", type=float, default=0.3, help="Target standing height (m)")
    parser.add_argument("--stiffness", type=float, default=40.0, help="Stiffness kp command (30-50 range)")

    args = parser.parse_args()

    if not os.path.exists(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)

    deployer = Go2CompliantStanceDeployer(
        policy_path=args.policy,
        interface=args.interface,
        domain_id=args.domain,
        target_height=args.height,
        stiffness_kp=args.stiffness,
    )

    deployer.run(duration=args.duration)


if __name__ == "__main__":
    main()
