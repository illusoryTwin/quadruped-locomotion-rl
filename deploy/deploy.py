"""
Generic config-driven policy deployer for Unitree Go2.

Loads a deploy config YAML that describes the observation layout,
then builds observations dynamically for any trained policy.

Usage:
    # Compliant stance (stand with soft compliance)
    python deploy/deploy.py \
        --policy logs/rsl_rl/unitree_go2_walk_soft/2026-03-05_11-35-27/exported/policy.pt \
        --config deploy/configs/compliant_stance.yaml

    # Soft position tracking (walk forward at 0.5 m/s)
    python deploy/deploy.py \
        --policy logs/rsl_rl/unitree_go2_walk/2026-03-05_12-47-15/exported/policy.pt \
        --config deploy/configs/soft_pos_xy_tracking.yaml \
        --cmd position_commands=0.5,0.0

    # Orientation tracking (rotate at 0.5 rad/s)
    python deploy/deploy.py \
        --policy logs/rsl_rl/unitree_go2_walk/XXXX/exported/policy.pt \
        --config deploy/configs/orientation_tracking.yaml \
        --cmd orientation_commands=0.0,0.0,0.5
"""

import sys
import os
import time
import argparse
from collections import deque

import numpy as np
import torch
import yaml

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
    """Generic config-driven policy deployer for Unitree Go2.

    Reads a YAML config that specifies observation term order, dimensions,
    and data sources. Builds observations dynamically for any task.
    """

    # Joint ordering mappings (shared across all Go2 tasks)
    # For _sdk_to_isaac: result[i] = sdk_array[ISAAC_TO_SDK[i]]
    ISAAC_TO_SDK = [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    # For _isaac_to_sdk: result[i] = isaac_array[SDK_TO_ISAAC[i]]
    SDK_TO_ISAAC = [1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10]

    DEFAULT_POS_ISAAC = np.array([
        0.1, -0.1, 0.1, -0.1,       # FL_hip, FR_hip, RL_hip, RR_hip
        0.8, 0.8, 1.0, 1.0,         # FL_thigh, FR_thigh, RL_thigh, RR_thigh
        -1.5, -1.5, -1.5, -1.5,     # FL_calf, FR_calf, RL_calf, RR_calf
    ], dtype=np.float32)

    # Supported observation sources
    SENSOR_SOURCES = {"imu_gyroscope", "imu_gravity", "joint_pos_rel", "joint_vel"}
    INTERNAL_SOURCES = {"last_action"}
    COMMAND_SOURCE = "command"

    def __init__(
        self,
        policy_path: str,
        config: dict,
        config_name: str = "",
        interface: str = "lo",
        domain_id: int = 1,
        cmd_overrides: dict = None,
    ):
        self.config = config
        self.config_name = config_name
        self.obs_terms = config["observation_terms"]
        self.history_length = config["history_length"]
        self.action_scale = config["action_scale"]
        self.control_dt = config["control_dt"]

        kp = config["kp"]
        kd = config["kd"]
        self.kp = np.full(12, kp, dtype=np.float32)
        self.kd = np.full(12, kd, dtype=np.float32)

        # Build command values (defaults + overrides)
        self.commands = {}
        for term in self.obs_terms:
            if term["source"] == self.COMMAND_SOURCE:
                name = term["name"]
                default = np.array(term["default"], dtype=np.float32)
                if cmd_overrides and name in cmd_overrides:
                    default = np.array(cmd_overrides[name], dtype=np.float32)
                self.commands[name] = default

        # Validate observation config
        total_per_frame = sum(t["dim"] for t in self.obs_terms)
        total_obs = total_per_frame * self.history_length
        print(f"[INFO] Observation: {total_per_frame} dims/frame x {self.history_length} history = {total_obs} total")

        # Initialize DDS
        print(f"[INFO] Initializing DDS: interface={interface}, domain_id={domain_id}")
        ChannelFactoryInitialize(domain_id, interface)

        # Load policy
        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        self.crc = CRC()

        # State buffers
        self.state: LowState_ = None
        self.state_received = False
        self.last_action = np.zeros(12, dtype=np.float32)
        self.term_histories = None

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

        # Smoothly move to default standing pose before running the policy
        self._go_to_default_pose()

        # Initialize history buffers with first real observation (now at default pose)
        self._init_history_from_state()

    def _state_callback(self, msg: LowState_):
        self.state = msg
        self.state_received = True

    def _go_to_default_pose(self, duration: float = 3.0, hold: float = 2.0):
        """Smoothly interpolate from current pose to default standing pose.

        This is critical for MuJoCo deployment: the simulator starts with all
        joints at 0 (straight legs), which is far from the training default.
        Without this step, the policy receives out-of-distribution observations
        and produces extreme actions.
        """
        # Read current joint positions (SDK order) → convert to Isaac order
        sdk_pos = np.array(
            [self.state.motor_state[i].q for i in range(12)], dtype=np.float32
        )
        start_pos_isaac = sdk_pos[self.ISAAC_TO_SDK]

        n_steps = int(duration / self.control_dt)
        print(f"[INFO] Moving to default pose ({duration:.1f}s)...")

        for i in range(n_steps):
            t = (i + 1) / n_steps
            target_isaac = start_pos_isaac + t * (self.DEFAULT_POS_ISAAC - start_pos_isaac)
            target_sdk = target_isaac[self.SDK_TO_ISAAC]

            cmd = unitree_go_msg_dds__LowCmd_()
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF
            cmd.level_flag = 0xFF
            cmd.gpio = 0

            for j in range(20):
                cmd.motor_cmd[j].mode = 0x01
                cmd.motor_cmd[j].q = 0.0
                cmd.motor_cmd[j].kp = 0.0
                cmd.motor_cmd[j].dq = 0.0
                cmd.motor_cmd[j].kd = 0.0
                cmd.motor_cmd[j].tau = 0.0

            for j in range(12):
                cmd.motor_cmd[j].q = float(target_sdk[j])
                cmd.motor_cmd[j].kp = float(self.kp[j])
                cmd.motor_cmd[j].dq = 0.0
                cmd.motor_cmd[j].kd = float(self.kd[j])
                cmd.motor_cmd[j].tau = 0.0

            cmd.crc = self.crc.Crc(cmd)
            self.cmd_pub.Write(cmd)
            time.sleep(self.control_dt)

        # Hold default pose
        n_hold = int(hold / self.control_dt)
        print(f"[INFO] Holding default pose ({hold:.1f}s)...")
        target_sdk = self.DEFAULT_POS_ISAAC[self.SDK_TO_ISAAC]

        for _ in range(n_hold):
            cmd = unitree_go_msg_dds__LowCmd_()
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF
            cmd.level_flag = 0xFF
            cmd.gpio = 0

            for j in range(20):
                cmd.motor_cmd[j].mode = 0x01
                cmd.motor_cmd[j].q = 0.0
                cmd.motor_cmd[j].kp = 0.0
                cmd.motor_cmd[j].dq = 0.0
                cmd.motor_cmd[j].kd = 0.0
                cmd.motor_cmd[j].tau = 0.0

            for j in range(12):
                cmd.motor_cmd[j].q = float(target_sdk[j])
                cmd.motor_cmd[j].kp = float(self.kp[j])
                cmd.motor_cmd[j].dq = 0.0
                cmd.motor_cmd[j].kd = float(self.kd[j])
                cmd.motor_cmd[j].tau = 0.0

            cmd.crc = self.crc.Crc(cmd)
            self.cmd_pub.Write(cmd)
            time.sleep(self.control_dt)

        print("[INFO] Default pose reached!")

    def _read_term_value(self, term: dict) -> np.ndarray:
        """Read a single observation term's current value."""
        source = term["source"]
        name = term["name"]

        if source == "joint_pos_rel":
            sdk = np.array([self.state.motor_state[i].q for i in range(12)], dtype=np.float32)
            return sdk[self.ISAAC_TO_SDK] - self.DEFAULT_POS_ISAAC

        elif source == "joint_vel":
            sdk = np.array([self.state.motor_state[i].dq for i in range(12)], dtype=np.float32)
            return sdk[self.ISAAC_TO_SDK]

        elif source == "imu_gyroscope":
            g = self.state.imu_state.gyroscope
            return np.array([g[0], g[1], g[2]], dtype=np.float32)

        elif source == "imu_gravity":
            q = self.state.imu_state.quaternion  # [w, x, y, z]
            w, x, y, z = q[0], q[1], q[2], q[3]
            gx = 2.0 * (-z * x + w * y)
            gy = -2.0 * (z * y + w * x)
            gz = 1.0 - 2.0 * (w * w + z * z)
            return np.array([gx, gy, gz], dtype=np.float32)

        elif source == "last_action":
            return self.last_action.copy()

        elif source == self.COMMAND_SOURCE:
            return self.commands[name].copy()

        else:
            raise ValueError(f"Unknown observation source: {source}")

    def _init_history_from_state(self):
        """Initialize history buffers with first real observation.

        Matches Isaac Lab's CircularBuffer behavior: all slots filled
        with the first observation on first push.
        """
        self.term_histories = {}
        for term in self.obs_terms:
            value = self._read_term_value(term)
            buf = deque(maxlen=self.history_length)
            for _ in range(self.history_length):
                buf.append(value.copy())
            self.term_histories[term["name"]] = buf
        print("[INFO] History buffers initialized from first robot state")

    def build_observation(self) -> np.ndarray:
        """Build observation with per-term history (oldest to newest)."""
        if self.state is None:
            return None

        obs_parts = []
        for term in self.obs_terms:
            name = term["name"]
            value = self._read_term_value(term)
            self.term_histories[name].append(value)
            term_hist = np.concatenate(list(self.term_histories[name]))
            obs_parts.append(term_hist)

        return np.concatenate(obs_parts).astype(np.float32)

    def step(self):
        """Execute one control step: observe -> infer -> command."""
        obs = self.build_observation()
        if obs is None:
            return

        with torch.inference_mode():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            action = self.policy(obs_tensor).numpy().squeeze()

        self.last_action = action.astype(np.float32)

        # Action -> target joint positions (Isaac Lab order -> SDK order)
        target_pos_isaac = action * self.action_scale + self.DEFAULT_POS_ISAAC
        target_pos_sdk = target_pos_isaac[self.SDK_TO_ISAAC]

        # Build and send LowCmd
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

    def run(self, duration: float = 60.0):
        """Run the control loop."""
        print("=" * 60)
        print(f"[INFO] Deploying policy")
        print(f"[INFO] Config: {self.config_name}")
        print(f"[INFO] Control frequency: {1.0/self.control_dt:.1f} Hz")
        print(f"[INFO] Duration: {duration:.1f} s")
        for name, val in self.commands.items():
            print(f"[INFO] Command '{name}': {val.tolist()}")
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


def parse_cmd_overrides(cmd_args: list) -> dict:
    """Parse --cmd arguments like 'position_commands=0.5,0.0' into a dict."""
    overrides = {}
    if not cmd_args:
        return overrides
    for arg in cmd_args:
        if "=" not in arg:
            raise ValueError(f"Invalid --cmd format: '{arg}'. Expected 'name=v1,v2,...'")
        name, values_str = arg.split("=", 1)
        values = [float(v) for v in values_str.split(",")]
        overrides[name] = values
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Generic config-driven policy deployer for Go2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compliant stance
  python deploy/deploy.py \\
      --policy logs/.../exported/policy.pt \\
      --config deploy/configs/compliant_stance.yaml

  # Soft position tracking (walk forward)
  python deploy/deploy.py \\
      --policy logs/.../exported/policy.pt \\
      --config deploy/configs/soft_pos_xy_tracking.yaml \\
      --cmd position_commands=0.5,0.0

  # Override stiffness
  python deploy/deploy.py \\
      --policy logs/.../exported/policy.pt \\
      --config deploy/configs/compliant_stance.yaml \\
      --cmd stiffness_commands=35.0
        """
    )

    parser.add_argument("--policy", type=str, required=True, help="Path to exported policy.pt")
    parser.add_argument("--config", type=str, required=True, help="Path to deploy config YAML")
    parser.add_argument("--cmd", type=str, action="append", default=None,
                        help="Override command values: name=v1,v2,... (repeatable)")
    parser.add_argument("--interface", type=str, default="lo", help="Network interface ('lo' for sim)")
    parser.add_argument("--domain", type=int, default=1, help="DDS domain ID (1 for sim, 0 for real)")
    parser.add_argument("--duration", type=float, default=60.0, help="Run duration in seconds")

    args = parser.parse_args()

    if not os.path.exists(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    cmd_overrides = parse_cmd_overrides(args.cmd)

    deployer = Go2PolicyDeployer(
        policy_path=args.policy,
        config=config,
        config_name=os.path.basename(args.config),
        interface=args.interface,
        domain_id=args.domain,
        cmd_overrides=cmd_overrides,
    )

    deployer.run(duration=args.duration)


if __name__ == "__main__":
    main()
