#!/usr/bin/env python3
"""
Run go2_vanilla_walk policy with unitree_mujoco.
Self-contained - no external rl_controller dependency.

Usage:
    1. Start unitree_mujoco:
       cd ~/sber_ws/RL/unitree_mujoco/simulate_python
       python unitree_mujoco.py

    2. Run this script:
       cd ~/Workspace/Projects/quadruped-locomotion/quadruped-locomotion-rl
       python -m deploy.mujoco.run_vanilla_walk
"""

import time
import numpy as np
import torch
import yaml
from pathlib import Path

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

# Policy path - your trained policy
POLICY_DIR = Path("/home/ekaterina-mozhegova/Workspace/Projects/quadruped-locomotion/quadruped-locomotion-rl/logs/rsl_rl/unitree_go2_walk/2026-01-30_11-37-42/exported")


def quat_rotate_inverse(q, v):
    """Rotate vector by inverse of quaternion (w,x,y,z format)."""
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


class SimpleRLController:
    """Simple RL controller without external dependencies."""

    def __init__(self, policy_path: str, config: dict, device="cpu"):
        self.device = device
        self.config = config

        # Load JIT policy
        print(f"[INFO] Loading policy: {policy_path}")
        self.policy = torch.jit.load(policy_path, map_location=device)
        self.policy.eval()

        # Joint order from config
        self.joint_order = config["joint_order"]
        self.num_joints = len(self.joint_order)

        # Default joint positions
        self.default_joint_pos = torch.tensor([
            config["init_state"]["default_joint_angles"][name]
            for name in self.joint_order
        ], dtype=torch.float32, device=device)

        # Observation config
        obs_order_raw = config["observation"]["order"]
        self.obs_dims = config["observation"]["dims"]
        self.num_obs_hist = config["observation"].get("num_obs_hist", 1)

        # Handle per-term history (locomotion_tasks format)
        if isinstance(self.num_obs_hist, dict):
            self.obs_hist_len = max(self.num_obs_hist.values())
            # Deduplicate observation order (locomotion_tasks repeats it)
            seen = set()
            self.obs_order = []
            for name in obs_order_raw:
                if name not in seen:
                    seen.add(name)
                    self.obs_order.append(name)
        else:
            self.obs_hist_len = self.num_obs_hist
            self.obs_order = obs_order_raw

        # Single frame observation dimension
        self.single_obs_dim = sum(self.obs_dims[name] for name in self.obs_order)
        self.total_obs_dim = self.single_obs_dim * self.obs_hist_len

        print(f"[INFO] Obs order: {self.obs_order}")
        print(f"[INFO] Obs dim: {self.single_obs_dim} x {self.obs_hist_len} = {self.total_obs_dim}")

        # Action config
        self.action_scale = config["control"]["action_scale"]

        # Command scales - deduplicate command names
        cmd_names_raw = config["commands"]["names"]
        seen_cmds = set()
        self.cmd_names = []
        for name in cmd_names_raw:
            if name not in seen_cmds:
                seen_cmds.add(name)
                self.cmd_names.append(name)

        self.cmd_scales = {
            name: config["commands"]["scales"].get(name, 1.0)
            for name in self.cmd_names
        }

        print(f"[INFO] Commands: {self.cmd_names}")

        # State buffers - separate history buffer for each observation term
        # Isaac Lab flattens per-term history: [term1_t0..t9, term2_t0..t9, ...]
        self.obs_history = {name: None for name in self.obs_order}
        self.last_action = torch.zeros(self.num_joints, dtype=torch.float32, device=device)
        self.commands = {name: 0.0 for name in self.cmd_names}

    def set_commands(self, **kwargs):
        for name, value in kwargs.items():
            if name in self.commands:
                self.commands[name] = value

    def get_action(self, obs_dict):
        """Get action from policy."""
        # Build per-term observations and update per-term history buffers
        # Isaac Lab with flatten_history_dim=True orders as: [term1_t0..t9, term2_t0..t9, ...]
        obs_parts = []
        for name in self.obs_order:
            if name == "dof_pos":
                # Relative to default
                val = obs_dict["dof_pos"] - self.default_joint_pos
            elif name == "actions":
                val = self.last_action
            elif name == "commands":
                # Build command tensor with scales (use deduplicated cmd_names)
                val = torch.tensor([
                    self.commands[n] * self.cmd_scales[n]
                    for n in self.cmd_names
                ], dtype=torch.float32, device=self.device)
            else:
                val = obs_dict[name]

            if isinstance(val, np.ndarray):
                val = torch.tensor(val, dtype=torch.float32, device=self.device)

            # Update per-term history buffer
            term_dim = self.obs_dims[name]
            if self.obs_history[name] is None:
                # Initialize with current observation repeated
                self.obs_history[name] = val.repeat(self.obs_hist_len)
            else:
                # Shift history: drop oldest, append newest
                self.obs_history[name] = torch.cat([
                    self.obs_history[name][term_dim:],
                    val
                ])

            # Add this term's flattened history to observation
            obs_parts.append(self.obs_history[name])

        # Concatenate all term histories
        full_obs = torch.cat(obs_parts, dim=-1)

        # Get policy output
        with torch.no_grad():
            action = self.policy(full_obs.unsqueeze(0)).squeeze()

        # Store for next step
        self.last_action = action.clone()

        # Scale and add to default
        target_pos = action * self.action_scale + self.default_joint_pos

        return target_pos.cpu().numpy()


class MujocoRunner:
    """Run policy in MuJoCo simulator."""

    def __init__(self, controller: SimpleRLController, config: dict):
        self.controller = controller
        self.config = config

        # State buffers
        self.joint_pos = np.zeros(12, dtype=np.float32)
        self.joint_vel = np.zeros(12, dtype=np.float32)
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.array([0, 0, -1], dtype=np.float32)
        self.imu_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        self.state_received = False

        # URDF order (MuJoCo/SDK)
        self.urdf_order = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ]

        # Mapping: urdf_idx -> policy_idx
        self.urdf_to_policy = [
            controller.joint_order.index(name) for name in self.urdf_order
        ]
        # Mapping: policy_idx -> urdf_idx
        self.policy_to_urdf = [
            self.urdf_order.index(name) for name in controller.joint_order
        ]

        # DDS setup
        ChannelFactoryInitialize(1, "lo")
        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self._state_callback, 10)
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.crc = CRC()

    def _state_callback(self, msg):
        # Get state in URDF order
        urdf_pos = np.array([msg.motor_state[i].q for i in range(12)], dtype=np.float32)
        urdf_vel = np.array([msg.motor_state[i].dq for i in range(12)], dtype=np.float32)

        # Convert to policy order
        self.joint_pos = urdf_pos[self.policy_to_urdf]
        self.joint_vel = urdf_vel[self.policy_to_urdf]

        # IMU
        self.imu_quat = np.array(msg.imu_state.quaternion, dtype=np.float32)
        self.base_ang_vel = np.array(msg.imu_state.gyroscope, dtype=np.float32)
        self.projected_gravity = quat_rotate_inverse(self.imu_quat, np.array([0, 0, -1]))

        self.state_received = True

    def wait_for_state(self, timeout=10.0):
        start = time.time()
        while not self.state_received:
            if time.time() - start > timeout:
                return False
            time.sleep(0.01)
        return True

    def send_command(self, target_policy, kp=20.0, kd=0.5):
        """Send command to robot. target_policy is in policy joint order."""
        cmd = unitree_go_msg_dds__LowCmd_()
        cmd.head[0] = 0xFE
        cmd.head[1] = 0xEF
        cmd.level_flag = 0xFF
        cmd.gpio = 0

        for urdf_idx in range(12):
            policy_idx = self.urdf_to_policy[urdf_idx]
            cmd.motor_cmd[urdf_idx].mode = 0x01
            cmd.motor_cmd[urdf_idx].q = float(target_policy[policy_idx])
            cmd.motor_cmd[urdf_idx].kp = float(kp)
            cmd.motor_cmd[urdf_idx].dq = 0.0
            cmd.motor_cmd[urdf_idx].kd = float(kd)
            cmd.motor_cmd[urdf_idx].tau = 0.0

        cmd.crc = self.crc.Crc(cmd)
        self.pub.Write(cmd)

    def run(self):
        print("[INFO] Waiting for simulation...")
        if not self.wait_for_state():
            print("[ERROR] Timeout!")
            return

        print("[INFO] Connected!")

        # Control parameters from config
        stiffness = self.config["control"].get("stiffness", 25.0)
        damping = self.config["control"].get("damping", 0.5)
        # Handle dict or scalar stiffness/damping
        if isinstance(stiffness, dict):
            kp = list(stiffness.values())[0]
        else:
            kp = stiffness
        if isinstance(damping, dict):
            kd = list(damping.values())[0]
        else:
            kd = damping
        control_dt = self.config["sim"]["dt"] * self.config["control"]["decimation"]
        print(f"[INFO] Using kp={kp}, kd={kd}, control_dt={control_dt:.4f}s")

        # Default pose
        default_pos = np.array([
            self.config["init_state"]["default_joint_angles"][name]
            for name in self.controller.joint_order
        ], dtype=np.float32)

        # Go to initial pose
        print("[INFO] Going to initial pose...")
        start_pos = self.joint_pos.copy()
        for t in np.linspace(0, 1, 150):
            target = start_pos + t * (default_pos - start_pos)
            self.send_command(target, kp=30.0, kd=1.0)
            time.sleep(0.02)

        print("[INFO] At initial pose. Waiting 2s...")
        for _ in range(100):
            self.send_command(default_pos, kp=30.0, kd=1.0)
            time.sleep(0.02)

        print("[INFO] Starting RL control!")
        step = 0

        try:
            while True:
                t_start = time.perf_counter()

                # Build observation dict
                obs_dict = {
                    "base_ang_vel": torch.tensor(self.base_ang_vel, dtype=torch.float32),
                    "projected_gravity": torch.tensor(self.projected_gravity, dtype=torch.float32),
                    "dof_pos": torch.tensor(self.joint_pos, dtype=torch.float32),
                    "dof_vel": torch.tensor(self.joint_vel, dtype=torch.float32),
                }

                # Get action
                target_pos = self.controller.get_action(obs_dict)

                # Send command
                self.send_command(target_pos, kp=kp, kd=kd)

                # Debug
                if step % 50 == 0:
                    print(f"[Step {step}] gravity: {self.projected_gravity}, target: {target_pos[:4]}...")

                step += 1

                # Timing
                elapsed = time.perf_counter() - t_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")


def main():
    # Load config
    config_path = POLICY_DIR / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create controller - try body_latest.jit first, then policy.pt
    policy_path = POLICY_DIR / "body_latest.jit"
    if not policy_path.exists():
        policy_path = POLICY_DIR / "policy.pt"
    controller = SimpleRLController(str(policy_path), config)

    # Set velocity command (include height_offset if it exists)
    controller.set_commands(
        lin_vel_x=0.5,
        lin_vel_y=0.0,
        ang_vel_z=0.0,
        height_offset=0.0,  # Only used if policy expects it
    )

    # Run
    runner = MujocoRunner(controller, config)
    runner.run()


if __name__ == "__main__":
    main()
