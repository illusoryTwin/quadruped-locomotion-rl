import time
import numpy as np
from typing import TYPE_CHECKING

from utils.joint_mapper import URDFPolicyMapper
from runners.dds_interface import DDSInterface

if TYPE_CHECKING:
    from core.policy_controller import PolicyController
    from core.config import Config


class DDSRunner:
    """Runs the RL policy control loop via DDS communication."""

    def __init__(self, controller: "PolicyController", policy_config: "Config"):
        self.controller = controller
        self.policy_config = policy_config

        # Initialize joint mapper with policy's joint order
        self.joint_mapper = URDFPolicyMapper(policy_config.joint_order)

        # Initialize DDS interface with joint mapper
        self.dds_interface = DDSInterface(self.joint_mapper)

    def send_command(self, target_pos, kp, kd):
        self.dds_interface.send_command(target_pos, kp, kd)

    def get_observation(self):
        """Get current observation from robot state."""
        state = self.dds_interface.get_state()
        if state is None:
            return None
        return state

    def _go_to_initial_pose(self):
        """Smoothly transition to default pose before starting RL control."""
        print("[INFO] Going to initial pose...")

        # Get current joint positions
        obs = self.get_observation()
        start_pos = obs["dof_pos"].numpy()
        default_pos = self.policy_config.default_joint_pos.numpy()

        # Interpolate to default pose over 3 seconds (150 steps at 50Hz)
        for t in np.linspace(0, 1, 150):
            target = start_pos + t * (default_pos - start_pos)
            self.send_command(target, kp=30.0, kd=1.0)
            time.sleep(0.02)

        # Hold at default pose for 2 seconds
        print("[INFO] At initial pose. Stabilizing...")
        for _ in range(100):
            self.send_command(default_pos, kp=30.0, kd=1.0)
            time.sleep(0.02)

    def run(self):
        step = 0
        print("[INFO] Starting control loop. Press Ctrl+C to stop.")

        # Wait for first state
        print("[INFO] Waiting for robot state...")
        while self.dds_interface.latest_state is None:
            time.sleep(0.01)
        print("[INFO] Robot state received.")

        # Go to initial pose first
        self._go_to_initial_pose()

        print("[INFO] Starting RL control!")
        kp = self.policy_config.stiffness
        kd = self.policy_config.damping
        print(f"[INFO] Using kp={kp}, kd={kd}, control_dt={self.policy_config.control_dt:.4f}s")

        try:
            while True:
                start_time = time.perf_counter()

                obs_dict = self.get_observation()
                if obs_dict is None:
                    continue

                target_pos = self.controller.get_action(obs_dict)

                self.send_command(target_pos, kp, kd)

                # Debug output every 50 steps
                if step % 50 == 0:
                    gravity = obs_dict["projected_gravity"].numpy()
                    print(f"[Step {step}] gravity: {gravity}, target: {target_pos[:4].numpy()}...")

                step += 1

                elapsed = time.perf_counter() - start_time
                if elapsed < self.policy_config.control_dt:
                    time.sleep(self.policy_config.control_dt - elapsed)

        except KeyboardInterrupt:
            print(f"\n[INFO] Control stopped by user after {step} steps.")
