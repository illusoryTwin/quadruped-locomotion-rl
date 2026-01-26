#!/usr/bin/env python3
"""
Test script to send mock commands to MuJoCo simulator.

This bypasses the policy and sends simple joint position commands
to verify the communication and robot response.

Usage:
    1. Start the MuJoCo simulator:
       python -m deploy.mujoco.launch_sim

    2. Run this test script:
       python -m deploy.mujoco.test_mock_commands
"""

import time
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


# Default standing pose (MuJoCo order: FR, FL, RR, RL by leg)
# Each leg: hip, thigh, calf
DEFAULT_STAND_POS = np.array([
    -0.1, 0.8, -1.5,   # FR: hip, thigh, calf
     0.1, 0.8, -1.5,   # FL: hip, thigh, calf
    -0.1, 1.0, -1.5,   # RR: hip, thigh, calf
     0.1, 1.0, -1.5,   # RL: hip, thigh, calf
], dtype=np.float32)


class MockController:
    def __init__(self):
        self.state_received = False
        self.low_state = None
        self.crc = CRC()

    def lowstate_handler(self, msg: LowState_):
        self.low_state = msg
        self.state_received = True

    def run(self):
        # Initialize DDS
        print("[INFO] Initializing DDS: domain_id=1, interface=lo")
        ChannelFactoryInitialize(1, "lo")

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

        # Print current joint positions
        current_pos = [self.low_state.motor_state[i].q for i in range(12)]
        print(f"[INFO] Current joint positions: {[f'{p:.2f}' for p in current_pos]}")

        input("\nPress Enter to move robot to standing pose...")

        # Control parameters
        kp = 25.0
        kd = 0.5
        control_dt = 0.02  # 50 Hz

        # Slowly move to standing pose
        print("[INFO] Moving to standing pose (2 second ramp)...")
        ramp_time = 2.0
        start_time = time.time()

        # Get initial positions
        init_pos = np.array([self.low_state.motor_state[i].q for i in range(12)], dtype=np.float32)

        while time.time() - start_time < ramp_time:
            step_start = time.perf_counter()

            # Interpolate between initial and target
            alpha = min(1.0, (time.time() - start_time) / ramp_time)
            target_pos = init_pos + alpha * (DEFAULT_STAND_POS - init_pos)

            # Also ramp gains
            current_kp = alpha * kp
            current_kd = alpha * kd

            for i in range(12):
                cmd.motor_cmd[i].q = float(target_pos[i])
                cmd.motor_cmd[i].kp = current_kp
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = current_kd
                cmd.motor_cmd[i].tau = 0.0

            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)

            elapsed = time.perf_counter() - step_start
            if elapsed < control_dt:
                time.sleep(control_dt - elapsed)

        print("[INFO] Standing pose reached!")
        print(f"[INFO] Target positions: {[f'{p:.2f}' for p in DEFAULT_STAND_POS]}")

        # Hold standing pose
        print("[INFO] Holding pose... (Ctrl+C to stop)")
        step_count = 0
        try:
            while True:
                step_start = time.perf_counter()

                for i in range(12):
                    cmd.motor_cmd[i].q = float(DEFAULT_STAND_POS[i])
                    cmd.motor_cmd[i].kp = kp
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = kd
                    cmd.motor_cmd[i].tau = 0.0

                cmd.crc = self.crc.Crc(cmd)
                pub.Write(cmd)

                # Print status every second
                if step_count % 50 == 0:
                    current_pos = [self.low_state.motor_state[i].q for i in range(12)]
                    error = np.array(current_pos) - DEFAULT_STAND_POS
                    print(f"[INFO] Step {step_count}: pos_error max={np.abs(error).max():.4f}")

                step_count += 1

                elapsed = time.perf_counter() - step_start
                if elapsed < control_dt:
                    time.sleep(control_dt - elapsed)

        except KeyboardInterrupt:
            print("\n[INFO] Stopping...")
            # Zero out commands
            for i in range(12):
                cmd.motor_cmd[i].q = 0.0
                cmd.motor_cmd[i].kp = 0.0
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 0.0
                cmd.motor_cmd[i].tau = 0.0
            cmd.crc = self.crc.Crc(cmd)
            pub.Write(cmd)
            print("[INFO] Done.")


def main():
    print("=" * 60)
    print("Mock Command Test for MuJoCo Simulator")
    print("=" * 60)
    print("\nThis script sends simple standing pose commands to test")
    print("communication with the MuJoCo simulator.\n")

    controller = MockController()
    controller.run()


if __name__ == "__main__":
    main()
