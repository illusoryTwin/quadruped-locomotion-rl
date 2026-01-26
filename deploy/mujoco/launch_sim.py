#!/usr/bin/env python3
"""
Helper script to launch MuJoCo simulator for Go2.

This script launches the unitree_mujoco simulator from the external repo.

Usage:
    python -m deploy.mujoco.launch_sim
    python -m deploy.mujoco.launch_sim --monitor  # Also print received commands

Or directly:
    python deploy/mujoco/launch_sim.py
"""

import subprocess
import sys
import os
import argparse
import threading
import time
from pathlib import Path

# Path to unitree_mujoco
UNITREE_MUJOCO_PATH = Path.home() / "Workspace/Projects/unitree_robotics/unitree_mujoco"
SIMULATE_PYTHON_PATH = UNITREE_MUJOCO_PATH / "simulate_python"


def monitor_commands():
    """Monitor and print commands being sent to the robot."""
    from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_

    ChannelFactoryInitialize(1, "lo")

    cmd_count = 0
    last_print_time = time.time()

    def cmd_handler(msg: LowCmd_):
        nonlocal cmd_count, last_print_time
        cmd_count += 1

        # Print every 1 second
        if time.time() - last_print_time >= 1.0:
            # Get first 4 motor commands
            positions = [msg.motor_cmd[i].q for i in range(4)]
            kps = [msg.motor_cmd[i].kp for i in range(4)]
            print(f"[CMD] count={cmd_count}, pos[:4]={[f'{p:.2f}' for p in positions]}, kp={kps[0]:.1f}")
            last_print_time = time.time()

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(cmd_handler, 10)

    print("[INFO] Monitoring commands on rt/lowcmd...")
    while True:
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Launch MuJoCo simulator")
    parser.add_argument("--monitor", action="store_true", help="Monitor and print commands")
    args = parser.parse_args()

    if not SIMULATE_PYTHON_PATH.exists():
        print(f"[ERROR] unitree_mujoco not found at: {UNITREE_MUJOCO_PATH}")
        print("Please clone it:")
        print(f"  git clone https://github.com/unitreerobotics/unitree_mujoco {UNITREE_MUJOCO_PATH}")
        sys.exit(1)

    print(f"[INFO] Launching MuJoCo simulator from: {SIMULATE_PYTHON_PATH}")
    print("[INFO] Press Ctrl+C to stop")

    # Start command monitor in background if requested
    if args.monitor:
        monitor_thread = threading.Thread(target=monitor_commands, daemon=True)
        monitor_thread.start()

    # Change to simulator directory and run
    os.chdir(SIMULATE_PYTHON_PATH)
    subprocess.run([sys.executable, "unitree_mujoco.py"])


if __name__ == "__main__":
    main()
