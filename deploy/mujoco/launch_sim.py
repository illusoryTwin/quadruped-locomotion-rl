#!/usr/bin/env python3
"""
Helper script to launch MuJoCo simulator for Go2.

This script launches the unitree_mujoco simulator from the external repo.

Usage:
    python -m deploy.mujoco.launch_sim

Or directly:
    python deploy/mujoco/launch_sim.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Path to unitree_mujoco
UNITREE_MUJOCO_PATH = Path.home() / "Workspace/Projects/unitree_robotics/unitree_mujoco"
SIMULATE_PYTHON_PATH = UNITREE_MUJOCO_PATH / "simulate_python"


def main():
    if not SIMULATE_PYTHON_PATH.exists():
        print(f"[ERROR] unitree_mujoco not found at: {UNITREE_MUJOCO_PATH}")
        print("Please clone it:")
        print(f"  git clone https://github.com/unitreerobotics/unitree_mujoco {UNITREE_MUJOCO_PATH}")
        sys.exit(1)

    print(f"[INFO] Launching MuJoCo simulator from: {SIMULATE_PYTHON_PATH}")
    print("[INFO] Press Ctrl+C to stop")

    # Change to simulator directory and run
    os.chdir(SIMULATE_PYTHON_PATH)
    subprocess.run([sys.executable, "unitree_mujoco.py"])


if __name__ == "__main__":
    main()
