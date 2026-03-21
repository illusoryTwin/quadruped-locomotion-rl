#!/bin/bash
# =============================================================================
# Deploy entrypoint: launches MuJoCo simulator + policy controller
# =============================================================================

set -e

echo "[entrypoint] Starting MuJoCo simulator..."
cd /workspace/unitree_mujoco/simulate_python
python unitree_mujoco.py &
MUJOCO_PID=$!

# Wait for simulator to initialize
sleep 3

echo "[entrypoint] Starting policy controller..."
cd /workspace/quadruped-locomotion-rl/deploy
python run.py &
POLICY_PID=$!

echo "[entrypoint] Both processes running (MuJoCo PID=$MUJOCO_PID, Policy PID=$POLICY_PID)"
echo "[entrypoint] Press Ctrl+C to stop."

# Wait for either process to exit, then shut down both
trap "kill $MUJOCO_PID $POLICY_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait -n $MUJOCO_PID $POLICY_PID
echo "[entrypoint] One process exited, shutting down..."
kill $MUJOCO_PID $POLICY_PID 2>/dev/null
wait
