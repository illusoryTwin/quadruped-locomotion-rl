#!/bin/bash
# =============================================================================
# Policy runner entrypoint: launches MuJoCo simulator + policy deployer
# =============================================================================
#
# Usage:
#   docker compose run quadruped-policy <task_name>
#
# Examples:
#   docker compose run quadruped-policy go2_pos_xy_tracking
#   docker compose run quadruped-policy go2_compliant_stance
#   DURATION=60 docker compose run quadruped-policy go2_soft_pos_xy_tracking
#   CMD_ARGS="position_commands=0.5,0.0" docker compose run quadruped-policy go2_pos_xy_tracking
#
# Env var overrides:
#   POLICY_PATH    - override auto-resolved policy path
#   DURATION       - run duration in seconds (default: 120)
#   CMD_ARGS       - extra --cmd arguments
#   DDS_INTERFACE  - network interface (default: lo)
#   DDS_DOMAIN     - DDS domain ID (default: 1)

set -e

TASK_MAPPING="/workspace/quadruped-locomotion-rl/deploy/task_mapping.yaml"
LOGS_ROOT="/workspace/quadruped-locomotion-rl/logs/rsl_rl"
DEPLOY_POLICIES="/workspace/quadruped-locomotion-rl/deploy/policies"
CONFIGS_DIR="/workspace/quadruped-locomotion-rl/deploy/configs"

DDS_INTERFACE="${DDS_INTERFACE:-lo}"
DDS_DOMAIN="${DDS_DOMAIN:-1}"
DURATION="${DURATION:-120}"
CMD_ARGS="${CMD_ARGS:-}"

# ---------------------------------------------------------------------------
# Parse task name
# ---------------------------------------------------------------------------
TASK_NAME="${1:-}"
if [ -z "$TASK_NAME" ]; then
    echo "[entrypoint] ERROR: No task name provided."
    echo ""
    echo "Usage: docker compose run quadruped-policy <task_name>"
    echo ""
    echo "Available tasks:"
    grep -E '^[a-z]' "$TASK_MAPPING" | sed 's/:$//' | sed 's/^/  /'
    exit 1
fi

# ---------------------------------------------------------------------------
# Look up task in mapping
# ---------------------------------------------------------------------------
if ! grep -q "^${TASK_NAME}:" "$TASK_MAPPING"; then
    echo "[entrypoint] ERROR: Unknown task '$TASK_NAME'"
    echo ""
    echo "Available tasks:"
    grep -E '^[a-z]' "$TASK_MAPPING" | sed 's/:$//' | sed 's/^/  /'
    exit 1
fi

EXPERIMENT_NAME=$(grep -A2 "^${TASK_NAME}:" "$TASK_MAPPING" | grep "experiment_name:" | awk '{print $2}')
DEPLOY_CONFIG=$(grep -A2 "^${TASK_NAME}:" "$TASK_MAPPING" | grep "deploy_config:" | awk '{print $2}')

CONFIG_PATH="${CONFIGS_DIR}/${DEPLOY_CONFIG}.yaml"

# ---------------------------------------------------------------------------
# Resolve policy path (auto-find latest run, or use override)
# ---------------------------------------------------------------------------
if [ -n "$POLICY_PATH" ]; then
    echo "[entrypoint] Using override POLICY_PATH=$POLICY_PATH"
else
    # 1) Check deploy/policies/ first (curated, committed policies)
    DEPLOY_MATCH=$(ls -1d "$DEPLOY_POLICIES/${EXPERIMENT_NAME}"* 2>/dev/null | sort -r | head -1)
    if [ -n "$DEPLOY_MATCH" ] && [ -f "$DEPLOY_MATCH/policy.pt" ]; then
        POLICY_PATH="$DEPLOY_MATCH/policy.pt"
        echo "[entrypoint] Found deploy policy: $POLICY_PATH"
    else
        # 2) Fall back to logs/ (local training results)
        EXPERIMENT_DIR="${LOGS_ROOT}/${EXPERIMENT_NAME}"
        if [ ! -d "$EXPERIMENT_DIR" ]; then
            echo "[entrypoint] ERROR: No policy found in deploy/policies/ or logs/"
            echo "[entrypoint] Either copy a policy to deploy/policies/ or train: python scripts/train.py --task=$TASK_NAME"
            exit 1
        fi

        # Find latest run directory (by name, timestamps sort lexicographically)
        LATEST_RUN=$(ls -1 "$EXPERIMENT_DIR" | sort -r | head -1)
        POLICY_PATH="${EXPERIMENT_DIR}/${LATEST_RUN}/exported/policy.pt"

        if [ ! -f "$POLICY_PATH" ]; then
            echo "[entrypoint] ERROR: No exported policy at $POLICY_PATH"
            echo "[entrypoint] Training may still be in progress (no checkpoint exported yet)."
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Launch MuJoCo simulator
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting MuJoCo simulator..."
cd /workspace/unitree_mujoco/simulate_python
python unitree_mujoco.py &
MUJOCO_PID=$!

sleep 3

# ---------------------------------------------------------------------------
# Launch policy deployer
# ---------------------------------------------------------------------------
echo "[entrypoint] Starting policy deployer..."
echo "[entrypoint]   task:     $TASK_NAME"
echo "[entrypoint]   policy:   $POLICY_PATH"
echo "[entrypoint]   config:   $DEPLOY_CONFIG"
echo "[entrypoint]   duration: ${DURATION}s"
if [ -n "$CMD_ARGS" ]; then
    echo "[entrypoint]   commands: $CMD_ARGS"
fi

# Build --cmd flags from CMD_ARGS
CMD_FLAGS=""
for cmd_arg in $CMD_ARGS; do
    CMD_FLAGS="$CMD_FLAGS --cmd $cmd_arg"
done

cd /workspace/quadruped-locomotion-rl
python deploy/deploy.py \
    --policy "$POLICY_PATH" \
    --config "$CONFIG_PATH" \
    --interface "$DDS_INTERFACE" \
    --domain "$DDS_DOMAIN" \
    --duration "$DURATION" \
    $CMD_FLAGS &
POLICY_PID=$!

echo "[entrypoint] Both processes running (MuJoCo PID=$MUJOCO_PID, Policy PID=$POLICY_PID)"
echo "[entrypoint] Press Ctrl+C to stop."

# Wait for either process to exit, then shut down both
trap "kill $MUJOCO_PID $POLICY_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait -n $MUJOCO_PID $POLICY_PID
echo "[entrypoint] One process exited, shutting down..."
kill $MUJOCO_PID $POLICY_PID 2>/dev/null
wait
