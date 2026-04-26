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
#   KEYBOARD       - set to 1 to enable WASD velocity control
#   POLICY_SOURCE  - "deploy" (default) or "logs" to control resolution priority

set -e

TASK_MAPPING="/workspace/quadruped-locomotion-rl/deploy/task_mapping.yaml"
LOGS_ROOT="/workspace/quadruped-locomotion-rl/logs/rsl_rl"
DEPLOY_POLICIES="/workspace/quadruped-locomotion-rl/deploy/policies"
CONFIGS_DIR="/workspace/quadruped-locomotion-rl/deploy/configs"

DDS_INTERFACE="${DDS_INTERFACE:-lo}"
DDS_DOMAIN="${DDS_DOMAIN:-1}"
DURATION="${DURATION:-120}"
CMD_ARGS="${CMD_ARGS:-}"
KEYBOARD="${KEYBOARD:-0}"
POLICY_SOURCE="${POLICY_SOURCE:-deploy}"

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
resolve_from_deploy() {
    local match
    match=$(ls -1d "$DEPLOY_POLICIES/${EXPERIMENT_NAME}"* 2>/dev/null | sort -r | head -1)
    if [ -n "$match" ] && [ -f "$match/policy.pt" ]; then
        POLICY_PATH="$match/policy.pt"
        echo "[entrypoint] Found deploy policy: $POLICY_PATH"
        return 0
    fi
    return 1
}

resolve_from_logs() {
    local experiment_dir="${LOGS_ROOT}/${EXPERIMENT_NAME}"
    if [ ! -d "$experiment_dir" ]; then
        return 1
    fi
    local latest_run
    latest_run=$(ls -1 "$experiment_dir" | sort -r | head -1)
    local candidate="${experiment_dir}/${latest_run}/exported/policy.pt"
    if [ -f "$candidate" ]; then
        POLICY_PATH="$candidate"
        echo "[entrypoint] Found logs policy: $POLICY_PATH"
        return 0
    fi
    return 1
}

if [ -n "$POLICY_PATH" ]; then
    echo "[entrypoint] Using override POLICY_PATH=$POLICY_PATH"
elif [ "$POLICY_SOURCE" = "logs" ]; then
    if ! resolve_from_logs; then
        if ! resolve_from_deploy; then
            echo "[entrypoint] ERROR: No policy found in logs/ or deploy/policies/"
            echo "[entrypoint] Train one first: python scripts/train.py --task=$TASK_NAME"
            exit 1
        fi
    fi
else
    if ! resolve_from_deploy; then
        if ! resolve_from_logs; then
            echo "[entrypoint] ERROR: No policy found in deploy/policies/ or logs/"
            echo "[entrypoint] Either copy a policy to deploy/policies/ or train: python scripts/train.py --task=$TASK_NAME"
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Launch MuJoCo simulator
# ---------------------------------------------------------------------------
# Overlay custom MuJoCo files (perturbation support, force visualization)
MUJOCO_OVERLAY="/workspace/quadruped-locomotion-rl/deploy/mujoco"
MUJOCO_TARGET="/workspace/unitree_mujoco/simulate_python"
SCENE_TARGET="/workspace/unitree_mujoco/unitree_robots/go2"
if [ -d "$MUJOCO_OVERLAY" ]; then
    echo "[entrypoint] Applying MuJoCo overlay from deploy/mujoco/..."
    cp -v "$MUJOCO_OVERLAY"/*.py "$MUJOCO_TARGET"/
    # Overlay custom scene XML if present
    if [ -f "$MUJOCO_OVERLAY/scene.xml" ]; then
        cp -v "$MUJOCO_OVERLAY/scene.xml" "$SCENE_TARGET/scene.xml"
    fi
fi

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

KB_FLAG=""
if [ "$KEYBOARD" = "1" ]; then
    KB_FLAG="--keyboard"
fi

cd /workspace/quadruped-locomotion-rl
python deploy/deploy.py \
    --policy "$POLICY_PATH" \
    --config "$CONFIG_PATH" \
    --interface "$DDS_INTERFACE" \
    --domain "$DDS_DOMAIN" \
    --duration "$DURATION" \
    $CMD_FLAGS $KB_FLAG &
POLICY_PID=$!

echo "[entrypoint] Both processes running (MuJoCo PID=$MUJOCO_PID, Policy PID=$POLICY_PID)"
echo "[entrypoint] Press Ctrl+C to stop."

# Wait for either process to exit, then shut down both
trap "kill $MUJOCO_PID $POLICY_PID 2>/dev/null; exit 0" SIGINT SIGTERM

wait -n $MUJOCO_PID $POLICY_PID
echo "[entrypoint] One process exited, shutting down..."
kill $MUJOCO_PID $POLICY_PID 2>/dev/null
wait
