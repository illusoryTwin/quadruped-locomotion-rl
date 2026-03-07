#!/bin/bash
# =============================================================================
# Sequential multi-task training
# =============================================================================
# Usage:
#   bash scripts/train_all.sh
#   bash scripts/train_all.sh --device cuda:1
#
# Each task runs as a separate process since Isaac Sim cannot reinitialize
# within the same process.

set -e

DEVICE="${1:---device cuda:1}"
COMMON_ARGS="--headless --num_envs 4096 $DEVICE"

TASKS=(
    "go2_compliant_stance       --max_iterations 5000"
    "go2_default_stance         --max_iterations 5000"
    "go2_walk_flat              --max_iterations 5000"
    "go2_pos_xy_tracking        --max_iterations 5000"
    "go2_soft_pos_xy_tracking   --max_iterations 5000"
)

for entry in "${TASKS[@]}"; do
    TASK=$(echo "$entry" | awk '{print $1}')
    EXTRA_ARGS=$(echo "$entry" | cut -d' ' -f2-)

    echo "============================================="
    echo " Training: $TASK"
    echo "============================================="

    python3 scripts/train.py --task="$TASK" $COMMON_ARGS $EXTRA_ARGS

    echo ""
    echo " Finished: $TASK"
    echo "============================================="
    echo ""
done

echo "All tasks completed."
