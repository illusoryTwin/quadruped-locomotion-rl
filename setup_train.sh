#!/usr/bin/env bash
# =============================================================================
# setup_train.sh — Training environment setup for compliant-quadruped-locomotion
# =============================================================================
#
# Sets up the conda environment needed to train policies with Isaac Sim + Isaac Lab.
# Deploy (MuJoCo testing) is handled separately via Docker — see deploy/docker/.
#
# Prerequisites (must be in place before running this script):
#   - NVIDIA GPU with CUDA-capable drivers (driver >= 525 recommended)
#   - ~30 GB free disk space (Isaac Sim is large)
#   - Internet access (downloads Isaac Sim, Isaac Lab, dependencies)
#   - conda / miniconda installed
#     If not: https://docs.anaconda.com/miniconda/install/#quick-command-line-install
#
# Usage:
#   bash setup_train.sh
#
# Configurable via environment variables (all have defaults):
#   ISAACLAB_PATH       Where to clone Isaac Lab  (default: ../IsaacLab)
#   UNITREE_RL_LAB_PATH Where to clone unitree_rl_lab (default: ../unitree_robotics/unitree_rl_lab)
#   CONDA_ENV_NAME      Conda env name (default: env_isaaclab)
#
# After setup, train with:
#   conda activate env_isaaclab
#   python scripts/train.py --task=go2_compliant_stance --num_envs=4096 --headless
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

ISAACLAB_PATH="${ISAACLAB_PATH:-$(dirname "$PROJECT_ROOT")/IsaacLab}"
UNITREE_RL_LAB_PATH="${UNITREE_RL_LAB_PATH:-$(dirname "$PROJECT_ROOT")/unitree_robotics/unitree_rl_lab}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-env_isaaclab}"

PYTHON_VERSION="3.11"
ISAACSIM_VERSION="5.1.0.0"
ISAACLAB_TAG="v2.3.1"
RSL_RL_VERSION="3.0.1"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
section() { echo; echo "=== $* ==="; }
info()    { echo "[setup] $*"; }
ok()      { echo "[setup] OK: $*"; }
fail()    { echo "[setup] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites check
# ---------------------------------------------------------------------------
section "Checking prerequisites"

if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi not found. An NVIDIA GPU with installed drivers is required."
fi
info "GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/         /'

if ! command -v conda &>/dev/null; then
    fail "conda not found. Install miniconda first: https://docs.anaconda.com/miniconda/install/"
fi
info "conda: $(conda --version)"

DISK_AVAIL_GB=$(df -BG "$HOME" | awk 'NR==2 {gsub("G",""); print $4}')
if [ "$DISK_AVAIL_GB" -lt 25 ]; then
    fail "Less than 25 GB free in \$HOME ($DISK_AVAIL_GB GB). Isaac Sim requires ~20 GB."
fi
info "Disk space available: ${DISK_AVAIL_GB} GB"

# ---------------------------------------------------------------------------
# 2. Conda environment
# ---------------------------------------------------------------------------
section "Conda environment: $CONDA_ENV_NAME"

# Source conda so we can activate inside this script
# shellcheck disable=SC1091
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    info "Conda env '$CONDA_ENV_NAME' already exists — skipping creation."
else
    info "Creating conda env '$CONDA_ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    ok "Conda env created."
fi

conda activate "$CONDA_ENV_NAME"
PIP="$(conda run -n "$CONDA_ENV_NAME" which pip)"
PYTHON="$(conda run -n "$CONDA_ENV_NAME" which python)"
info "Python: $($PYTHON --version)"

# ---------------------------------------------------------------------------
# 3. Isaac Sim (pip install from NVIDIA's PyPI)
# ---------------------------------------------------------------------------
section "Isaac Sim $ISAACSIM_VERSION"

if "$PIP" show isaacsim &>/dev/null; then
    INSTALLED_SIM=$("$PIP" show isaacsim | grep '^Version' | awk '{print $2}')
    info "Isaac Sim already installed (version $INSTALLED_SIM) — skipping."
else
    info "Installing Isaac Sim $ISAACSIM_VERSION from NVIDIA PyPI..."
    info "(This is a large download — ~15 GB. Go get a coffee.)"
    # NOTE: On first run this will prompt you to accept NVIDIA's EULA.
    "$PIP" install \
        "isaacsim==${ISAACSIM_VERSION}" \
        --extra-index-url https://pypi.nvidia.com \
        --extra-index-url https://download.pytorch.org/whl/cu128
    ok "Isaac Sim installed."
fi

# ---------------------------------------------------------------------------
# 4. Isaac Lab (clone + install editable)
# ---------------------------------------------------------------------------
section "Isaac Lab $ISAACLAB_TAG"

if [ -d "$ISAACLAB_PATH/.git" ]; then
    info "Isaac Lab repo already at $ISAACLAB_PATH — skipping clone."
else
    info "Cloning Isaac Lab $ISAACLAB_TAG into $ISAACLAB_PATH..."
    mkdir -p "$(dirname "$ISAACLAB_PATH")"
    git clone --branch "$ISAACLAB_TAG" --depth 1 \
        https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_PATH"
    ok "Isaac Lab cloned."
fi

info "Installing Isaac Lab packages (editable)..."
for pkg in isaaclab isaaclab_assets isaaclab_rl isaaclab_tasks; do
    PKG_PATH="$ISAACLAB_PATH/source/$pkg"
    if [ -d "$PKG_PATH" ]; then
        if "$PIP" show "$pkg" &>/dev/null; then
            info "  $pkg already installed — skipping."
        else
            info "  Installing $pkg..."
            "$PIP" install -e "$PKG_PATH" --quiet
        fi
    fi
done
ok "Isaac Lab packages installed."

# ---------------------------------------------------------------------------
# 5. RSL-RL (PPO implementation)
# ---------------------------------------------------------------------------
section "RSL-RL $RSL_RL_VERSION"

if "$PIP" show rsl-rl-lib &>/dev/null; then
    info "rsl-rl-lib already installed — skipping."
else
    "$PIP" install "rsl-rl-lib==${RSL_RL_VERSION}"
    ok "rsl-rl-lib installed."
fi

# ---------------------------------------------------------------------------
# 6. unitree_rl_lab (Unitree robot assets + IsaacLab extension)
#    Used by train_export.py to register Go2 environments.
# ---------------------------------------------------------------------------
section "unitree_rl_lab"

if [ -d "$UNITREE_RL_LAB_PATH/.git" ]; then
    info "unitree_rl_lab already at $UNITREE_RL_LAB_PATH — skipping clone."
else
    info "Cloning unitree_rl_lab into $UNITREE_RL_LAB_PATH..."
    mkdir -p "$(dirname "$UNITREE_RL_LAB_PATH")"
    git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_rl_lab.git "$UNITREE_RL_LAB_PATH"
    ok "unitree_rl_lab cloned."
fi

URL_SRC="$UNITREE_RL_LAB_PATH/source/unitree_rl_lab"
if [ -d "$URL_SRC" ]; then
    if "$PIP" show unitree-rl-lab &>/dev/null 2>&1 || "$PIP" show unitree_rl_lab &>/dev/null 2>&1; then
        info "unitree_rl_lab already installed — skipping."
    else
        info "Installing unitree_rl_lab (editable)..."
        "$PIP" install -e "$URL_SRC" --quiet
        ok "unitree_rl_lab installed."
    fi
fi

# ---------------------------------------------------------------------------
# 7. This project
# ---------------------------------------------------------------------------
section "compliant-quadruped-locomotion"

if "$PIP" show compliant-quadruped-locomotion &>/dev/null; then
    info "compliant-quadruped-locomotion already installed — skipping."
else
    info "Installing compliant-quadruped-locomotion (editable)..."
    "$PIP" install -e "$PROJECT_ROOT" --quiet
    ok "compliant-quadruped-locomotion installed."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
section "Setup complete"
echo
echo "  Activate the environment:"
echo "    conda activate $CONDA_ENV_NAME"
echo
echo "  Train:"
echo "    python scripts/train.py --task=go2_compliant_stance --num_envs=4096 --headless"
echo "    python scripts/train.py --task=go2_soft_walk        --num_envs=4096 --headless"
echo
echo "  Visualize:"
echo "    python scripts/play.py  --task=go2_compliant_stance --num_envs=4"
echo
echo "  Deploy (separate Docker environment):"
echo "    docker compose run --rm quadruped-policy go2_soft_walk"
echo
