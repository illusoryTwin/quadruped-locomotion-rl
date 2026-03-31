# =============================================================================
# Deploy/Test container: MuJoCo + PyTorch + Unitree SDK for policy testing
# =============================================================================
#
# Build context: quadruped-locomotion-rl/ repo root
#
# Lightweight container for:
#   - Running trained policies in MuJoCo simulation
#   - Testing deployment code before real robot
#   - Policy inference with unitree_sdk2py DDS communication
#
# Usage:
#   docker compose build
#   docker compose run quadruped-deploy

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

SHELL ["/bin/bash", "-c"]

LABEL description="Quadruped locomotion deployment and MuJoCo testing"

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Paths
ARG QUADRUPED_RL_PATH=/workspace/quadruped-locomotion-rl
ARG UNITREE_SDK_PATH=/workspace/unitree_sdk2py
ARG UNITREE_MUJOCO_PATH=/workspace/unitree_mujoco

ENV QUADRUPED_RL_PATH=${QUADRUPED_RL_PATH}
ENV UNITREE_SDK_PATH=${UNITREE_SDK_PATH}
ENV UNITREE_MUJOCO_PATH=${UNITREE_MUJOCO_PATH}

USER root

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    wget && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Make python3.10 the default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# PyTorch (CUDA 12.8)
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch --index-url https://download.pytorch.org/whl/cu128

# -----------------------------------------------------------------------------
# MuJoCo
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install mujoco numpy pyyaml opencv-python pygame

# -----------------------------------------------------------------------------
# Unitree SDK2 Python (DDS communication)
# -----------------------------------------------------------------------------
ARG UNITREE_SDK2_PYTHON_REPO=https://github.com/unitreerobotics/unitree_sdk2_python.git
RUN git clone --depth 1 ${UNITREE_SDK2_PYTHON_REPO} ${UNITREE_SDK_PATH}

RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${UNITREE_SDK_PATH} && pip install -e .

# -----------------------------------------------------------------------------
# Unitree MuJoCo simulator
# -----------------------------------------------------------------------------
ARG UNITREE_MUJOCO_REPO=https://github.com/unitreerobotics/unitree_mujoco.git
RUN git clone --depth 1 ${UNITREE_MUJOCO_REPO} ${UNITREE_MUJOCO_PATH}

# Install MuJoCo simulator Python dependencies if any
RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${UNITREE_MUJOCO_PATH}/simulate_python && \
    if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# -----------------------------------------------------------------------------
# Quadruped RL repo (deploy/ is inside)
# -----------------------------------------------------------------------------
COPY . ${QUADRUPED_RL_PATH}

# Install quadruped-locomotion-rl package if pyproject.toml exists
RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${QUADRUPED_RL_PATH} && \
    if [ -f pyproject.toml ]; then pip install -e .; fi

# -----------------------------------------------------------------------------
# Entrypoint scripts
# -----------------------------------------------------------------------------
COPY deploy/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY deploy/docker/entrypoint-policy.sh /usr/local/bin/entrypoint-policy.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/entrypoint-policy.sh

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
RUN echo "export QUADRUPED_RL_PATH=${QUADRUPED_RL_PATH}" >> /root/.bashrc && \
    echo "export UNITREE_MUJOCO_PATH=${UNITREE_MUJOCO_PATH}" >> /root/.bashrc && \
    echo "export TZ=$(date +%Z)" >> /root/.bashrc

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
