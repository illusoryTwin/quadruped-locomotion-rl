# =============================================================================
# Training container: Isaac Sim 5.1 + IsaacLab + quadruped-locomotion-rl
# =============================================================================
#
# NOTE: This Dockerfile is OPTIONAL. It is only needed if you want to train
# inside a container. If you have Isaac Sim + IsaacLab installed locally,
# you can ignore this file entirely. The deploy profile does NOT depend on it.
#
# Build context: Workspace/Projects/ (two levels up from this file)
#
# Usage:
#   cd quadruped-locomotion-rl/docker
#   docker compose build
#   docker compose run quadruped-train

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS base

SHELL ["/bin/bash", "-c"]

LABEL description="Quadruped locomotion RL training on Isaac Sim + IsaacLab"

# Paths
ARG ISAACSIM_ROOT_PATH=/isaac-sim
ARG ISAACLAB_PATH=/workspace/isaaclab
ARG QUADRUPED_RL_PATH=/workspace/quadruped-locomotion-rl
ARG UNITREE_RL_LAB_PATH=/workspace/unitree_rl_lab

ENV ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH}
ENV ISAACLAB_PATH=${ISAACLAB_PATH}
ENV QUADRUPED_RL_PATH=${QUADRUPED_RL_PATH}
ENV UNITREE_RL_LAB_PATH=${UNITREE_RL_LAB_PATH}

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root

# -----------------------------------------------------------------------------
# System dependencies
# -----------------------------------------------------------------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libglib2.0-0 \
    ncurses-term \
    wget && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# IsaacLab
# -----------------------------------------------------------------------------
COPY IsaacLab/ ${ISAACLAB_PATH}

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh

# Symlink Isaac Sim into IsaacLab
RUN ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

# Install toml first (needed by isaaclab.sh)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

# Install apt deps declared by IsaacLab extensions
RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# Install IsaacLab and all its Python dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

# Remove problematic quadprog dependency (matches upstream workaround)
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog

# -----------------------------------------------------------------------------
# Unitree RL Lab (IsaacLab extension)
# -----------------------------------------------------------------------------
COPY unitree_robotics/unitree_rl_lab/ ${UNITREE_RL_LAB_PATH}

RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${UNITREE_RL_LAB_PATH}/source/unitree_rl_lab && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .

# -----------------------------------------------------------------------------
# Quadruped Locomotion RL project
# -----------------------------------------------------------------------------
COPY quadruped-locomotion-rl/ ${QUADRUPED_RL_PATH}

RUN --mount=type=cache,target=/root/.cache/pip \
    cd ${QUADRUPED_RL_PATH} && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install -e .

# -----------------------------------------------------------------------------
# Isaac Sim cache directories
# -----------------------------------------------------------------------------
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache && \
    mkdir -p /root/.cache/ov && \
    mkdir -p /root/.cache/pip && \
    mkdir -p /root/.cache/nvidia/GLCache && \
    mkdir -p /root/.nv/ComputeCache && \
    mkdir -p /root/.nvidia-omniverse/logs && \
    mkdir -p /root/.local/share/ov/data && \
    mkdir -p /root/Documents

# -----------------------------------------------------------------------------
# Convenience aliases
# -----------------------------------------------------------------------------
RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "export QUADRUPED_RL_PATH=${QUADRUPED_RL_PATH}" >> /root/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> /root/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /root/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /root/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> /root/.bashrc && \
    echo "export TZ=$(date +%Z)" >> /root/.bashrc

WORKDIR ${QUADRUPED_RL_PATH}
ENTRYPOINT ["bash"]
