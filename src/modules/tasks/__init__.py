import gymnasium as gym

from .flat_walk_env_cfg import UnitreeGo2WalkEnvCfg
from .flat_walk_soft_env_cfg import UnitreeGo2WalkSoftEnvCfg
from .rough_walk_env_cfg import UnitreeGo2WalkRoughEnvCfg
from .stairs_climbing_env_cfg import UnitreeGo2WalkStairsEnvCfg
from .compliant_stance_env_cfg import UnitreeGo2StanceEnvCfg
from src.algorithms.rsl_rl_ppo_cfg import UnitreeGo2PPORunnerCfg

gym.register(
    id="go2_walk_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.flat_walk_env_cfg:UnitreeGo2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_walk_rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.rough_walk_env_cfg:UnitreeGo2WalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_stairs_climbing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.stairs_climbing_env_cfg:UnitreeGo2WalkStairsEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_compliant_locomotion",
    entry_point="src.modules.envs.compliant_rl_env:CompliantRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.flat_walk_soft_env_cfg:UnitreeGo2WalkSoftEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2SoftPPORunnerCfg",
    },
) 

# gym.register(
#     id="go2_compliant_stance",
#     entry_point="src.modules.envs.compliant_rl_env:CompliantRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": "src.modules.tasks.compliant_stance_env_cfg:UnitreeGo2StanceEnvCfg",
#         "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2SoftPPORunnerCfg",
#     },
# ) 

gym.register(
    id="go2_compliant_stance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    # entry_point="src.modules.envs.compliant_stability_rl_env:CompliantStabilityRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.compliant_stance_env_cfg:UnitreeGo2StanceEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2SoftPPORunnerCfg",
    },
) 


# ===========================

gym.register(
    id="go2_orientation_tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.experiments.track_heading_env_cfg:UnitreeGo2OrientationEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_pos_xy_tracking",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.experiments.pos_xy_tracking_env_cfg:UnitreeGo2PosTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)