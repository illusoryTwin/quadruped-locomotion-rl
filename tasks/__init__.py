import gymnasium as gym

from .flat_walk_env_cfg import UnitreeGo2WalkEnvCfg
from .rough_terrain_walk_env_cfg import UnitreeGo2WalkRoughEnvCfg
from .rsl_rl_ppo_cfg import UnitreeGo2PPORunnerCfg

gym.register(
    id="go2_walk_flat_terrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.flat_walk_env_cfg:UnitreeGo2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_walk_rough_terrain",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.rough_terrain_walk_env_cfg:UnitreeGo2WalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)
