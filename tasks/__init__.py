import gymnasium as gym

from .walk_env_cfg import UnitreeGo2WalkEnvCfg
from .rsl_rl_ppo_cfg import UnitreeGo2PPORunnerCfg

gym.register(
    id="go2_walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.walk_env_cfg:UnitreeGo2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)
