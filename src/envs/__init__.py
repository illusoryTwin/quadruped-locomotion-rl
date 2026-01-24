import gymnasium as gym

from .flat_walk_env_cfg import UnitreeGo2WalkEnvCfg
from .rough_walk_env_cfg import UnitreeGo2WalkRoughEnvCfg
from .stairs_climbing_env_cfg import UnitreeGo2WalkStairsEnvCfg
from ..algorithms.rsl_rl_ppo_cfg import UnitreeGo2PPORunnerCfg

gym.register(
    id="go2_walk_flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.flat_walk_env_cfg:UnitreeGo2WalkEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_walk_rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.rough_walk_env_cfg:UnitreeGo2WalkRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(
    id="go2_stairs_climbing",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.stairs_climbing_env_cfg:UnitreeGo2WalkStairsEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",
    },
)

gym.register(                                                                                                                                                                                                                                                                           
    id="go2_compliant_locomotion",                                                                                                                                                                                                                                                                    
    entry_point="modules.envs:CompliantRLEnv",                                                                                                                                                                                                              
    disable_env_checker=True,                                                                                                                                                                                                                                                           
    kwargs={                                                                                                                                                                                                                                                                            
        "env_cfg_entry_point": "tasks.rough_walk_env_cfg:UnitreeGo2WalkRoughEnvCfg",                                                                                                                                                                                                    
        "rsl_rl_cfg_entry_point": "tasks.rsl_rl_ppo_cfg:UnitreeGo2PPORunnerCfg",                                                                                                                                                                                                        
    },                                                                                                                                                                                                                                                                                  
) 