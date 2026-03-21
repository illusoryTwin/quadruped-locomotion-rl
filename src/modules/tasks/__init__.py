import gymnasium as gym

from .flat_walk_env_cfg import UnitreeGo2WalkEnvCfg
from .compliant_stance_env_cfg import UnitreeGo2StanceEnvCfg
from .compliant_stance_fixed_stiffness_env_cfg import UnitreeGo2StanceFixedStiffnessEnvCfg

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
    id="go2_compliant_stance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.compliant_stance_env_cfg:UnitreeGo2StanceEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2SoftStancePPORunnerCfg",
    },
) 

gym.register(
    id="go2_compliant_stance_fixed_stiffness",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.compliant_stance_fixed_stiffness_env_cfg:UnitreeGo2StanceFixedStiffnessEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2FixedStiffnessStancePPORunnerCfg",
    },
)

gym.register(
    id="go2_default_stance",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.stance_env_cfg:UnitreeGo2DefaultStanceEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2DefaultStancePPORunnerCfg",
    },
) 

gym.register(
    id="go2_soft_walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "src.modules.tasks.compliant_walk_env_cfg:UnitreeGo2SoftWalkEnvCfg",
        "rsl_rl_cfg_entry_point": "src.algorithms.rsl_rl_ppo_cfg:UnitreeGo2SoftWalkPPORunnerCfg",
    },
