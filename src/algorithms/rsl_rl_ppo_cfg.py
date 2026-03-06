from isaaclab.utils import configclass 

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlPpoActorCriticCfg

@configclass
class UnitreeGo2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_walk"
    empirical_normalization = True  # For RSL_RL 2.x compatibility
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 256],
        activation="elu",
        actor_obs_normalization=True,   # RSL_RL 3.x: normalize actor observations
        critic_obs_normalization=True,  # RSL_RL 3.x: normalize critic observations
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=0.5,
        # smooth_loss_coef=2e-3,
    )


@ configclass
class UnitreeGo2PPORunnerCfg2500(UnitreeGo2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_walk"
        self.max_iterations = 2500


@ configclass
class UnitreeGo2SoftPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_walk_soft"
        self.max_iterations = 2500


@configclass
class UnitreeGo2PosTrackingPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_pos_xy_tracking"
        self.max_iterations = 2500


@configclass
class UnitreeGo2SoftPosTrackingPPORunnerCfg(UnitreeGo2PPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "go2_soft_pos_xy_tracking"
        self.max_iterations = 2500