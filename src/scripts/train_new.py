# # parse args
# import cli_args
# import gymnasium as gym
# import os
# import argparse
# from rsl_rl.runners import OnPolicyRunner


# from isaaclab.app import AppLauncher

# from datetime import datetime 

# parser = argparse.ArgumentParser()
# parser.add_argument("--task", type=str, required=True)
# parser.add_argument("--max_iterations", type=int, default=None)
# parser.add_argument("--video", action="store_true")
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# # Note: --device is added by AppLauncher.add_app_launcher_args()
# parser.add_argument(
#     "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
# )


# cli_args.add_rsl_rl_args(parser)
# AppLauncher.add_app_launcher_args(parser)
# args_cli, hydra_args = parser.parse_known_args()

# # clear out sys.argv for Hydra
# import sys
# sys.argv = [sys.argv[0]] + hydra_args

# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app 

# import logging
# import torch
# from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit
# from isaaclab_tasks.utils.hydra import hydra_task_config
# from isaaclab.utils.io import dump_yaml

# # Import envs to register gym environments
# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))  # Add src/ to path
# import envs

# logger = logging.getLogger(__name__)

# # PLACEHOLDER: Extension template (do not remove this comment)

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False




# # def parse_args():
# #     if args_cli.max_iterations:
# #         agent_cfg.max_iterations = args_cli.max_iterations
# #     if args_cli.video:
# #         agent_cfg.video = True
# #     if args_cli.device:
# #         agent_cfg.device = args_cli.device

# #     cli_args.add_rsl_rl_args(parser)

# # parse_args



# # def export_policy_as_jit(runner, normalizer, export_path, filename="policy.*"):
# #     # export policy
# #     pass 


# def export_jit_policy(env, runner, export_dir):
#     runner.eval_mode()

#     # RSL-RL 3.x stores actor_critic as .policy
#     actor_critic = runner.alg.policy

#     # RSL-RL 3.x: normalizer is inside actor_critic, not on runner
#     normalizer = None
#     if hasattr(actor_critic, 'actor_obs_normalizer'):
#         normalizer = actor_critic.actor_obs_normalizer
#         print(f"[DEBUG] Found normalizer in actor_critic.actor_obs_normalizer")
#     elif hasattr(runner, 'obs_normalizer'):
#         normalizer = runner.obs_normalizer
#         print(f"[DEBUG] Found normalizer in runner.obs_normalizer")
#     else:
#         print(f"[WARNING] No normalizer found!")

#     # Debug info
#     print(f"[DEBUG] Exporting policy to: {export_dir}")
#     print(f"[DEBUG] actor_critic type: {type(actor_critic)}")
#     print(f"[DEBUG] normalizer type: {type(normalizer)}")
#     if normalizer is not None and hasattr(normalizer, '_mean'):
#         print(f"[DEBUG] normalizer mean shape: {normalizer._mean.shape}")
#         print(f"[DEBUG] normalizer mean range: [{normalizer._mean.min():.3f}, {normalizer._mean.max():.3f}]")
#         print(f"[DEBUG] normalizer std range: [{normalizer._std.min():.3f}, {normalizer._std.max():.3f}]")

#     # Export to JIT
#     filename = "policy.pt"
#     export_policy_as_jit(actor_critic, normalizer=normalizer, path=export_dir, filename=filename)

#     runner.train_mode()


# def callback(env, env_cfg, agent_cfg, log_dir, runner):
#     # dump the configuration into log-directory
#     dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
#     dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

#     export_model_dir = os.path.join(log_dir, "exported")

#     def callback():
#         export_policy(export_model_dir)
#     runner.save = callback

# @hydra_task_config(args_cli.task, args_cli.agent)
# def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
#     agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
#     agent_cfg.max_iterations = (
#         args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
#     )

#     env_cfg.seed = agent_cfg.seed
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    

#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     log_dir = os.path.join(log_root_path, log_dir)
#     env_cfg.log_dir = log_dir

#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
#     runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device) 

#     env.seed(agent_cfg.seed)

#     # Patch runner's save method to also export JIT
#     original_save = runner.save
#     def save_with_jit_export(it):
#         original_save(it)
#         export_jit_policy(env, runner, os.path.join(log_dir, "exported"))   
#     runner.save = save_with_jit_export

#     # run training 
#     runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

#     # close the simulator 
#     env.close()

# if __name__ == "__main__":
#     main()
#     # close sim app
#     simulation_app.close()

