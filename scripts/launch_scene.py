import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Launch MySceneCfg scene.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


# import torch

# import isaaclab.sim as sim_utils
# from isaaclab.scene import InteractiveScene

# # Import scene configuration
# import sys
# sys.path.insert(0, "/home/kate/Workspace/Projects/thesis/quadruped-locomotion-rl")
# from tasks.walk_env_cfg import MySceneCfg


# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     """Runs the simulation loop."""
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     print("[INFO]: Starting simulation. Press Ctrl+C or close the window to exit.")

#     while simulation_app.is_running():
#         # Reset periodically
#         if count % 500 == 0:
#             sim_time = 0.0
#             count = 0

#             # Reset scene to initial state
#             root_state = scene["robot"].data.default_root_state.clone()
#             scene["robot"].write_root_pose_to_sim(root_state[:, :7])
#             scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

#             joint_pos = scene["robot"].data.default_joint_pos.clone()
#             joint_vel = scene["robot"].data.default_joint_vel.clone()
#             scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
#             scene["robot"].reset()

#             print(f"[INFO]: Reset scene at t={sim_time:.2f}s")

#         # Apply default joint positions (standing pose)
#         joint_pos_target = scene["robot"].data.default_joint_pos.clone()
#         scene["robot"].set_joint_position_target(joint_pos_target)
#         scene["robot"].write_data_to_sim()

#         # Step simulation
#         sim.step()
#         sim_time += sim_dt
#         count += 1

#         # Update scene
#         scene.update(sim_dt)


# def main():
#     """Main function."""
#     # Initialize simulation context
#     sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
#     sim = sim_utils.SimulationContext(sim_cfg)

#     # Set camera view
#     sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.3])

#     # Create scene configuration
#     scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)

#     # Instantiate the scene
#     scene = InteractiveScene(scene_cfg)

#     # Reset simulation
#     sim.reset()

#     print("[INFO]: Setup complete!")
#     print(f"[INFO]: Number of environments: {args_cli.num_envs}")

#     # Run simulator
#     run_simulator(sim, scene)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()
