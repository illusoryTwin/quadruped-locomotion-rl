"""
Script to spawn Unitree Go2 robot in Isaac Sim.

Usage:
    # Activate isaacsim conda environment first:
    conda activate isaacsim

    # Run with Isaac Lab launcher:
    cd /home/ekaterina-mozhegova/sber_ws/RL/IsaacLab
    ./isaaclab.sh -p /home/ekaterina-mozhegova/Workspace/Projects/quadruped-locomotion-rl/scripts/spawn_unitree_go2.py
"""

import argparse

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Spawn Unitree Go2 robot in Isaac Sim.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of Go2 robots to spawn.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# Import Unitree Go2 configuration
from isaaclab_assets import UNITREE_GO2_CFG


def design_scene(num_robots: int = 1) -> tuple[dict, list[list[float]]]:
    """Designs the scene with Unitree Go2 robot(s)."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Calculate grid layout for multiple robots
    origins = []
    spacing = 1.5
    cols = int(torch.ceil(torch.sqrt(torch.tensor(num_robots))).item())

    for i in range(num_robots):
        row = i // cols
        col = i % cols
        x = col * spacing - (cols - 1) * spacing / 2
        y = row * spacing - ((num_robots - 1) // cols) * spacing / 2
        origins.append([x, y, 0.0])

    # Spawn Go2 robots
    robots = {}
    for i, origin in enumerate(origins):
        prim_path = f"/World/Go2_{i}"
        prim_utils.create_prim(prim_path, "Xform", translation=origin)
        robot = Articulation(UNITREE_GO2_CFG.replace(prim_path=f"{prim_path}/Robot"))
        robots[f"go2_{i}"] = robot

    return robots, origins


def run_simulator(sim: sim_utils.SimulationContext, robots: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("[INFO]: Starting simulation. Press Ctrl+C or close the window to exit.")

    while simulation_app.is_running():
        # Reset robots periodically
        if count % 500 == 0:
            sim_time = 0.0
            count = 0

            for index, robot in enumerate(robots.values()):
                # Root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # Joint state
                joint_pos = robot.data.default_joint_pos.clone()
                joint_vel = robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                robot.reset()

            print(f"[INFO]: Reset robots to default state at t={sim_time:.2f}s")

        # Apply default joint positions (standing pose)
        for robot in robots.values():
            joint_pos_target = robot.data.default_joint_pos.clone()
            robot.set_joint_position_target(joint_pos_target)
            robot.write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1

        # Update robot buffers
        for robot in robots.values():
            robot.update(sim_dt)


def main():
    """Main function."""
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])

    # Design scene
    robots, origins = design_scene(num_robots=args_cli.num_envs)
    origins = torch.tensor(origins, device=sim.device)

    # Reset simulation
    sim.reset()
    print("[INFO]: Setup complete. Unitree Go2 spawned successfully!")
    print(f"[INFO]: Number of robots: {args_cli.num_envs}")

    # Run simulator
    run_simulator(sim, robots, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
