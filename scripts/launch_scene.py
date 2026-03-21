"""
Script to launch the MySceneCfg scene with Unitree Go2 robot.

Usage:
    cd /path/to/IsaacLab
    ./isaaclab.sh -p /home/kate/Workspace/Projects/thesis/quadruped-locomotion-rl/scripts/launch_scene.py --num_envs 4
"""

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

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Scene configuration with Unitree Go2 robot."""

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    print("[INFO]: Starting simulation. Press Ctrl+C or close the window to exit.")

    while simulation_app.is_running():
        # Reset periodically
        if count % 500 == 0:
            sim_time = 0.0
            count = 0

            # Reset scene to initial state
            root_state = scene["robot"].data.default_root_state.clone()
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = scene["robot"].data.default_joint_pos.clone()
            joint_vel = scene["robot"].data.default_joint_vel.clone()
            scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
            scene["robot"].reset()

            print(f"[INFO]: Reset scene at t={sim_time:.2f}s")

        # Apply default joint positions (standing pose)
        joint_pos_target = scene["robot"].data.default_joint_pos.clone()
        scene["robot"].set_joint_position_target(joint_pos_target)
        scene["robot"].write_data_to_sim()

        # Step simulation
        sim.step()
        sim_time += sim_dt
        count += 1

        # Update scene
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set camera view
    sim.set_camera_view(eye=[3.0, 3.0, 2.0], target=[0.0, 0.0, 0.3])

    # Create scene configuration
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)

    # Instantiate the scene
    scene = InteractiveScene(scene_cfg)

    # Reset simulation
    sim.reset()

    print("[INFO]: Setup complete!")
    print(f"[INFO]: Number of environments: {args_cli.num_envs}")

    # Run simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
