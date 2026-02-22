import math
import torch

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
import isaaclab.envs.mdp as mdp
from isaaclab.sensors import RayCasterCfg, ContactSensorCfg, patterns
from isaaclab.managers import RewardTermCfg as RewardTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from src.compliance.compliance_manager_cfg import ComplianceManagerCfg
from src.modules.events import apply_sinusoidal_forces, apply_sinusoidal_forces_xy, apply_sinusoidal_forces_z
from src.modules.commands.stiffness_command import StiffnessCommandCfg
from src.modules.commands.base_position_command import BasePositionCommandCfg



def track_base_position_exp(
    env: ManagerBasedRLEnv,
    command_name: str = "base_position",
    std: float = 0.1,
) -> torch.Tensor:
    """Reward for tracking the commanded base position.

    reward = exp(-||pos_actual - (env_origin + pos_cmd)||^2 / std^2)
    """
    robot = env.scene["robot"]
    pos_cmd = env.command_manager.get_command(command_name)
    target_pos = env.scene.env_origins[:, :3] + pos_cmd
    pos_err = (robot.data.root_pos_w[:, :3] - target_pos).square().sum(dim=1)
    return torch.exp(-pos_err / std**2)


def base_cartesian_deformation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation term: base body Cartesian deformation from MSD. Shape [num_envs, 3]."""
    if hasattr(env, 'compliance_manager') and env.compliance_manager is not None:
        msd = env.compliance_manager._msd_system
        if msd is not None:
            return msd.state['x_def'][:, 0:3]
    return torch.zeros(env.num_envs, 3, device=env.device)


def track_compliant_base_pos_tanh(
    env: ManagerBasedRLEnv,
    pos_scale: float = 0.5,
) -> torch.Tensor:
    """Saturated position tracking penalty using tanh.

    reward = -tanh(||x_sim - x_ref|| / pos_scale)

    Gives gradient for small errors but saturates at -1 for large drift,
    preventing accumulated integration error from dominating the reward.
    """
    if not hasattr(env, '_compliant_ref_pos') or env._compliant_ref_pos is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    pos_err = (robot.data.root_pos_w[:, :3] - env._compliant_ref_pos).norm(dim=1)

    return -torch.tanh(pos_err / pos_scale)


def track_compliant_velocity_l2(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for deviating from compliant velocity reference.

    vel_ref = commanded_velocity_world + MSD_deformation_velocity

    When no external forces: dx_def ~ 0 -> tracks commanded velocity.
    When pushed: dx_def != 0 -> robot is rewarded for yielding to the push.

    reward = -||v_sim - vel_ref||^2
    """
    if not hasattr(env, '_compliant_ref_vel') or env._compliant_ref_vel is None:
        return torch.zeros(env.num_envs, device=env.device)

    robot = env.scene["robot"]
    vel_err = torch.sum((robot.data.root_lin_vel_w[:, :3] - env._compliant_ref_vel) ** 2, dim=1)

    return -vel_err



@configclass
class RoughTerrainSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
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

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass 
class CommandsCfg:
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 0.0),
    #         lin_vel_y=(0.0, 0.0),
    #         ang_vel_z=(0.0, 0.0),
    #         heading=(-math.pi, math.pi)
    #     )
    # )
    base_position = BasePositionCommandCfg(
        resampling_time_range=(10.0, 10.0),
        ranges=BasePositionCommandCfg.Ranges(
            x=(-1.5, 1.5),
            y=(-1.5, 1.5),
            z=(0.3, 0.3),
        ),
    )
    # stiffness = StiffnessCommandCfg(
    #     resampling_time_range=(10.0, 10.0), # (5.0, 10.0),
    #     ranges=StiffnessCommandCfg.Ranges(kp=(50.0, 70.0)), # 70.0, 100.0)), # kp=(100.0, 200.0)),
    # )

@configclass 
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.5,
        use_default_offset=True
    )



@configclass 
class ObservationsCfg:
    @configclass 
    class PolicyCfg(ObsGroup):
        # Observation history
        concatenate_terms = True 
        history_length = 10 
        flatten_history_dim = True

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(mdp.last_action)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # velocity_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_velocity"}
        # )
        base_position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_position"}
        )
        # stiffness_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "stiffness"}
        # )
        # base_ang_vel = 

    @configclass 
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # cartesian_deformation = ObsTerm(func=base_cartesian_deformation)

    policy = PolicyCfg()
    critic = CriticCfg()


@configclass 
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
    )

    
    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_offset,
    #     mode="reset",
    #     params={
    #         "position_range": (-1.0, 1.0),
    #         "velocity_range": (-0.5, 0.5),
    #     }
    # )
    # pull_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval", 
    #     interval_range_s=(5.0, 5.5),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #         "force_range": (-20.0, 20.0),
    #         "torque_range": (-5.0, 5.0),
    #     }
    # )
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     # interval_range_s=(0.1, 0.2),
    #     interval_range_s=(0.1, 0.5),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-3.0, 3.0),
    #         # "force_range": (-20.0, 20.0),
    #         # "torque_range": (-5.0, 5.0),
    #     }
    # )

    # Apply sinusoidal forces to base body every step (XY only)
    compliance_push_xy = EventTerm(
        func=apply_sinusoidal_forces_xy,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_amplitude": [20.0],
            "frequency": 0.5,
        },
    ) 



@configclass 
class RewardsCfg:
    # track_lin_vel = RewardTerm(
    #     func=mdp.track_lin_vel_xy_exp,
    #     weight=1.5, # 0.25, # 1.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "std": 0.5,
    #     }
    # )
    # track_ang_vel = RewardTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=0.75, # 0.125, # 0.5,
    #     params={
    #         "command_name": "base_velocity",
    #         "std": 0.5,
    #     }
    # )

    track_base_pos = RewardTerm(
        func=track_base_position_exp,
        weight=1.0,
        params={
            "command_name": "base_position",
            "std": 0.1,
        },
    )
    joint_default_pos = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    flat_orientation = RewardTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    base_height = RewardTerm(
        func=mdp.base_height_l2,
        weight=-1.5,
        params={"asset_cfg": SceneEntityCfg("robot"), 
                "target_height": 0.3
        },
    )

    dof_torques = RewardTerm(mdp.joint_torques_l2, weight=-1e-7)
    dof_acc_l2 = RewardTerm(func=mdp.joint_acc_l2, weight=-2e-7)
    action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.01)



@configclass 
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                                    "threshold": 1.0}
    )


@configclass 
class CurriculumCfg:
    pass 


@configclass
class UnitreeGo2StanceEnvCfg(ManagerBasedRLEnvCfg): # LocomotionVelocityRoughEnvCfg):
        scene: RoughTerrainSceneCfg = RoughTerrainSceneCfg(num_envs=4096, env_spacing=2.5)
        commands: CommandsCfg = CommandsCfg()
        actions: ActionsCfg = ActionsCfg()
        observations: ObservationsCfg = ObservationsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        # compliance: ComplianceManagerCfg = ComplianceManagerCfg(
        #     enabled=True,
        #     compliant_bodies={"base": 1.0},
        #     dt=0.02,
        #     base_stiffness=100.0,
        #     base_inertia=0.5,
        # )

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 20.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt
