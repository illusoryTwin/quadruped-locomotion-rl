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

from isaaclab.envs import ManagerBasedRLEnv
from compliance.compliance_manager_cfg import ComplianceManagerCfg
from modules.events import apply_sinusoidal_forces
from modules.commands.stiffness_command import StiffnessCommandCfg

def track_compliant_body_positions_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking MSD-deformed Cartesian body positions.

    Compares the actual Cartesian displacement of each compliant body
    (approximated via linear Jacobian) to the MSD-desired deformation x_def.

    reward = exp(-||J @ (q_actual - q_target) - x_def||^2 / std^2)

    All comparisons are in Cartesian space (meters), avoiding the J_pinv
    linearization error that a joint-space reward would introduce.
    """
    if not hasattr(env, 'compliance_manager') or env.compliance_manager is None:
        return torch.zeros(env.num_envs, device=env.device)

    cm = env.compliance_manager
    if cm._msd_system is None or cm._deformations is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    n_bodies = len(cm._compliant_body_names)
    body_indices = [asset.body_names.index(n) for n in cm._compliant_body_names]

    # Desired Cartesian deformation from MSD: [num_envs, n_bodies, 3]
    x_def = cm._msd_system.state['x_def']
    x_def_3d = x_def.reshape(env.num_envs, n_bodies, 3)

    # Per-env linear Jacobians for compliant bodies: [num_envs, n_bodies, 3, n_dofs]
    J_lin = asset.root_physx_view.get_jacobians()[:, body_indices, :3, :]
    if cm._joint_mask is not None:
        J_lin = J_lin.clone()
        J_lin[:, :, :, ~cm._joint_mask] = 0

    # Joint position error (actuated joints): [num_envs, num_joints]
    q_error = asset.data.joint_pos - asset._data.joint_pos_target

    # Pad with zeros for floating base DOFs: [num_envs, n_dofs]
    q_error_full = torch.zeros(env.num_envs, J_lin.shape[-1], device=env.device)
    q_error_full[:, 6:] = q_error

    # Actual Cartesian displacement of compliant bodies: [num_envs, n_bodies, 3]
    delta_p = torch.einsum('ebjd,ed->ebj', J_lin, q_error_full)

    # Error between actual and desired Cartesian deformation
    error = (delta_p - x_def_3d).reshape(env.num_envs, -1)

    return torch.exp(-torch.sum(error * error, dim=1) / (std * std))


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


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
    base_velocity =  mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.5, 1.5),
            heading=(-math.pi, math.pi)
        )
    )

    stiffness = StiffnessCommandCfg(
        resampling_time_range=(5.0, 10.0),
        ranges=StiffnessCommandCfg.Ranges(kp=(1000.0, 2000.0)), # 5.0, 20.0)),
    )


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
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"},
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        stiffness_cmd = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "stiffness"},
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15), scale=1.0)

    
    policy = PolicyCfg()
    critic = CriticCfg()



@configclass
class EventCfg:
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
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    pull_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 5.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*base"),
            "force_range": (-20.0, 20.0),
            "torque_range": (-5.0, 5.0),
        },
    )

    # Apply real physical forces - compliance manager will read these
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.1, 0.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-3.0, 3.0),
        },
    )

    # pull_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(3.0, 5.5),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*base"),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-3.0, 3.0),
    #     },
    # )

    # # Apply real physical forces - compliance manager will read these
    # push_robot = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.1, 2.5),
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
    #         "force_range": (-10.0, 10.0),
    #         "torque_range": (-3.0, 3.0),
    #     },
    # )

    # Apply sinusoidal forces to monitored bodies every step
    compliance_push = EventTerm(
        func=apply_sinusoidal_forces,
        mode="step",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base","FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
            "force_amplitude": 10.0,
            "frequency": 0.5,
        },
    )


@configclass 
class RewardsCfg:
    # -- task
    track_lin_vel_xy_exp = RewardTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.75, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    # lin_vel_z_l2 = RewardTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    track_compliant_targets = RewardTerm(
        func=track_compliant_body_positions_exp,
        weight=0.75,
        params={"std": 0.1}, # 05},  # meters (Cartesian space)
    )
    dof_torques_l2 = RewardTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    # dof_torques = RewardTerm(mdp.joint_torques_l2, weight=-1e-7)

    dof_acc_l2 = RewardTerm(func=mdp.joint_acc_l2, weight=-2e-7)
    action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.01)
    # feet_air_time = RewardTerm(
    #     func=feet_air_time,
    #     weight=0.25,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )_compute_compliance_targets


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
class UnitreeGo2WalkSoftEnvCfg(LocomotionVelocityRoughEnvCfg):
        scene: RoughTerrainSceneCfg = RoughTerrainSceneCfg(num_envs=4096, env_spacing=2.5)
        commands: CommandsCfg = CommandsCfg()
        actions: ActionsCfg = ActionsCfg()
        observations: ObservationsCfg = ObservationsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        compliance: ComplianceManagerCfg = ComplianceManagerCfg(
            enabled=True,
            compliant_bodies={
                "base": 1.0,
                "FL_calf": 0.8,
                "FR_calf": 0.8,
                "RL_calf": 0.8,
                "RR_calf": 0.8,
            },
            dt=0.02, # 0.004,
            base_stiffness=1500.0, # 10.0, # 30.0, # 60.0,
            base_inertia=0.5,
        )

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 20.0 
            self.sim.dt = 0.005
            
            


