import math
import torch

from isaaclab.utils import configclass
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
from src.modules.events import apply_sinusoidal_forces_z, apply_sinusoidal_forces_xy
from src.modules.commands.stiffness_command import StiffnessCommandCfg
from src.modules.commands.base_position_command import BasePositionCommandCfg
from src.modules.commands.compliance_command import ComplianceCommandCfg
from src.modules.rewards import track_compliant_base_pos_cmd_exp, base_cartesian_deformation, feet_contact
from src.modules.actions import EMAJointPositionActionCfg

from isaaclab.managers import CurriculumTermCfg as CurrTerm
import isaaclab.envs.mdp as mdp_curr
from src.modules.curriculums import ramp_force_amplitude

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from src.compliance.compliance_manager_cfg import ComplianceManagerCfg
from src.modules.events import apply_sinusoidal_forces_z, apply_sinusoidal_forces_xy, apply_constant_force_z, log_env0_compliance
from src.modules.commands.stiffness_command import StiffnessCommandCfg
from src.modules.commands.base_position_command import BasePositionCommandCfg
from src.modules.commands.compliance_command import ComplianceCommandCfg
from src.modules.rewards import track_compliant_base_pos_cmd_exp, base_cartesian_deformation, feet_contact, ang_vel_z_l2, lin_vel_xy_l2
from src.modules.curriculums import staged_force_ramp, multi_stage_stiffness


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
    base_position = BasePositionCommandCfg(
        # resampling_time_range=(10.0, 10.0),
        resampling_time_range=(7.0, 7.0),
        ranges=BasePositionCommandCfg.Ranges(
            x=(0.0, 0.0),
            y=(0.0, 0.0),
            z=(0.3, 0.3),
            # z=(0.25, 0.4),
            # z=(0.3, 0.38),
        ),
    )
    
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
        resampling_time_range=(5.0, 5.0),
        ranges=StiffnessCommandCfg.Ranges(kp=(400.0, 400.0)), # 30.0, 50.0)),
    )

    compliance = ComplianceCommandCfg(
        resampling_time_range=(1e9, 1e9),  # never resample
        compliance_cfg=ComplianceManagerCfg(),
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
        position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_position"}
        )
        stiffness_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "stiffness"}
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

    @configclass 
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        cartesian_deformation = ObsTerm(func=base_cartesian_deformation)

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
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        }
    )
    # Z-only sinusoidal force on base, resampled every 5-5.5s
    compliance_push = EventTerm(
        func=apply_sinusoidal_forces_z,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        # interval_range_s=(5.0, 5.5),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_amplitude": [70.0], # [100.0],
            # "force_amplitude": [50.0],
            # "frequency": 0.3,
            "frequency": 0.5,
        },
    )



@configclass
class RewardsCfg:
    # Compliant position tracking (XYZ) — command + MSD deformation
    track_lin_vel_xy_exp = RewardTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=1.5, # 1.0, # 1.5, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=0.75, # 0.5, # 0.75, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_compliant_pos = RewardTerm(
        func=track_compliant_base_pos_cmd_exp,
        weight=2.5,
        params={"command_name": "base_position", "std":0.05}, #  0.04},
    )

    # ang_vel_xy_l2 = RewardTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # lin_vel_z_l2 = RewardTerm(func=mdp.lin_vel_z_l2, weight=-0.075)

    illegal_contact = RewardTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh"]),
                "threshold": 1.0},
    )
    # feet_on_ground = RewardTerm(
    #     func=feet_contact,
    #     weight=0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #             "threshold": 1.0},
    # )
    flat_orientation = RewardTerm(func=mdp.flat_orientation_l2, weight=-1.0) # -0.5) # -1.0)
    joint_default_pos = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.08, # -0.1, # -0.075, 
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    # action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.15) # -0.3) # -0.25) # -0.15) #-0.01)
    # dof_torques = RewardTerm(mdp.joint_torques_l2, weight=-1e-7) # -5e-7)
    # dof_acc_l2 = RewardTerm(func=mdp.joint_acc_l2, weight=-5e-7)
    
    # # action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.25) # -0.25) # -0.15) #-0.01)
    # # dof_torques = RewardTerm(mdp.joint_torques_l2, weight=-1e-7)
    # # dof_acc_l2 = RewardTerm(func=mdp.joint_acc_l2, weight=-2.5e-6) # -2.5e-6) # -5e-7)
    
    dof_torques = RewardTerm(mdp.joint_torques_l2, weight=-2e-7) # -1e-7)
    dof_acc_l2 = RewardTerm(func=mdp.joint_acc_l2, weight=-5e-7) # -2e-7)
    # action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.05) #-0.01)
    action_rate_l2 = RewardTerm(func=mdp.action_rate_l2, weight=-0.1) # -0.15) #-0.01)


@configclass 
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(func=mdp.illegal_contact,
                            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                                    "threshold": 1.0}
    )


@configclass
class CurriculumCfg:
    # Simple force ramp: 0 N for first 1000 iters, then linearly 0→70 N over next 1000 iters
    # warmup_steps = 1000 iters × 24 steps/iter = 24000
    # ramp_steps   = 1000 iters × 24 steps/iter = 24000
    force_amplitude = CurrTerm(
        func=mdp_curr.modify_term_cfg,
        params={
            "address": "events.compliance_push.params.force_amplitude",
            "modify_fn": ramp_force_amplitude,
            "modify_params": {
                "initial": 0.0,
                "final": 70.0,
                "warmup_steps": 24000,   # 1000 iters × 24 steps
                "ramp_steps": 24000,     # 1000 iters × 24 steps
            },
        },
    )


@configclass
class UnitreeGo2SoftWalkEnvCfg(ManagerBasedRLEnvCfg):
        scene: RoughTerrainSceneCfg = RoughTerrainSceneCfg(num_envs=4096, env_spacing=2.5)
        commands: CommandsCfg = CommandsCfg()
        actions: ActionsCfg = ActionsCfg()
        observations: ObservationsCfg = ObservationsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        events: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 20.0
            self.sim.dt = 0.005
            self.sim.render_interval = self.decimation
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt

