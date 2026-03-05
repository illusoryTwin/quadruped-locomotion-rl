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
from src.modules.commands.heading_command import HeadingVelocityCommandCfg
from src.modules.commands.position_command import PositionCommandCfg


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

    # Disable height_scanner from parent class (robot_ros2 doesn't use it)
    height_scanner = None

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass 
class CommandsCfg:
    # base_orientation = HeadingVelocityCommandCfg(
    #     asset_name="robot",
    #     heading_command=True,
    #     debug_vis=True,
    #     resampling_time_range=(10.0, 10.0),
    #     ranges=HeadingVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.0, 0.0),
    #         lin_vel_y=(0.0, 0.0),
    #         ang_vel_z=(-1.5, 1.5),
    #         heading=(-math.pi, math.pi),
    #     ),
    # )
      
    base_position = PositionCommandCfg(
        asset_name="robot",
        position_control_stiffness=1.0,
        debug_vis=True,
        resampling_time_range=(10.0, 10.0),
        ranges=PositionCommandCfg.Ranges(
            # pos_x=(-0.5, 0.5),
            # pos_y=(-0.5, 0.5),
            # pos_x=(-1.0, 1.0),
            # pos_y=(-1.0, 1.0),
            # pos_x=(-1.5, 1.5),
            # pos_y=(-1.5, 1.5),
            pos_x=(-2.0, 2.0),
            pos_y=(-2.0, 2.0),
            vel=(-1.5, 1.5),
        ),
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
        # Observation history (10 frames)
        concatenate_terms = True
        history_length = 10
        flatten_history_dim = True

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # orientation_commands = ObsTerm(
        #     func=mdp.generated_commands,
        #     params={"command_name": "base_orientation"},
        # )
        position_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_position"},
        )

        # Disable height_scan
        height_scan = None

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15), scale=1.0)


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

@configclass
class RewardsCfg:
    # -- task
    # track_ang_vel_z_exp = RewardTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=1.5,
    #     params={"command_name": "base_orientation", "std": math.sqrt(0.25)}
    # )

    track_lin_vel_xy_exp = RewardTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_position", "std": math.sqrt(0.25)},
    )

    lin_vel_z_l2 = RewardTerm(func=mdp.lin_vel_z_l2, weight=-1.0) # -2.0)

    # -- stance stability
    base_height_l2 = RewardTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": 0.3},
    )
    
    flat_orientation_l2 = RewardTerm(
        func=mdp.flat_orientation_l2, 
        weight=-1.0
    )
    
    
    joint_default_pos = RewardTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # -- penalties
    dof_torques = RewardTerm(func=mdp.joint_torques_l2, weight=-1e-7)
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
class UnitreeGo2PosTrackingEnvCfg(LocomotionVelocityRoughEnvCfg):
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
            
            


