import math
from isaaclab.utils import configclass 

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
from isaaclab.managers import ObservationGroupCfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
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
    class PolicyCfg(ObservationGroupCfg):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"": ""})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

    @configclass 
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.15, n_max=0.15), scale=1.0)

    
    policy = PolicyCfg()
    critic = CriticCfg()



@configclass
class EventCfg:
    pass 


@configclass 
class RewardsCfg:
    pass 


@configclass
class TerminationsCfg:
    pass


@configclass
class CurriculumCfg:
    pass 



@configclass 
class UnitreeGo2WalkEnvCfg(LocomotionVelocityRoughEnvCfg):
        scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
        commands: CommandsCfg = CommandsCfg()
        actions: ActionsCfg = ActionsCfg()
        observations: ObservationsCfg = ObservationsCfg()
        rewards: RewardsCfg = RewardsCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        event: EventCfg = EventCfg()
        curriculum: CurriculumCfg = CurriculumCfg()

        def __post_init__(self):
            self.decimation = 4
            self.episode_length_s = 20.0 
            self.sim.dt = 0.005
            
            


