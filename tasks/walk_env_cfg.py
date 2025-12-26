from isaaclab.utils import configclass 

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class MySceneCfg(InteractiveSceneCfg):
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


@configclass 
class CommandsCfg:
    pass 


@configclass 
class ActionsCfg:
    pass 
    # joint_pos = mdp.JointPositionActionCfg(

    # ) 


@configclass
class ObservationsCfg:
    pass 


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
            
            


