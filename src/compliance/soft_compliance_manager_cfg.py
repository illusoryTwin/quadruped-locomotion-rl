"""Configuration for soft compliance manager."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SoftComplianceManagerCfg:
    """Configuration for soft joint compliance using MSD model.

    Attributes:
        enabled: Whether compliance is enabled
        robot_name: Name of the robot articulation in the scene
        monitored_bodies: Bodies where external forces are measured
        stiffness_config: Joint names -> stiffness scale factors
        dt: MSD timestep
        base_stiffness
        base_inertia
        debug: Enable debug printing
    """

    enabled: bool = True
    robot_name: str = "robot"

    # TODO: change for dog's bodies
    monitored_bodies: List[str] = field(default_factory=lambda: [
        # "torso_yaw_cover",
        # "head_link",
        # "left_hand_base_link",
        # "right_hand_base_link",
    ])

    # TODO: change for dog's bodies
    stiffness_config: Dict[str, float] = field(default_factory=lambda: {
        # "torso_yaw_joint": 2.0,
        # "left_shoulder_pitch_joint": 0.8,
        # "right_shoulder_pitch_joint": 0.8,
        # "left_shoulder_roll_joint": 0.8,
        # "right_shoulder_roll_joint": 0.8,
        # "left_shoulder_yaw_joint": 0.8,
        # "right_shoulder_yaw_joint": 0.8,
        # "left_elbow_pitch_joint": 0.6,
        # "right_elbow_pitch_joint": 0.6,
        # "left_elbow_yaw_joint": 0.6,
        # "right_elbow_yaw_joint": 0.6,
        # "neck_yaw_joint": 0.5,
        # "neck_pitch_joint": 0.5,
    })

    dt: float = 0.004
    base_stiffness: float = 60.0 #TODO: check stiffness value
    base_inertia: float = 0.5 #TODO: check inertia value
    debug: bool = False