"""Configuration for soft compliance manager."""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ComplianceManagerCfg:
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

    # Unitree Go2 bodies for force monitoring
    monitored_bodies: List[str] = field(default_factory=lambda: [
        "base",
        # "FL_calf",
        # "FR_calf",
        # "RL_calf",
        # "RR_calf",
    ])

    # Unitree Go2 joint stiffness scale factors
    stiffness_config: Dict[str, float] = field(default_factory=lambda: {
        "FL_hip_joint": 1.0,
        "FL_thigh_joint": 1.0,
        "FL_calf_joint": 0.8,
        "FR_hip_joint": 1.0,
        "FR_thigh_joint": 1.0,
        "FR_calf_joint": 0.8,
        "RL_hip_joint": 1.0,
        "RL_thigh_joint": 1.0,
        "RL_calf_joint": 0.8,
        "RR_hip_joint": 1.0,
        "RR_thigh_joint": 1.0,
        "RR_calf_joint": 0.8,
    })

    dt: float = 0.02 # 0.004
    base_stiffness: float = 60.0 #TODO: check stiffness value
    base_inertia: float = 0.5 #TODO: check inertia value
    debug: bool = False