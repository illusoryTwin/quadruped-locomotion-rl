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
        # stiffness_config: Joint names -> stiffness scale factors # for joint-space
        
        stiffness_config: body names -> stiffness scale factors
        dt: MSD timestep
        base_stiffness
        base_inertia
        debug: Enable debug printing
    """

    enabled: bool = True
    robot_name: str = "robot"

    compliant_bodies: Dict[str, float] = field(default_factory=lambda: {
        "base": 1.0,
        "FL_calf": 0.8,
        "FR_calf": 0.8,
        "RL_calf": 0.8,
        "RR_calf": 0.8,
    })

    dt: float = 0.02 # 0.004
    base_stiffness: float = 60.0 #TODO: check stiffness value
    base_inertia: float = 0.5 #TODO: check inertia value
    max_deformation: float = 0.5  # max absolute joint deformation in radians
    debug: bool = False

    # Teacher-student annealing: inject deformation into PD targets,
    # linearly reducing injection so the policy learns to produce compliance.
    injection_alpha_start: float = 1.0   # initial injection fraction (1.0 = full)
    injection_alpha_end: float = 0.0     # final injection fraction (0.0 = none)
    injection_anneal_iters: int = 3000   # linear anneal over this many training iters