from dataclasses import dataclass
import torch
import yaml


@dataclass
class ObsTerm:
    name: str
    dim: int 
    hist_len: int 
    buffer: torch.Tensor = None 

    def update(self, val: torch.Tensor):
        if self.buffer is None:
            self.buffer = val.repeat(self.hist_len)
        else:
            self.buffer = torch.cat([self.buffer[self.dim:], val])
        return self.buffer 



@dataclass
class Config:
    joint_order: list[str]
    default_joint_pos: torch.Tensor
    action_scale: float
    obs_terms: list[ObsTerm]
    stiffness: float
    damping: float
    control_dt: float
    cmd_names: list[str]
    cmd_scales: dict
    deploy_stiffness: float = 12.5

    @classmethod
    def from_yaml(cls, path: str, device="cpu"):
        with open(path) as f:
            cfg = yaml.safe_load(f)

        # Build observation terms from config
        obs_terms = []
        obs_cfg = cfg["observation"]
        num_obs_hist = obs_cfg.get("num_obs_hist", 1)
        for name in obs_cfg["order"]:
            dim = obs_cfg["dims"][name]
            if isinstance(num_obs_hist, dict):
                hist_len = num_obs_hist.get(name, 1)
            else:
                hist_len = num_obs_hist
            obs_terms.append(ObsTerm(name=name, dim=dim, hist_len=hist_len))

        # Build default joint positions in policy joint order
        # Prefer active_joint_order if joint_order is empty
        joint_order = cfg.get("joint_order") or cfg.get("active_joint_order", [])
        default_angles = cfg["init_state"]["default_joint_angles"]
        default_joint_pos = torch.tensor(
            [default_angles[name] for name in joint_order],
            dtype=torch.float32,
            device=device
        )

        # Command config
        cmd_names = cfg["commands"]["names"]
        cmd_scales = {
            name: cfg["commands"]["scales"].get(name, 1.0)
            for name in cmd_names
        }

        # PD gains — handle dict or scalar or empty
        raw_stiffness = cfg["control"].get("stiffness", 25.0)
        if isinstance(raw_stiffness, dict):
            raw_stiffness = list(raw_stiffness.values())[0] if raw_stiffness else 25.0
        raw_damping = cfg["control"].get("damping", 0.5)
        if isinstance(raw_damping, dict):
            raw_damping = list(raw_damping.values())[0] if raw_damping else 0.5

        # Deploy stiffness for compliance observation (midpoint of training range)
        deploy_stiffness = cfg.get("deploy", {}).get("stiffness", 12.5) if "deploy" in cfg else 12.5

        return cls(
            joint_order=joint_order,
            default_joint_pos=default_joint_pos,
            action_scale=cfg["control"]["action_scale"],
            obs_terms=obs_terms,
            stiffness=raw_stiffness,
            damping=raw_damping,
            control_dt=cfg["sim"]["dt"] * cfg["control"]["decimation"],
            cmd_names=cmd_names,
            cmd_scales=cmd_scales,
            deploy_stiffness=deploy_stiffness,
        )
    