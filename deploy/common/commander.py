"""
Command generator for robot control.

Handles:
- Velocity commands (vx, vy, wz)
- Command scaling and clipping
- Extensible for additional command types (gait params, etc.)
"""

import torch
from typing import Dict, List, Optional, Union


class Commander:
    """Command generator for locomotion control."""

    def __init__(
        self,
        config: Dict,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            config: Commands configuration dict with keys:
                - names: List of command names (e.g., ["vx", "vy", "wz"])
                - ranges: Dict mapping command names to [min, max] ranges
                - scales: Dict mapping command names to scale factors
                - defaults: Dict mapping command names to default values
            device: Torch device
        """
        self.cfg = config
        self.device = device

        # Command names
        self.cmd_names: List[str] = self.cfg.get(
            "names", ["vx", "vy", "wz"]
        )
        self.num_commands = len(self.cmd_names)

        # Command ranges for clipping
        default_ranges = {
            "vx": [-1.5, 1.5],
            "vy": [-1.0, 1.0],
            "wz": [-1.5, 1.5],
        }
        self.cmd_ranges = self.cfg.get("ranges", default_ranges)

        # Command scales (applied before output)
        default_scales = {name: 1.0 for name in self.cmd_names}
        self.cmd_scales = torch.tensor(
            [self.cfg.get("scales", default_scales).get(name, 1.0) for name in self.cmd_names],
            dtype=torch.float32,
            device=device,
        )

        # Initialize commands to defaults
        default_values = self.cfg.get("defaults", {})
        self.commands = torch.tensor(
            [default_values.get(name, 0.0) for name in self.cmd_names],
            dtype=torch.float32,
            device=device,
        )

    def set_cmd(self, name: str, value: float):
        """Set a single command value.

        Args:
            name: Command name (e.g., "vx")
            value: Command value (will be clipped to range)
        """
        if name not in self.cmd_names:
            raise ValueError(f"Unknown command '{name}'. Available: {self.cmd_names}")

        idx = self.cmd_names.index(name)

        # Clip to range
        if name in self.cmd_ranges:
            min_val, max_val = self.cmd_ranges[name]
            value = max(min_val, min(max_val, value))

        self.commands[idx] = value

    def set_cmds(self, cmds_dict: Dict[str, float]):
        """Set multiple commands at once.

        Args:
            cmds_dict: Dict mapping command names to values
        """
        for name, value in cmds_dict.items():
            self.set_cmd(name, value)

    def get_cmd(
        self,
        names: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """Get command values.

        Args:
            names: Command name(s) to get. If None, returns all commands.

        Returns:
            Command tensor (with scaling applied)
        """
        if names is None:
            return self.commands * self.cmd_scales

        if isinstance(names, str):
            idx = self.cmd_names.index(names)
            return self.commands[idx] * self.cmd_scales[idx]

        indices = [self.cmd_names.index(n) for n in names]
        return self.commands[indices] * self.cmd_scales[indices]

    def get_cmd_unscaled(
        self,
        names: Optional[Union[str, List[str]]] = None
    ) -> torch.Tensor:
        """Get raw command values without scaling.

        Args:
            names: Command name(s) to get. If None, returns all commands.

        Returns:
            Command tensor (without scaling)
        """
        if names is None:
            return self.commands.clone()

        if isinstance(names, str):
            idx = self.cmd_names.index(names)
            return self.commands[idx].clone()

        indices = [self.cmd_names.index(n) for n in names]
        return self.commands[indices].clone()

    def reset(self):
        """Reset commands to defaults."""
        default_values = self.cfg.get("defaults", {})
        for i, name in enumerate(self.cmd_names):
            self.commands[i] = default_values.get(name, 0.0)

    @property
    def dim(self) -> int:
        """Number of commands."""
        return self.num_commands
