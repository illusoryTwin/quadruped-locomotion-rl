"""
Observation processing with history stacking support.

Handles:
- Observation scaling and clipping
- History buffer management
- Observation concatenation in configured order
"""

import torch
from typing import Dict, List, Optional, Union


class Observation:
    """Observation processor with history stacking support."""

    def __init__(self, config: Dict, device: torch.device = torch.device("cpu")):
        """
        Args:
            config: Observation configuration dict with keys:
                - order: List of observation term names
                - dims: Dict mapping term names to dimensions
                - scale: Dict mapping term names to scale factors (default 1.0)
                - clip: Dict mapping term names to clip bounds (default inf)
                - num_obs_hist: Number of observation history frames (default 1)
                - hist_by_term: If True, stack history per-term; if False, global buffer
            device: Torch device for tensors
        """
        self.device = device
        self.cfg = config

        # Observation term order
        self.obs_names: List[str] = self.cfg["order"]

        # Parse per-term attributes
        self.dims = self._parse_attr("dims", default=None)
        self.scale = self._parse_attr("scale", default=1.0)
        self.clip = self._parse_attr("clip", default=float("inf"))

        # Ensure clip is [min, max] format
        for k in self.clip.keys():
            if not isinstance(self.clip[k], (list, tuple)):
                self.clip[k] = [-self.clip[k], self.clip[k]]

        # History settings
        self.num_obs_hist = self._parse_attr("num_obs_hist", default=1)
        self.obs_buf_size = max(self.num_obs_hist.values())
        self.hist_by_term = self.cfg.get("hist_by_term", False)

        # Compute total observation dimension
        self.obs_dim = sum(self.dims[name] for name in self.obs_names)
        self.obs_dim_with_hist = self._compute_obs_dim_with_hist()

        # Observation buffer
        self.obs_buf: Optional[torch.Tensor] = None

        self.reset()

    def _parse_attr(self, key: str, default=None) -> Dict:
        """Parse attribute from config, expanding scalars to per-term dict."""
        attr = self.cfg.get(key, default)
        if attr is None:
            return {name: default for name in self.obs_names}
        if isinstance(attr, dict):
            # Fill missing keys with default
            return {name: attr.get(name, default) for name in self.obs_names}
        else:
            # Scalar value - apply to all terms
            return {name: attr for name in self.obs_names}

    def _compute_obs_dim_with_hist(self) -> int:
        """Compute total observation dimension including history."""
        if self.hist_by_term:
            # Each term has its own history length
            return sum(
                self.dims[name] * self.num_obs_hist[name]
                for name in self.obs_names
            )
        else:
            # Global history buffer
            return self.obs_dim * self.obs_buf_size

    def process_observation(
        self,
        obs: torch.Tensor,
        name: str,
        inplace: bool = True
    ) -> torch.Tensor:
        """Process single observation term: scale and clip.

        Args:
            obs: Observation tensor
            name: Observation term name
            inplace: If False, clone before modifying

        Returns:
            Processed observation tensor
        """
        if not inplace:
            obs = obs.clone()

        # Apply scaling
        if name in self.scale and self.scale[name] != 1.0:
            obs = obs * self.scale[name]

        # Apply clipping
        if name in self.clip:
            obs = torch.clamp(obs, self.clip[name][0], self.clip[name][1])

        return obs

    def prepare_observations(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Prepare observation vector from observation dict.

        Args:
            obs_dict: Dict mapping observation names to tensors

        Returns:
            Flattened observation tensor with history
        """
        # Process and concatenate current observations
        prep_obs = []
        for name in self.obs_names:
            if name not in obs_dict:
                raise KeyError(
                    f"Missing observation '{name}'. "
                    f"Expected: {self.obs_names}, got: {list(obs_dict.keys())}"
                )
            obs = obs_dict[name]
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
            obs = self.process_observation(obs.to(self.device), name)
            prep_obs.append(obs.flatten())

        observation = torch.cat(prep_obs, dim=-1)

        # Update history buffer
        if self.obs_buf is None:
            # Initialize buffer by replicating first observation
            self.obs_buf = observation.repeat(self.obs_buf_size)
        else:
            # Shift buffer and append new observation
            self.obs_buf = torch.cat(
                (self.obs_buf[observation.shape[0]:], observation),
                dim=-1
            )

        # Return observation with history
        if self.hist_by_term:
            return self._get_hist_by_term(observation)
        else:
            return self.obs_buf

    def _get_hist_by_term(self, current_obs: torch.Tensor) -> torch.Tensor:
        """Extract history stacked per observation term.

        For each term, extracts the last N frames where N is num_obs_hist[term].
        """
        obs_parts = []
        n_obs = current_obs.shape[0]  # Single frame observation size
        offset_d = 0

        for name in self.obs_names:
            o_d = self.dims[name]  # Dimension of this term
            o_h = self.num_obs_hist[name]  # History length for this term

            term_hist = torch.zeros(o_d * o_h, device=self.device)

            # Take latest o_h frames for this term
            offset_l = n_obs * (self.obs_buf_size - o_h)
            for h in range(o_h):
                start_idx = offset_l + offset_d + h * n_obs
                end_idx = start_idx + o_d
                term_hist[o_d * h: o_d * (h + 1)] = self.obs_buf[start_idx:end_idx]

            obs_parts.append(term_hist)
            offset_d += o_d

        return torch.cat(obs_parts, dim=-1)

    def reset(self):
        """Reset observation buffer."""
        self.obs_buf = None

    @property
    def output_dim(self) -> int:
        """Total output dimension including history."""
        return self.obs_dim_with_hist
