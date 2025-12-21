"""Train PPO to pick frontiers using FrontierSelectionEnv with a custom CNN+MLP."""
from __future__ import annotations

import argparse
from collections import deque
from typing import Callable, Dict


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EventCallback

from SAREnv import SAREnv
from ppo_frontier_wrapper import FrontierSelectionEnv


class FrontierFeaturesExtractor(BaseFeaturesExtractor):
    """Custom extractor combining knowledge grid (CNN) and frontier features (MLP)."""

    def __init__(self, observation_space: gym.spaces.Dict, frontier_limit: int):
        super().__init__(observation_space, features_dim=1)  # will set real dim below
        self.frontier_limit = frontier_limit

        k_space = observation_space["knowledge"]
        self.grid_shape = k_space.shape  # (W, H)
        self.num_classes = 13  # states 0-12
        w, h = self.grid_shape

        # CNN over one-hot knowledge grid (channels=13)
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_classes, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out_dim = 64 * w * h

        # Frontier + agent MLP
        # Per-frontier features: coords(2) + dist_to_agent(1) + dist_exit(1)
        # + dist_person(1) + local cell onehot(13) = 18
        self.per_frontier_feats = 18
        frontier_dim = frontier_limit * self.per_frontier_feats
        frontier_mask_dim = frontier_limit  # valid mask
        agent_misc_dim = 2 + 4 + 1  # pos(2) + dir one-hot(4) + rescued(1)
        mlp_in = frontier_dim + frontier_mask_dim + agent_misc_dim

        self.mlp = nn.Sequential(
            nn.Linear(conv_out_dim + mlp_in, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self._features_dim = 256

        # Precompute scaling constants
        self.max_coord = float(max(w, h))
        self.people_high = float(observation_space["people_rescued"].high[0])

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Knowledge grid -> one-hot channels (B, C=13, W, H)
        k_int = observations["knowledge"].long()  # (B, W, H)
        k_onehot = F.one_hot(k_int, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        conv_feats = self.conv(k_onehot)
        conv_flat = torch.flatten(conv_feats, start_dim=1)

        # Frontier features
        frontiers = observations["frontiers"].float()  # (B, F, 2)
        mask = (frontiers[..., 0] >= 0).float()  # (B, F)
        frontiers_clamped = torch.clamp(frontiers, min=0.0)
        frontiers_int = torch.clamp(frontiers.long(), min=0)

        # Distances to agent
        agent_pos = observations["agent_pos"].float()  # (B,2)
        dist_agent = (
            torch.abs(frontiers_clamped[..., 0] - agent_pos[:, None, 0])
            + torch.abs(frontiers_clamped[..., 1] - agent_pos[:, None, 1])
        ) / self.max_coord

        # Distances to nearest exit / person from knowledge grid
        batch_size, frontier_count = frontiers.shape[0], frontiers.shape[1]
        w, h = self.grid_shape

        def min_manhattan_to(k_mask: torch.Tensor) -> torch.Tensor:
            # k_mask: (B,W,H) bool
            dists = torch.full((batch_size, frontier_count), fill_value=self.max_coord * 2, device=frontiers.device)
            for b in range(batch_size):
                coords = k_mask[b].nonzero(as_tuple=False)
                if coords.numel() == 0:
                    continue
                fx = frontiers_clamped[b, :, 0].unsqueeze(1)
                fy = frontiers_clamped[b, :, 1].unsqueeze(1)
                dist = torch.abs(fx - coords[:, 0]) + torch.abs(fy - coords[:, 1])
                dists[b] = dist.min(dim=1).values
            return dists / self.max_coord

        dist_exit = min_manhattan_to(k_onehot[:, 4] > 0)  # EXIT idx
        dist_person = min_manhattan_to(k_onehot[:, 5] > 0)  # PERSON idx

        # Local cell type at frontier (one-hot 13)
        idx_flat = frontiers_int[..., 0] * h + frontiers_int[..., 1]
        k_flat = k_int.view(batch_size, -1)
        cell_vals = torch.zeros((batch_size, frontier_count), device=frontiers.device, dtype=torch.long)
        in_bounds = (frontiers_int[..., 0] < w) & (frontiers_int[..., 1] < h)
        safe_idx = torch.where(in_bounds, idx_flat, torch.zeros_like(idx_flat))

        # Fill cell_vals per batch to avoid shape/broadcast issues
        for b in range(batch_size):
            mask_b = in_bounds[b]
            if mask_b.any():
                cell_vals[b, mask_b] = k_flat[b, safe_idx[b, mask_b]]
        cell_onehot = F.one_hot(cell_vals, num_classes=self.num_classes).float()

        # Assemble per-frontier features
        coords_norm = (frontiers_clamped / self.max_coord) * mask.unsqueeze(-1)
        dist_agent = dist_agent * mask
        dist_exit = dist_exit * mask
        dist_person = dist_person * mask
        cell_onehot = cell_onehot * mask.unsqueeze(-1)

        frontier_feats = torch.cat(
            [
                coords_norm,
                dist_agent.unsqueeze(-1),
                dist_exit.unsqueeze(-1),
                dist_person.unsqueeze(-1),
                cell_onehot,
            ],
            dim=-1,
        )

        frontier_flat = frontier_feats.reshape(frontier_feats.shape[0], -1)
        frontier_mask_flat = mask.reshape(mask.shape[0], -1)

        # Agent + status
        agent_pos = agent_pos / self.max_coord
        agent_dir = observations["agent_dir"].long().squeeze(-1)
        agent_dir_onehot = F.one_hot(agent_dir, num_classes=4).float()
        people = observations["people_rescued"].float() / max(self.people_high, 1.0)

        misc = torch.cat([agent_pos, agent_dir_onehot, people], dim=1)

        fused = torch.cat([conv_flat, frontier_flat, frontier_mask_flat, misc], dim=1)
        return self.mlp(fused)


class ProgressCallback(EventCallback):
    """Lightweight console progress printer."""

    def __init__(self, print_freq_steps: int = 5000):
        super().__init__()
        self.print_freq_steps = print_freq_steps
        self.last_print = 0
        self.last_ep_reward = None
        self.last_ep_len = None
        self.recent_eps = deque(maxlen=100)
        self.ep_since_print = 0

    def _on_step(self) -> bool:
        # Capture recent episode stats if present
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.last_ep_reward = ep.get("r")
                self.last_ep_len = ep.get("l")
                self.recent_eps.append((self.last_ep_reward, self.last_ep_len))
                self.ep_since_print += 1

        if self.model.num_timesteps - self.last_print >= self.print_freq_steps:
            msg = f"[progress] steps={self.model.num_timesteps:,}"
            if self.last_ep_reward is not None:
                msg += f" | last_ep_reward={self.last_ep_reward:.2f}"
            if self.last_ep_len is not None:
                msg += f" | last_ep_len={self.last_ep_len}"
            if self.recent_eps:
                mean_r = sum(r for r, _ in self.recent_eps) / len(self.recent_eps)
                mean_l = sum(l for _, l in self.recent_eps) / len(self.recent_eps)
                msg += f" | mean_r(last {len(self.recent_eps)} eps)={mean_r:.2f}"
                msg += f" | mean_l={mean_l:.1f}"
            msg += f" | eps_since_print={self.ep_since_print}"
            print(msg)
            self.last_print = self.model.num_timesteps
            self.ep_since_print = 0
        return True


def make_env(frontier_limit: int) -> Callable[[], gym.Env]:
    def _init():
        base = SAREnv(
            room_size=6,
            num_rows=3,
            num_cols=3,
            num_people=3,
            num_exits=2,
            num_collapsed_floors=8,
            agent_view_size=3,
            render_mode=None,
        )
        return FrontierSelectionEnv(base, frontier_limit=frontier_limit)

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on frontier selection")
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--frontier-limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--print-freq", type=int, default=5000, help="Steps between console progress prints")
    parser.add_argument("--save-path", type=str, default="ppo_frontier.zip")
    args = parser.parse_args()

    vec_env = make_vec_env(make_env(args.frontier_limit), n_envs=args.num_envs, seed=args.seed)

    policy_kwargs = dict(
        features_extractor_class=FrontierFeaturesExtractor,
        features_extractor_kwargs=dict(frontier_limit=args.frontier_limit),
    )

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=args.lr,
        n_steps=1024,
        batch_size=256,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )

    callback = ProgressCallback(print_freq_steps=args.print_freq)
    model.learn(total_timesteps=args.total_steps, callback=callback)
    model.save(args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
