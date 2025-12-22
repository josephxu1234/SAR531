from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from SAREnv import SAREnv
from ppo_frontier_wrapper import FrontierSelectionEnv

# greedy policy
def pick_nearest_frontier(frontiers: np.ndarray, agent_pos: np.ndarray) -> int | None:
    """Return index of nearest valid frontier; None if none exist."""
    mask = frontiers[:, 0] >= 0
    if not np.any(mask):
        return None

    valid = frontiers[mask]
    distances = np.abs(valid[:, 0] - agent_pos[0]) + np.abs(valid[:, 1] - agent_pos[1])
    nearest_idx_among_valid = int(np.argmin(distances))

    # Convert back to original indices (since mask was applied)
    original_indices = np.nonzero(mask)[0]
    return int(original_indices[nearest_idx_among_valid])


def pick_ppo_frontier(model: PPO, obs: dict) -> int:
    """Use the trained PPO policy to select a frontier index."""
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def run_demo(render: bool = True, model_path: Optional[str] = None, frontier_limit: int = 64):
    base_env = SAREnv(
        room_size=6,
        num_rows=3,
        num_cols=3,
        num_people=3,
        num_exits=2,
        num_collapsed_floors=8,
        agent_view_size=3,
        render_mode="human" if render else None,
    )

    env = FrontierSelectionEnv(base_env, frontier_limit=frontier_limit)
    obs, info = env.reset()
    last_obs = obs

    model: Optional[PPO] = None
    if model_path:
        print(f"Loading PPO model from {model_path}")
        model = PPO.load(model_path)

    total_reward = 0.0
    max_steps = 2000

    for step in range(max_steps):
        if render:
            env.render()

        agent_pos = obs["agent_pos"]
        frontiers = obs["frontiers"]

        greedy_action = pick_nearest_frontier(frontiers, agent_pos)

        if model is not None:
            ppo_action = pick_ppo_frontier(model, obs)
            action = ppo_action
            if greedy_action is not None:
                print(
                    f"[Step {step}] PPO action={ppo_action} | greedy={greedy_action}"
                )
            else:
                print(f"[Step {step}] PPO action={ppo_action} | greedy=None")
        else:
            action = greedy_action
            print(f"[Step {step}] Greedy action={action}")

        if action is None:
            print(f"[Step {step}] No frontiers left; terminating.")
            break

        obs, reward, terminated, truncated, info = env.step(action)
        last_obs = obs
        total_reward += reward

        if terminated or truncated:
            reason = "terminated" if terminated else "truncated"
            print(f"[Step {step}] Episode {reason}. Info: {info}")
            break

    final_obs = last_obs
    people_rescued = final_obs.get("people_rescued")
    rescued_str = (
        f"{people_rescued[0]}/{base_env.num_people}" if people_rescued is not None else "unknown"
    )
    print(f"Finished after {step+1} steps. Total reward: {total_reward:.2f}")
    print(f"People rescued: {rescued_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frontier selection demo (PPO vs greedy)")
    parser.add_argument("--model", type=str, default=None, help="Path to trained PPO model (zip). If omitted, uses greedy only.")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering window")
    parser.add_argument("--frontier-limit", type=int, default=64, help="Frontier list padding/limit")
    args = parser.parse_args()

    run_demo(render=not args.no_render, model_path=args.model, frontier_limit=args.frontier_limit)
