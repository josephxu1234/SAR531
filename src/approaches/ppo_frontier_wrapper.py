"""
Wrapper environment that reduces the PPO action space to frontier selection.

The wrapped agent only chooses which frontier cell to head toward next. The
wrapper takes care of:
1) Maintaining the knowledge grid via `SearchAgent` so frontiers are always
   up-to-date and non-stale.
2) Translating a chosen frontier into primitive MiniGrid actions (turn/forward)
   and executing one low-level step.
3) Automatically switching to rescue mode (path to person, pickup, deliver to
   exit) when a person is already known, using the existing search/rescue
   logic as guidance.

Observation space
-----------------
Dict containing:
* ``knowledge``: (W,H) int8 grid of known cell states (see search_agent.py
  constants).
* ``agent_pos``: (2,) int16 position of the agent in global coords.
* ``agent_dir``: (1,) int8 direction (0=right,1=down,2=left,3=up).
* ``frontiers``: (F,2) int16 list of frontier coordinates padded with -1,
  sorted by Manhattan distance to the agent. ``F`` is configurable via
  ``frontier_limit``.
* ``people_rescued``: (1,) int16 count from the underlying env.

Action space
------------
Discrete(frontier_limit). Index selects one of the currently cached frontiers
after sorting by distance. If the chosen index is out of range, the wrapper
falls back to the nearest frontier. When no frontiers exist and no people are
known, the episode terminates early with zero reward.

Reward / termination
--------------------
Rewards are the sum of the underlying `SAREnv` rewards for the executed
primitive actions. Termination/truncation mirrors the base environment or
occurs when exploration is exhausted and no persons are known.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gymnasium as gym
import numpy as np

from SAREnv import SAREnv
from rescue_agent import WALKABLE_STATES
from search_agent import EXIT, PERSON, SearchAgent


class FrontierSelectionEnv(gym.Env):
    """Gymnasium wrapper exposing a frontier-selection action space."""

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, env: SAREnv, frontier_limit: int = 64):
        super().__init__()
        self.env = env
        self.frontier_limit = frontier_limit

        # Map keeper keeps knowledge grid fresh and frontiers cached
        self.search_agent = SearchAgent(env)

        # Action: pick a frontier index from the sorted list (padded to limit)
        self.action_space = gym.spaces.Discrete(self.frontier_limit)

        # Observation: structured view of knowledge + agent state + frontiers
        self.observation_space = gym.spaces.Dict(
            {
                "knowledge": gym.spaces.Box(
                    low=0,
                    high=12,
                    shape=(env.width, env.height),
                    dtype=np.int8,
                ),
                "agent_pos": gym.spaces.Box(
                    low=0,
                    high=max(env.width, env.height),
                    shape=(2,),
                    dtype=np.int16,
                ),
                "agent_dir": gym.spaces.Box(
                    low=0,
                    high=3,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "frontiers": gym.spaces.Box(
                    low=-1,
                    high=max(env.width, env.height),
                    shape=(self.frontier_limit, 2),
                    dtype=np.int16,
                ),
                "people_rescued": gym.spaces.Box(
                    low=0,
                    high=env.num_people,
                    shape=(1,),
                    dtype=np.int16,
                ),
            }
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.search_agent = SearchAgent(self.env)
        self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
        return self._build_obs(), info

    def step(self, action: int):
        """
        If a person is already known, run rescue automatically (sequence of
        primitive steps). Otherwise, treat the action as the selected frontier
        index and execute a single primitive action toward that frontier.
        """

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # If already carrying someone, go straight to exit delivery
        if self.env.carrying is not None and getattr(self.env.carrying, "type", "") == "ball":
            delivery_reward, terminated, truncated, info, obs = self._deliver_to_exit()
            total_reward += delivery_reward

            # If delivery failed due to missing/blocked exit, fall back to exploration
            rescue_failed = info.get("rescue_failed")
            if not terminated and not truncated and rescue_failed in {"no_exit_known", "path_blocked"}:
                pass  # continue into exploration logic below
            else:
                self._prune_missing_people()
                return obs, total_reward, terminated, truncated, info

        # Rescue mode when any person is known (but not yet picked up)
        known_people = self.search_agent.get_known_people_locations()
        if known_people:
            rescue_reward, terminated, truncated, info, obs = self._run_rescue(known_people[0])
            total_reward += rescue_reward
            if terminated or truncated:
                self._prune_missing_people()
                return obs, total_reward, terminated, truncated, info
            self._prune_missing_people()
            return self._build_obs(), total_reward, terminated, truncated, info

        # No known person -> frontier-driven exploration
        frontier_list = self._sorted_frontiers()
        if not frontier_list:
            # Nothing left to explore
            terminated = True
            info["reason"] = "no_frontiers"
            return self._build_obs(), total_reward, terminated, truncated, info

        target_idx = int(action)
        if target_idx >= len(frontier_list):
            target_idx = 0  # Fallback to nearest frontier
        target_frontier = frontier_list[target_idx]

        # Plan path to frontier and take one primitive step along it
        path = self._find_path(
            tuple(self.env.agent_pos), target_frontier, allow_person=self.env.carrying is None
        )
        if path is None or len(path) < 2:
            # Can't reach, drop this frontier
            self.search_agent.frontiers.discard(target_frontier)
            info["unreachable_frontier"] = target_frontier
            return self._build_obs(), total_reward, terminated, truncated, info

        # Execute a single primitive action toward next cell (door-aware)
        next_cell = path[1]
        obs, reward, terminated, truncated, info = self._step_toward_cell(next_cell)
        total_reward += reward
        self._prune_missing_people()

        return obs, total_reward, terminated, truncated, info

    def render(self):  # pragma: no cover - passthrough
        return self.env.render()

    def close(self):  # pragma: no cover - passthrough
        return self.env.close()

    # ------------------------------------------------------------------
    # Rescue logic (macro executed internally)
    # ------------------------------------------------------------------
    def _run_rescue(self, person_pos: Tuple[int, int]):
        """Run a full rescue sequence for the first known person."""

        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {"mode": "rescue"}
        obs = self._build_obs()

        # Step 1: get adjacent to the person
        adjacent_targets = self._adjacent_walkable(person_pos)
        if not adjacent_targets:
            info["rescue_failed"] = "no_adjacent_walkable"
            return total_reward, terminated, truncated, info, obs

        # Pick closest adjacent tile
        start = tuple(self.env.agent_pos)
        adj_target = min(
            adjacent_targets,
            key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]),
        )

        reward_path, terminated, truncated, info, obs = self._follow_path(adj_target, info)
        total_reward += reward_path
        if terminated or truncated:
            return total_reward, terminated, truncated, info, obs

        # Step 2: pickup person (ensure facing)
        reward_pickup, terminated, truncated, info, obs = self._pickup_at(person_pos, info)
        total_reward += reward_pickup
        if terminated or truncated:
            return total_reward, terminated, truncated, info, obs

        # Step 3: navigate to nearest exit (may continue delivering later if interrupted)
        reward_exit, terminated, truncated, info, obs = self._deliver_to_exit(info)
        total_reward += reward_exit
        return total_reward, terminated, truncated, info, obs

    def _deliver_to_exit(self, info: Dict[str, Any] | None = None):
        """Navigate to nearest known exit while carrying a person."""
        if info is None:
            info = {"mode": "deliver"}

        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self._build_obs()

        exit_pos = self._nearest_exit()
        if exit_pos is None:
            info["rescue_failed"] = "no_exit_known"
            return total_reward, terminated, truncated, info, obs

        reward_exit, terminated, truncated, info, obs = self._follow_path(exit_pos, info)
        total_reward += reward_exit

        # If we delivered successfully, clear carrying and memory cleanup
        if self.env.carrying is None:
            info["delivered"] = True
            # Make sure any stale person markers are removed
            self._prune_missing_people()

        return total_reward, terminated, truncated, info, obs

    def _follow_path(self, goal: Tuple[int, int], info: Dict[str, Any]):
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self._build_obs()

        path = self._find_path(tuple(self.env.agent_pos), goal, allow_person=self.env.carrying is None)
        if path is None or len(path) < 2:
            info["rescue_failed"] = "path_blocked"
            return total_reward, terminated, truncated, info, obs

        for next_cell in path[1:]:
            obs, reward, terminated, truncated, step_info = self._step_toward_cell(next_cell)
            total_reward += reward
            info.update(step_info)

            if terminated or truncated:
                return total_reward, terminated, truncated, info, obs

        return total_reward, terminated, truncated, info, obs

    def _pickup_at(self, person_pos: Tuple[int, int], info: Dict[str, Any]):
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = self._build_obs()

        # Find direction to person if adjacent
        dx = person_pos[0] - self.env.agent_pos[0]
        dy = person_pos[1] - self.env.agent_pos[1]
        desired_dir = None
        if abs(dx) + abs(dy) != 1:
            info["rescue_failed"] = "not_adjacent_for_pickup"
            return total_reward, terminated, truncated, info, obs
        if dx == 1:
            desired_dir = 0
        elif dy == 1:
            desired_dir = 1
        elif dx == -1:
            desired_dir = 2
        elif dy == -1:
            desired_dir = 3

        # Turn toward the person if needed
        if desired_dir is not None and desired_dir != self.env.agent_dir:
            turn_action = self._turn_action(desired_dir)
            obs, reward, terminated, truncated, step_info = self.env.step(turn_action)
            total_reward += reward
            info.update(step_info)
            self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
            if terminated or truncated:
                return total_reward, terminated, truncated, info, obs

        # Pickup
        obs, reward, terminated, truncated, step_info = self.env.step(self.env.actions.pickup)
        total_reward += reward
        info.update(step_info)
        self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)

        # After pickup, remove from knowledge (and prune any stale marks)
        if self.env.carrying is not None:
            self.search_agent.remove_person_from_memory(person_pos)
            self._prune_missing_people()

        return total_reward, terminated, truncated, info, obs

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sorted_frontiers(self) -> List[Tuple[int, int]]:
        agent_pos = tuple(self.env.agent_pos)
        if self.env.carrying is not None:
            frontier_cells = [
                f for f in self.search_agent.frontiers if self.search_agent.knowledge_grid[f] != PERSON
            ]
        else:
            frontier_cells = list(self.search_agent.frontiers)

        return sorted(
            frontier_cells,
            key=lambda f: abs(f[0] - agent_pos[0]) + abs(f[1] - agent_pos[1]),
        )

    def _build_obs(self):
        frontier_list = self._sorted_frontiers()
        padded_frontiers = np.full((self.frontier_limit, 2), -1, dtype=np.int16)
        for i, (fx, fy) in enumerate(frontier_list[: self.frontier_limit]):
            padded_frontiers[i] = (fx, fy)

        return {
            "knowledge": self.search_agent.knowledge_grid.astype(np.int8),
            "agent_pos": np.array(self.env.agent_pos, dtype=np.int16),
            "agent_dir": np.array([self.env.agent_dir], dtype=np.int8),
            "frontiers": padded_frontiers,
            "people_rescued": np.array([self.env.people_rescued], dtype=np.int16),
        }

    def _action_toward(self, next_cell: Tuple[int, int]):
        curr = tuple(self.env.agent_pos)
        dx = next_cell[0] - curr[0]
        dy = next_cell[1] - curr[1]

        desired_dir = self._direction_from_delta(dx, dy)

        if desired_dir == self.env.agent_dir:
            return self.env.actions.forward
        return self._turn_action(desired_dir)

    def _step_toward_cell(self, next_cell: Tuple[int, int]):
        """Turn as needed, open doors if closed, then step forward; returns wrapped obs."""
        desired_dir = self._direction_to(next_cell)

        # Turn until facing desired_dir
        while desired_dir != self.env.agent_dir:
            turn_action = self._turn_action(desired_dir)
            obs_raw, reward, terminated, truncated, info = self.env.step(turn_action)
            self.search_agent.update_map(obs_raw, self.env.agent_pos, self.env.agent_dir)
            if terminated or truncated:
                return self._build_obs(), reward, terminated, truncated, info
            desired_dir = self._direction_to(next_cell)

        # If door ahead and closed, toggle to open
        cell_obj = self.env.grid.get(*next_cell)
        if cell_obj is not None and getattr(cell_obj, "type", None) == "door":
            if not getattr(cell_obj, "is_open", True):
                obs_raw, reward, terminated, truncated, info = self.env.step(self.env.actions.toggle)
                self.search_agent.update_map(obs_raw, self.env.agent_pos, self.env.agent_dir)
                if terminated or truncated:
                    return self._build_obs(), reward, terminated, truncated, info
                # Refresh desired_dir in case orientation changed (it shouldn't)
                desired_dir = self._direction_to(next_cell)
                if desired_dir != self.env.agent_dir:
                    turn_action = self._turn_action(desired_dir)
                    obs_raw, add_r, terminated, truncated, info = self.env.step(turn_action)
                    reward += add_r
                    self.search_agent.update_map(obs_raw, self.env.agent_pos, self.env.agent_dir)
                    if terminated or truncated:
                        return self._build_obs(), reward, terminated, truncated, info

        # Move forward
        obs_raw, reward, terminated, truncated, info = self.env.step(self.env.actions.forward)
        self.search_agent.update_map(obs_raw, self.env.agent_pos, self.env.agent_dir)
        return self._build_obs(), reward, terminated, truncated, info

    def _direction_to(self, target: Tuple[int, int]) -> int:
        curr = tuple(self.env.agent_pos)
        dx = target[0] - curr[0]
        dy = target[1] - curr[1]
        return self._direction_from_delta(dx, dy)

    def _find_path(self, start: Tuple[int, int], goal: Tuple[int, int], allow_person: bool):
        """BFS that optionally forbids stepping on PERSON cells (used when carrying)."""
        walkable = WALKABLE_STATES.union({PERSON}) if allow_person else WALKABLE_STATES
        queue = [start]
        came_from = {start: None}

        while queue:
            curr = queue.pop(0)
            if curr == goal:
                break

            cx, cy = curr
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.env.width and 0 <= ny < self.env.height):
                    continue
                cell_type = self.search_agent.knowledge_grid[nx, ny]
                if cell_type in walkable and (nx, ny) not in came_from:
                    came_from[(nx, ny)] = curr
                    queue.append((nx, ny))

        if goal not in came_from:
            return None

        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path

    def _prune_missing_people(self):
        """Remove PERSON marks where the underlying grid no longer has a person."""
        person_positions = np.argwhere(self.search_agent.knowledge_grid == PERSON)
        for px, py in person_positions:
            cell = self.env.grid.get(int(px), int(py))
            if cell is None or getattr(cell, "type", None) != "ball":
                self.search_agent.knowledge_grid[px, py] = 1  # EMPTY
                self.search_agent._update_frontier_set(int(px), int(py))

    @staticmethod
    def _direction_from_delta(dx: int, dy: int) -> int:
        if dx == 1:
            return 0
        if dy == 1:
            return 1
        if dx == -1:
            return 2
        if dy == -1:
            return 3
        # Default: keep current direction (should not happen if steps are adjacent)
        return 0

    def _turn_action(self, desired_dir: int):
        # Choose the shorter rotation (left vs right)
        diff = (desired_dir - self.env.agent_dir) % 4
        if diff == 1:
            return self.env.actions.right
        if diff == 3:
            return self.env.actions.left
        # 180 turn: choose right-right for determinism
        return self.env.actions.right

    def _adjacent_walkable(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        adj = []
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.env.width and 0 <= ny < self.env.height:
                if self.search_agent.knowledge_grid[nx, ny] in WALKABLE_STATES:
                    adj.append((nx, ny))
        return adj

    def _nearest_exit(self) -> Tuple[int, int] | None:
        exits = np.argwhere(self.search_agent.knowledge_grid == EXIT)
        if exits.size == 0:
            return None
        agent_pos = tuple(self.env.agent_pos)
        exits_list = [tuple(e) for e in exits]
        return min(
            exits_list,
            key=lambda p: abs(p[0] - agent_pos[0]) + abs(p[1] - agent_pos[1]),
        )
