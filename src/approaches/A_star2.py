import sys
import os
import time
import numpy as np
from heapq import heappush, heappop

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from minigrid.core.constants import OBJECT_TO_IDX
from SAREnv import SAREnv

# Map encoding
UNKNOWN = 0
EMPTY   = 1
WALL    = 2
LAVA    = 3
EXIT    = 4
PERSON  = 5

class AStarBaselineAgent:
    """
    Agent using A* to rescue all people and carry them to exits.
    Assumes full knowledge of the environment from the start.
    """

    def __init__(self, env):
        self.env = env
        self.width = env.width
        self.height = env.height

        # Build full map once
        self.full_map = np.zeros((self.width, self.height), dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                cell = self.env.grid.get(x, y)
                if cell is None:
                    self.full_map[x, y] = EMPTY
                elif cell.type == "wall":
                    self.full_map[x, y] = WALL
                elif cell.type == "lava":
                    self.full_map[x, y] = LAVA
                elif cell.type == "ball":
                    self.full_map[x, y] = PERSON
                elif cell.type == "exit":
                    self.full_map[x, y] = EXIT
                else:
                    self.full_map[x, y] = EMPTY

        # Cache positions
        self.people_positions = [tuple(int(v) for v in pos)
                                 for pos in np.argwhere(self.full_map == PERSON)]
        self.exit_positions = [tuple(e) for e in self.env.exit_positions]

        # Direction vectors: Right, Down, Left, Up
        self.DIR_TO_VEC = [(1,0),(0,1),(-1,0),(0,-1)]

    # ----------------------------
    # Manhattan A* pathfinding
    # ----------------------------
    def astar(self, start, goal):
        def h(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        open_set = []
        heappush(open_set, (h(start, goal), 0, start))
        came_from = {start: None}
        gscore = {start: 0}

        while open_set:
            _, cost, current = heappop(open_set)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return list(reversed(path))

            cx, cy = current
            for dx, dy in self.DIR_TO_VEC:
                nx, ny = cx+dx, cy+dy
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.full_map[nx, ny] in (WALL, LAVA):
                    continue
                new_cost = cost + 1
                if (nx,ny) not in gscore or new_cost < gscore[(nx,ny)]:
                    gscore[(nx,ny)] = new_cost
                    priority = new_cost + h((nx,ny), goal)
                    heappush(open_set,(priority,new_cost,(nx,ny)))
                    came_from[(nx,ny)] = current
        return None

    # ----------------------------
    # Get next action
    # ----------------------------
    def get_action(self, agent_pos, agent_dir):
        # Phase 1: not carrying a person
        if self.env.carrying is None:
            fx, fy = self.env.front_pos
            front_cell = self.env.grid.get(fx, fy)
            if front_cell is not None and front_cell.type == "ball":
                # Remove from known positions
                if (fx, fy) in self.people_positions:
                    self.people_positions.remove((fx, fy))
                return self.env.actions.pickup

            if not self.people_positions:
                print("No people left to rescue. Agent done.")
                time.sleep(1)
                return None

            # Closest person
            distances = [np.sum((np.array(agent_pos)-np.array(p))**2)
                         for p in self.people_positions]
            goal = self.people_positions[np.argmin(distances)]

        # Phase 2: carrying a person
        else:
            distances = [np.sum((np.array(agent_pos)-np.array(e))**2)
                         for e in self.exit_positions]
            goal = self.exit_positions[np.argmin(distances)]

            if tuple(agent_pos) in self.exit_positions:
                return self.env.actions.drop

        path = self.astar(tuple(agent_pos), goal)
        if not path or len(path)<2:
            print(f"No path from {tuple(agent_pos)} to {goal}. Agent paused.")
            time.sleep(1)
            return None

        next_cell = path[1]
        dx, dy = next_cell[0]-agent_pos[0], next_cell[1]-agent_pos[1]

        if dx==1: desired_dir=0
        elif dy==1: desired_dir=1
        elif dx==-1: desired_dir=2
        elif dy==-1: desired_dir=3
        else: desired_dir=agent_dir

        if desired_dir==agent_dir:
            return self.env.actions.forward
        elif (desired_dir-agent_dir)%4==1:
            return self.env.actions.right
        else:
            return self.env.actions.left


# ----------------------------
# Demo runner
# ----------------------------
def run_astar_baseline():
    from SAREnv import SAREnv

    env = SAREnv(
        room_size=5,
        num_rows=2,
        num_cols=2,
        num_people=2,
        num_exits=2,
        num_collapsed_floors=3,
        agent_view_size=3,
        render_mode="human"
    )

    obs, _ = env.reset()
    agent = AStarBaselineAgent(env)

    for step in range(500):
        env.render()
        action = agent.get_action(env.agent_pos, env.agent_dir)
        if action is None:
            print("Agent is temporarily paused.")
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            print("Episode complete.")
            break

    return env


if __name__=="__main__":
    env = run_astar_baseline()
