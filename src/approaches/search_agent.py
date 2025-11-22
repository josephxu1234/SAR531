import sys
import os

# Add the parent directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from SAREnv import SAREnv
from collections import deque

# Map States
UNKNOWN = 0
EMPTY = 1
WALL = 2
LAVA = 3
EXIT = 4
PERSON = 5

class FrontierAgent:
    def __init__(self, env: SAREnv):
        self.env = env
        self.width = env.width
        self.height = env.height
        
        # Internal knowledge grid
        # Default everything is UNKNOWN
        self.knowledge_grid = np.zeros((self.width, self.height), dtype=int)
        
        # Set of current frontier cells (tuples)
        self.frontiers = set()
        
        # Directions mapping for movement logic
        self.DIR_TO_VEC = [
            (1, 0),  # 0: Right
            (0, 1),  # 1: Down
            (-1, 0), # 2: Left
            (0, -1)  # 3: Up
        ]
        
        # For stuck detection purposes
        self.last_pos = None
        self.last_action = None
        self.stuck_count = 0

    def _has_unknown_neighbor(self, x, y):
        """Check if cell (x, y) has any UNKNOWN neighbors."""

        # Iterate through neighbors
        for dx, dy in self.DIR_TO_VEC:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.knowledge_grid[nx, ny] == UNKNOWN:
                    return True
        return False

    def _update_frontier_set(self, x, y):
        """
        Called when cell (x, y) changes from UNKNOWN to KNOWN.
        Updates the frontier set for (x, y) and its neighbors.
        """

        # If it's walkable and has unknown neighbors, it's a frontier
        # Idea: want to go to squares that are walkable and allow us to see unknowns
        is_walkable = self.knowledge_grid[x, y] in [EMPTY, EXIT]
        if is_walkable and self._has_unknown_neighbor(x, y):
            self.frontiers.add((x, y))
        
        # 2. Handle neighbors
        # Neighbors that were frontiers might stop being frontiers
        # because (x, y) is no longer unknown.
        for dx, dy in self.DIR_TO_VEC:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) in self.frontiers:
                    # If this neighbor no longer has ANY unknown neighbors, remove it
                    if not self._has_unknown_neighbor(nx, ny):
                        self.frontiers.remove((nx, ny))

    def update_map(self, obs, agent_pos, agent_dir):
        """
        Projects the local agent view (obs) onto the global knowledge grid.
        """

        # Convention: (0, 0) is top-left corner
        # j = 0 [ . . .]
        # j = 1 [ . . .]
        # j = 2 [ . . .]
        # i:      0 1 2

        # Image representation: (x, y, world_obj data)
        # world obj data: (type_idx, color_idx, state)
        view = obs['image'] # shape (W, H, 3)
        view_cols = view.shape[0]
        view_rows = view.shape[1]
        
        # Iterate over the local view
        for i in range(view_cols): # columns
            for j in range(view_rows): # rows
                obj_type_idx = view[i, j, 0]
                
                # Transform local (i, j) to global (gx, gy)
                # camera frame: Agent is at center bottom

                # local x, local y = coordinates in agent's frame of reference
                # lo_x: how far right from agent? Right (+), Left(-)
                # lo_y: how far forward from agent? Forward(-), Back (+); shouldn't have backward
                lo_x = i - (view_cols // 2)
                lo_y = j - (view_rows - 1)
                
                if agent_dir == 0: # Facing Right
                    gx = agent_pos[0] - lo_y
                    gy = agent_pos[1] + lo_x
                elif agent_dir == 1: # Facing Down
                    gx = agent_pos[0] - lo_x
                    gy = agent_pos[1] - lo_y
                elif agent_dir == 2: # Facing Left
                    gx = agent_pos[0] + lo_y
                    gy = agent_pos[1] - lo_x
                elif agent_dir == 3: # Facing Up
                    gx = agent_pos[0] + lo_x
                    gy = agent_pos[1] + lo_y
                
                # Check bounds
                if 0 <= gx < self.width and 0 <= gy < self.height:
                    old_state = self.knowledge_grid[gx, gy]
                    
                    # Determine new state
                    new_state = UNKNOWN # Default to UNKNOWN to be safe
                    
                    if obj_type_idx == OBJECT_TO_IDX['wall']:
                        new_state = WALL
                    elif obj_type_idx == OBJECT_TO_IDX['lava']:
                        new_state = LAVA
                    elif obj_type_idx == OBJECT_TO_IDX['ball']:
                        new_state = PERSON
                    elif obj_type_idx == OBJECT_TO_IDX['goal'] or obj_type_idx == OBJECT_TO_IDX['door']:
                        new_state = EXIT # Treat doors as exits/empty for nav
                    elif obj_type_idx == 1: # Empty
                        new_state = EMPTY
                    elif obj_type_idx == 0: # Unseen
                        new_state = UNKNOWN
                    else:
                        # Unknown object type - treat as UNKNOWN or WALL to be safe?
                        # Treating as EMPTY caused bugs. Let's keep it UNKNOWN.
                        new_state = UNKNOWN
                    
                    # Only update if we learned something new
                    # We allow overwriting UNKNOWN with anything
                    # We allow overwriting anything with WALL/LAVA/PERSON (more specific)
                    # We DO NOT allow overwriting with UNKNOWN (regression)
                    if new_state != UNKNOWN:
                        if old_state == UNKNOWN or old_state != new_state:
                            self.knowledge_grid[gx, gy] = new_state
                            self._update_frontier_set(gx, gy)
                    # updated knowledge, now see wall/lava
                    if old_state in [EMPTY, UNKNOWN] and new_state in [WALL, LAVA]:
                        self.knowledge_grid[gx, gy] = new_state
                        if (gx, gy) in self.frontiers:
                            self.frontiers.remove((gx, gy))

    def mark_wall_in_front(self, agent_pos, agent_dir):
        """Marks the cell directly in front of the agent as a WALL."""
        dx, dy = self.DIR_TO_VEC[agent_dir]
        fx, fy = agent_pos[0] + dx, agent_pos[1] + dy
        
        if 0 <= fx < self.width and 0 <= fy < self.height:
            print(f"Marking blocked cell ({fx}, {fy}) as WALL.")
            self.knowledge_grid[fx, fy] = WALL
            # If it was a frontier, remove it
            if (fx, fy) in self.frontiers:
                self.frontiers.remove((fx, fy))
            # Update neighbors' frontier status
            self._update_frontier_set(fx, fy)

    def find_path_bfs(self, start, goal):
        """Optimized BFS to find path on known map using deque and backtracking."""
        queue = deque([start])
        # Dictionary to store parent pointers
        # Also serves as visited set
        came_from = {start: None} 
        
        while queue:
            curr = queue.popleft()
            
            if curr == goal:
                break # Found goal, stop searching
            
            curr_x, curr_y = curr
            
            # iterate neighbors
            for dx, dy in self.DIR_TO_VEC:
                nx, ny = curr_x + dx, curr_y + dy
                
                # neighbor is in bounds
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell_type = self.knowledge_grid[nx, ny]
                    is_walkable = cell_type in [EMPTY, EXIT]
                    
                    if is_walkable and (nx, ny) not in came_from:
                        came_from[(nx, ny)] = curr
                        queue.append((nx, ny))
        
        # Reconstruct path if goal was found
        if goal not in came_from:
            return None
            
        path = []
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        
        path.reverse() # Path is built backwards, so reverse it
        return path

    def get_action(self, agent_pos, agent_dir):
        """Decides the next action."""
        
        # Check if stuck
        if self.last_pos == agent_pos and self.last_action == self.env.actions.forward:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            
        self.last_pos = agent_pos
        
        if self.stuck_count > 0:
            # We are stuck
            # Mark the cell in front as blocked (WALL) so we don't try again.
            if self.last_action == self.env.actions.forward:
                self.mark_wall_in_front(agent_pos, agent_dir)
            
            # Try turning right to unstick.
            print("Stuck! Taking evasive action.")
            self.last_action = self.env.actions.right 
            return self.last_action
        
        # Check if we found the person
        people_locs = np.argwhere(self.knowledge_grid == PERSON)
        if len(people_locs) > 0:
            print("Person found at:", people_locs[0])
            # TODO: return None to stop searching
            return None
            
        # Get Frontiers from cached set
        if not self.frontiers:
            print("Map fully explored, person not found.")
            return None
            
        # Find nearest frontier
        best_path = None
        min_dist = float('inf')
        
        # Sort frontiers by Manhattan distance to prioritize closer ones
        sorted_frontiers = sorted(list(self.frontiers), 
                                key=lambda f: abs(f[0]-agent_pos[0]) + abs(f[1]-agent_pos[1]))
        
        # Check the closest few frontiers (optimization)
        for f in sorted_frontiers: 
            path = self.find_path_bfs(agent_pos, f)
            if path:
                dist = len(path)
                if dist < min_dist:
                    min_dist = dist
                    best_path = path
        
        if best_path is None:
            print("No reachable frontiers remaining.")
            return None
        
        # check len(best_path) >= 2 bc len 1 path means agent is already at frontier
        if len(best_path) < 2:
            # If we can't reach the closest frontiers, try turning right instead of forward
            # This prevents blindly running into walls if forward is blocked
            self.last_action = self.env.actions.right
            return self.last_action
            
        # Determine action to move along path
        next_cell = best_path[1]
        if self.env.grid.get(next_cell[0], next_cell[1]) is not None:
            print("Next cell has object:", self.env.grid.get(next_cell[0], next_cell[1]).type)
        else:
            print("Next cell is empty.")
        
        dx = next_cell[0] - agent_pos[0]
        dy = next_cell[1] - agent_pos[1]
        
        desired_dir = -1
        if dx == 1: desired_dir = 0
        elif dy == 1: desired_dir = 1
        elif dx == -1: desired_dir = 2
        elif dy == -1: desired_dir = 3
        
        if desired_dir == agent_dir:
            action = self.env.actions.forward
            print("Moving forward to", next_cell)
        elif (desired_dir - agent_dir) % 4 == 1:
            action = self.env.actions.right
            print("Turning right towards", next_cell)
        else:
            action = self.env.actions.left
            print("Turning left towards", next_cell)
            
        self.last_action = action

        return action

def run_search_demo():
    # Setup specific scenario: 1 Person, 1 Exit
    env = SAREnv(
        room_size=5, 
        num_rows=2, 
        num_cols=2, 
        num_people=1, 
        num_exits=1,
        num_collapsed_floors=6,
        agent_view_size=3, 
        render_mode="human"
    )
    
    obs, _ = env.reset()
    agent = FrontierAgent(env)
    
    # Initial map update
    agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    for step in range(1000):
        env.render()        
        action = agent.get_action(env.agent_pos, env.agent_dir)
                
        if action is None:
            print("Search completed or failed!")
            break
            
        obs, reward, term, trunc, info = env.step(action)
        
        agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        if term:
            print("Episode terminated!")
            break

    return agent.knowledge_grid, env

if __name__ == "__main__":
    final_map, env = run_search_demo()
    
    print("Final Internal Map:")
    chars = {UNKNOWN: '?', EMPTY: '.', WALL: '#', LAVA: 'X', EXIT: 'E', PERSON: 'P'}
    for y in range(final_map.shape[1]):
        line = ""
        for x in range(final_map.shape[0]):
            line += chars[final_map[x, y]]
        print(line)

    # Verification Step
    print("Verifying Map Accuracy")
    print('Num cells not unknown:', np.sum(final_map != UNKNOWN))
    
    # Iterate over the entire grid
    correct_cells = 0
    total_known_cells = 0
    
    for x in range(env.width):
        for y in range(env.height):
            agent_val = final_map[x, y]
            
            # Skip UNKNOWN cells (agent didn't see them)
            if agent_val == UNKNOWN:
                continue
            else:
                print("Known cell at ({}, {}): {}".format(x, y, chars[agent_val]))
                
            total_known_cells += 1
            
            # Get ground truth from env
            cell = env.grid.get(x, y)
            
            # Determine ground truth type
            true_val = EMPTY # Default
            if cell is None:
                true_val = EMPTY
            elif cell.type == 'wall':
                true_val = WALL
            elif cell.type == 'lava':
                true_val = LAVA
            elif cell.type == 'ball':
                true_val = PERSON
            elif cell.type == 'door' or cell.type == 'goal':
                true_val = EXIT # We map doors/exits to EXIT

            # Check for match
            # Note: Agent maps Door -> EXIT, Goal -> EXIT. 
            # Agent maps Empty -> EMPTY.
            
            is_correct = False
            if agent_val == true_val:
                is_correct = True
            elif agent_val == EXIT and (cell is not None and (cell.type == 'door' or cell.type == 'goal')):
                 is_correct = True
            elif agent_val == EMPTY and cell is None:
                 is_correct = True
            
            if is_correct:
                correct_cells += 1
            else:
                print(f"Mismatch at ({x}, {y}): Agent thought {chars[agent_val]}, True is {cell.type if cell else 'Empty'}")

    print(f"Verification Complete: {correct_cells}/{total_known_cells} known cells correct.")
    assert correct_cells == total_known_cells, "Agent has incorrect knowledge."
