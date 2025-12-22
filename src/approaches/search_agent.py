import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
from SAREnv import SAREnv
from collections import deque

# Map States 
UNKNOWN = 0
EMPTY = 1
WALL = 2
LAVA = 3
EXIT = 4
PERSON = 5
DOOR = 6
# Floor colors (7-12)
FLOOR_RED = 7
FLOOR_GREEN = 8
FLOOR_BLUE = 9
FLOOR_PURPLE = 10
FLOOR_YELLOW = 11
FLOOR_GREY = 12

# which states are walkable for pathfinding
WALKABLE_STATES = {EMPTY, EXIT, PERSON, DOOR, FLOOR_RED, FLOOR_GREEN, FLOOR_BLUE, FLOOR_PURPLE, FLOOR_YELLOW, FLOOR_GREY}

class SearchAgent:
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
        for dx, dy in self.DIR_TO_VEC:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if self.knowledge_grid[nx, ny] == UNKNOWN:
                    return True
        return False

    def _update_frontier_set(self, x, y):
        """
        Updates the frontier set for (x, y) and its neighbors.
        """
        # If it's walkable and has unknown neighbors, it's a frontier
        is_walkable = self.knowledge_grid[x, y] in WALKABLE_STATES
        if is_walkable and self._has_unknown_neighbor(x, y):
            self.frontiers.add((x, y))
        else:
            # Not a frontier anymore, remove if present
            self.frontiers.discard((x, y))
        
        # Handle neighbors - they might stop being frontiers
        for dx, dy in self.DIR_TO_VEC:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if (nx, ny) in self.frontiers:
                    # Re-check if this neighbor is still a frontier
                    if not self._has_unknown_neighbor(nx, ny):
                        self.frontiers.discard((nx, ny))

    def update_map(self, obs, agent_pos, agent_dir):
        """
        Projects the local agent view (obs) onto the global knowledge grid.
        """
        view = obs['image']  # shape (view_width, view_height, 3)
        view_width = view.shape[0]
        view_height = view.shape[1]
        
        # Agent is at bottom-center of view
        agent_view_x = view_width // 2
        agent_view_y = view_height - 1
        
        # Iterate over the local view
        for view_x in range(view_width):
            for view_y in range(view_height):
                obj_type_idx = view[view_x, view_y, 0]
                
                # Calculate relative position to agent in view
                rel_x = view_x - agent_view_x
                rel_y = view_y - agent_view_y
                
                # Transform based on agent direction
                if agent_dir == 0:  # Facing Right
                    gx = agent_pos[0] - rel_y
                    gy = agent_pos[1] + rel_x
                elif agent_dir == 1:  # Facing Down
                    gx = agent_pos[0] - rel_x
                    gy = agent_pos[1] - rel_y
                elif agent_dir == 2:  # Facing Left
                    gx = agent_pos[0] + rel_y
                    gy = agent_pos[1] - rel_x
                elif agent_dir == 3:  # Facing Up
                    gx = agent_pos[0] + rel_x
                    gy = agent_pos[1] + rel_y
                
                # Check bounds
                if not (0 <= gx < self.width and 0 <= gy < self.height):
                    continue
                
                old_state = self.knowledge_grid[gx, gy]
                
                # Determine new state based on object type
                new_state = UNKNOWN
                
                # TODO: clean this up?
                if obj_type_idx == OBJECT_TO_IDX['wall']:
                    new_state = WALL
                elif obj_type_idx == OBJECT_TO_IDX['lava']:
                    new_state = LAVA
                elif obj_type_idx == OBJECT_TO_IDX['ball']:
                    new_state = PERSON
                elif obj_type_idx == OBJECT_TO_IDX['goal']:
                    new_state = EXIT
                elif obj_type_idx == OBJECT_TO_IDX['door']:
                    new_state = DOOR  # Track doors as separate state
                elif obj_type_idx == OBJECT_TO_IDX['floor']:
                    # Map floor color to specific floor state
                    color_idx = view[view_x, view_y, 1]
                    if color_idx == COLOR_TO_IDX['red']:
                        new_state = FLOOR_RED
                    elif color_idx == COLOR_TO_IDX['green']:
                        new_state = FLOOR_GREEN
                    elif color_idx == COLOR_TO_IDX['blue']:
                        new_state = FLOOR_BLUE
                    elif color_idx == COLOR_TO_IDX['purple']:
                        new_state = FLOOR_PURPLE
                    elif color_idx == COLOR_TO_IDX['yellow']:
                        new_state = FLOOR_YELLOW
                    elif color_idx == COLOR_TO_IDX['grey']:
                        new_state = FLOOR_GREY
                    else:
                        new_state = EMPTY  # Fallback for unknown floor color
                elif obj_type_idx == 1:  # Empty cell
                    new_state = EMPTY
                elif obj_type_idx == 0:  # Unseen (shouldn't happen in view)
                    new_state = UNKNOWN
                else:
                    # Unknown object type
                    new_state = EMPTY  # Default to empty for unknown objects
                
                # Update knowledge grid
                if new_state != UNKNOWN:
                    # Always update with new information
                    if old_state != new_state:
                        self.knowledge_grid[gx, gy] = new_state
                        self._update_frontier_set(gx, gy)

    def mark_wall_in_front(self, agent_pos, agent_dir):
        """Marks the cell directly in front of the agent as a WALL."""
        dx, dy = self.DIR_TO_VEC[agent_dir]
        fx, fy = agent_pos[0] + dx, agent_pos[1] + dy
        
        if 0 <= fx < self.width and 0 <= fy < self.height:
            print(f"Marking blocked cell ({fx}, {fy}) as WALL.")
            self.knowledge_grid[fx, fy] = WALL
            # Remove from frontiers if present
            self.frontiers.discard((fx, fy))
            # Update neighbors' frontier status
            self._update_frontier_set(fx, fy)

    def find_path_bfs(self, start, goal):
        """BFS to find path on known map."""
        queue = deque([start])
        came_from = {start: None}
        
        while queue:
            curr = queue.popleft()
            
            if curr == goal:
                break
            
            curr_x, curr_y = curr
            
            for dx, dy in self.DIR_TO_VEC:
                nx, ny = curr_x + dx, curr_y + dy
                
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    cell_type = self.knowledge_grid[nx, ny]
                    is_walkable = cell_type in WALKABLE_STATES or cell_type == PERSON
                    
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
        
        path.reverse()
        return path

    def get_known_people_locations(self):
        """Returns list of (x,y) tuples where PERSON is marked on grid"""
        return [tuple(x) for x in np.argwhere(self.knowledge_grid == PERSON)]

    def remove_person_from_memory(self, pos):
        """Called after successful rescue"""
        if self.knowledge_grid[pos] == PERSON:
            self.knowledge_grid[pos] = EMPTY
            self._update_frontier_set(pos[0], pos[1])

    def search_until_new_discovery(self):
        """
        Runs the get_action loop for one step.
        """
        # Check if we already know about a person
        if len(self.get_known_people_locations()) > 0:
            return 'FOUND_NEW_PERSON'
            
        action = self.get_action(self.env.agent_pos, self.env.agent_dir)
        
        if action is None:
            return 'EXPLORED_ALL'
            
        # Execute the action
        obs, reward, term, trunc, info = self.env.step(action)
        self.update_map(obs, self.env.agent_pos, self.env.agent_dir)
        
        return 'CONTINUE'

    def get_action(self, agent_pos, agent_dir):
        """Decides the next action."""
        
        # Check if stuck
        if self.last_pos == agent_pos and self.last_action == self.env.actions.forward:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            
        self.last_pos = agent_pos
        
        if self.stuck_count > 2:  # Stuck for multiple steps
            # We are stuck
            if self.last_action == self.env.actions.forward:
                self.mark_wall_in_front(agent_pos, agent_dir)
            
            # Try turning right to unstick.
            print("Stuck! Turning right to unstick.")
            self.stuck_count = 0  # Reset after handling
            self.last_action = self.env.actions.right 
            return self.last_action
        
        # Check if we found the person
        people_locs = np.argwhere(self.knowledge_grid == PERSON)
        if len(people_locs) > 0:
            print("Person found at:", people_locs[0])
            return None  # Stop searching, let controller handle rescue
            
        # Get Frontiers from cached set
        if not self.frontiers:
            print("No frontiers remaining.")
            # Check if we've explored enough
            explored_pct = np.sum(self.knowledge_grid != UNKNOWN) / self.knowledge_grid.size
            if explored_pct > 0.95:  # Explored 95% of map
                print("Map mostly explored, person not found.")
                return None
            else:
                # Try turning to potentially discover new areas
                print("Turning to look for new areas...")
                self.last_action = self.env.actions.right
                return self.last_action
            
        # Find nearest frontier
        best_path = None
        min_dist = float('inf')
        
        # Sort frontiers by Manhattan distance to prioritize closer ones
        sorted_frontiers = sorted(list(self.frontiers), 
                                key=lambda f: abs(f[0]-agent_pos[0]) + abs(f[1]-agent_pos[1]))
        
        # Check the closest frontiers
        checked = 0
        for f in sorted_frontiers: 
            if checked >= 10:
                break
            path = self.find_path_bfs(agent_pos, f)
            if path:
                dist = len(path)
                if dist < min_dist:
                    min_dist = dist
                    best_path = path
                    break  # Take first reachable frontier
            checked += 1
        
        if best_path is None:
            print("No reachable frontiers remaining.")
            # All frontiers are unreachable
            return None
        
        # check len(best_path) >= 2 bc len 1 path means agent is already at frontier
        if len(best_path) < 2:
            # Already at frontier, turn to see more
            self.last_action = self.env.actions.right
            return self.last_action
            
        # Determine action to move along path
        next_cell = best_path[1]
        
        dx = next_cell[0] - agent_pos[0]
        dy = next_cell[1] - agent_pos[1]
        
        desired_dir = -1
        if dx == 1: desired_dir = 0
        elif dy == 1: desired_dir = 1
        elif dx == -1: desired_dir = 2
        elif dy == -1: desired_dir = 3
        
        if desired_dir == agent_dir:
            action = self.env.actions.forward
            #print("Moving forward to", next_cell)
        elif (desired_dir - agent_dir) % 4 == 1:
            action = self.env.actions.right
            #print("Turning right towards", next_cell)
        else:
            action = self.env.actions.left
            #print("Turning left towards", next_cell)
            
        self.last_action = action

        return action

def run_search_demo():
    # Setup specific scenario
    env = SAREnv(render_mode="human")
    
    obs, _ = env.reset()
    agent = SearchAgent(env)
    
    # Initial map update
    agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    for step in range(1000):
        env.render()        
        action = agent.get_action(env.agent_pos, env.agent_dir)
                
        if action is None:
            print(f"Search completed at step {step}!")
            break
            
        obs, reward, term, trunc, info = env.step(action)
        agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        if term or trunc:
            print("Episode terminated!")
            break

    return agent.knowledge_grid, env

if __name__ == "__main__":
    final_map, env = run_search_demo()
