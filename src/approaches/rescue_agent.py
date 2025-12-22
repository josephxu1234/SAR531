import numpy as np
from heapq import heappush, heappop

# Map states
UNKNOWN = 0
EMPTY = 1
WALL = 2
LAVA = 3
EXIT = 4
PERSON = 5
DOOR = 6
FLOOR_RED = 7
FLOOR_GREEN = 8
FLOOR_BLUE = 9
FLOOR_PURPLE = 10
FLOOR_YELLOW = 11
FLOOR_GREY = 12

WALKABLE_STATES = {EMPTY, EXIT, PERSON, DOOR, FLOOR_RED, FLOOR_GREEN, FLOOR_BLUE, FLOOR_PURPLE, FLOOR_YELLOW, FLOOR_GREY}

class RescueAgent:
    def __init__(self, env, search_agent):
        """Initialize rescue agent with environment and search agent reference."""
        self.env = env
        self.search_agent = search_agent
        self.knowledge_grid = search_agent.knowledge_grid
        # Directions: 0=Right, 1=Down, 2=Left, 3=Up
        self.DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def _find_person(self):
        """Find first person location in knowledge grid."""
        locs = np.argwhere(self.knowledge_grid == PERSON)
        return tuple(locs[0]) if len(locs) > 0 else None

    def _find_nearest_exit(self, from_pos):
        """Find nearest exit to given position using Manhattan distance."""
        exit_locs = np.argwhere(self.knowledge_grid == EXIT)
        if len(exit_locs) == 0:
            return None
        
        min_dist = float('inf')
        nearest_exit = None
        for exit_loc in exit_locs:
            exit_pos = tuple(exit_loc)
            dist = abs(exit_pos[0] - from_pos[0]) + abs(exit_pos[1] - from_pos[1])
            if dist < min_dist:
                min_dist = dist
                nearest_exit = exit_pos
        return nearest_exit

    def _is_safe(self, x, y):
        """Check if cell is within bounds and walkable."""
        if not (0 <= x < self.env.width and 0 <= y < self.env.height):
            return False
        return self.knowledge_grid[x, y] in WALKABLE_STATES or self.knowledge_grid[x, y] == PERSON

    def _astar(self, start, goal):
        """A* pathfinding from start to goal with turn cost consideration."""
        # Start with current agent direction if at start position
        start_dir = self.env.agent_dir if start == tuple(self.env.agent_pos) else 0
        open_set = [(0, start, start_dir)]
        came_from = {(start, start_dir): None}
        g_score = {(start, start_dir): 0}
        
        MOVE_COST = 1
        TURN_COST = 0.5

        while open_set:
            _, current, current_dir = heappop(open_set)
            
            # Goal reached - reconstruct path
            if current == goal:
                path = []
                state = (current, current_dir)
                while state is not None:
                    path.append(state[0])
                    state = came_from[state]
                path.reverse()
                return path

            # Explore neighbors in all directions
            for new_dir, (dx, dy) in enumerate(self.DIR_TO_VEC):
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if not self._is_safe(nx, ny):
                    continue

                # Add turn cost if changing direction
                turn_cost = 0 if new_dir == current_dir else TURN_COST
                tentative_g = g_score[(current, current_dir)] + MOVE_COST + turn_cost
                
                neighbor_state = (neighbor, new_dir)
                if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                    g_score[neighbor_state] = tentative_g
                    f_score = tentative_g + abs(goal[0]-nx) + abs(goal[1]-ny)
                    heappush(open_set, (f_score, neighbor, new_dir))
                    came_from[neighbor_state] = (current, current_dir)
        return None

    def _get_action_to_face(self, current_dir, desired_dir):
        """Return list of turn actions needed to face desired direction."""
        diff = (desired_dir - current_dir) % 4
        if diff == 0:
            return []
        elif diff == 1:
            return [self.env.actions.right]
        elif diff == 2:
            return [self.env.actions.right, self.env.actions.right]
        else:
            return [self.env.actions.left]

    def _move_to(self, next_pos):
        """Move agent to adjacent position by turning and stepping forward."""
        agent_pos = tuple(self.env.agent_pos)
        next_pos = (int(next_pos[0]), int(next_pos[1]))
        
        # Calculate direction to target
        dx = next_pos[0] - agent_pos[0]
        dy = next_pos[1] - agent_pos[1]

        # Determine which direction to face
        desired_dir = -1
        if dx == 1 and dy == 0:
            desired_dir = 0  # Right
        elif dx == 0 and dy == 1:
            desired_dir = 1  # Down
        elif dx == -1 and dy == 0:
            desired_dir = 2  # Left
        elif dx == 0 and dy == -1:
            desired_dir = 3  # Up
        else:
            print(f"Error: Cannot move from {agent_pos} to {next_pos}")
            return False

        # Turn to face target direction
        turn_actions = self._get_action_to_face(self.env.agent_dir, desired_dir)
        for action in turn_actions:
            obs, _, _, _, _ = self.env.step(action)
            self.env.render()
            self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)

        # Move forward
        obs, _, _, _, _ = self.env.step(self.env.actions.forward)
        self.env.render()
        self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
        
        # Verify we reached target (allow small deviation for auto-pickup)
        actual_pos = tuple(self.env.agent_pos)
        if actual_pos != next_pos:
            if abs(actual_pos[0] - next_pos[0]) <= 1 and abs(actual_pos[1] - next_pos[1]) <= 1:
                return True
            return False
        return True

    def run_rescue(self, target_pos=None):
        """Execute full rescue operation: navigate to person, pick up, deliver to exit."""
        person_pos = self._find_person() if target_pos is None else target_pos
        if person_pos is None:
            print("No person found")
            return False

        print(f"\n=== Rescue Operation ===")
        print(f"Person at: {person_pos}")
        agent_pos = tuple(self.env.agent_pos)

        # Step 1: Navigate to cell adjacent to person
        target_pos = None
        for dx, dy in self.DIR_TO_VEC:
            adj_x, adj_y = person_pos[0] + dx, person_pos[1] + dy
            if 0 <= adj_x < self.env.width and 0 <= adj_y < self.env.height:
                if self._is_safe(adj_x, adj_y):
                    target_pos = (adj_x, adj_y)
                    break
        
        if target_pos is None:
            print("No walkable cell adjacent to person")
            return False
        
        # Check if already adjacent
        dx_to_person = abs(person_pos[0] - agent_pos[0])
        dy_to_person = abs(person_pos[1] - agent_pos[1])
        
        if dx_to_person + dy_to_person != 1:
            # Find and follow path to person
            path = self._astar(agent_pos, target_pos)
            if path is None:
                print("No path to person")
                return False

            for i, next_pos in enumerate(path[1:], 1):
                if not self._move_to(next_pos):
                    # Check if auto-pickup happened during movement
                    if self.env.carrying is not None and self.env.carrying.type == "ball":
                        break
                    return False
        
        # Step 2: Pick up person using pickup action
        if self.env.carrying is None:
            # Find which direction the person is in
            person_dir = None
            for i, (dx, dy) in enumerate(self.DIR_TO_VEC):
                check_x = int(self.env.agent_pos[0] + dx)
                check_y = int(self.env.agent_pos[1] + dy)
                if 0 <= check_x < self.env.width and 0 <= check_y < self.env.height:
                    cell = self.env.grid.get(check_x, check_y)
                    if cell is not None and cell.type == "ball":
                        person_dir = i
                        break
            
            if person_dir is None:
                print("Person not adjacent")
                return False
            
            # Face the person and pick them up
            for action in self._get_action_to_face(self.env.agent_dir, person_dir):
                obs, _, _, _, _ = self.env.step(action)
                self.env.render()
                self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
            
            obs, _, _, _, _ = self.env.step(self.env.actions.pickup)
            self.env.render()
            self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
        
        # Verify pickup succeeded
        if self.env.carrying is None or self.env.carrying.type != "ball":
            print("Failed to pick up person")
            return False
        
        print("Picked up person")
        
        # Step 3: Navigate to nearest exit
        agent_pos = tuple(self.env.agent_pos)
        exit_pos = self._find_nearest_exit(agent_pos)
        if exit_pos is None:
            print("No exit found")
            return False
        
        # Find and follow path to exit
        path = self._astar(agent_pos, exit_pos)
        if path is None:
            print("No path to exit")
            return False

        for next_pos in path[1:]:
            if not self._move_to(next_pos):
                return False
        
        print("Rescue complete!")
        return True