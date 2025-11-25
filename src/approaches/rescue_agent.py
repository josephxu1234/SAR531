import numpy as np
from heapq import heappush, heappop

# Map states (match SearchAgent)
UNKNOWN = 0
EMPTY = 1
WALL = 2
LAVA = 3
EXIT = 4
PERSON = 5

class RescueAgent:
    def __init__(self, env, search_agent):
        """
        env: SAREnv environment
        search_agent: SearchAgent instance (the map keeper)
        """
        self.env = env
        self.search_agent = search_agent
        self.knowledge_grid = search_agent.knowledge_grid # Direct reference
        
        # Directions: 0=Right, 1=Down, 2=Left, 3=Up
        self.DIR_TO_VEC = [
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0), # Left
            (0, -1)  # Up
        ]

    def _find_person(self):
        """Find first person location in knowledge grid."""
        locs = np.argwhere(self.knowledge_grid == PERSON)
        return tuple(locs[0]) if len(locs) > 0 else None

    def _find_nearest_exit(self, from_pos):
        """Find nearest exit to given position."""
        exit_locs = np.argwhere(self.knowledge_grid == EXIT)
        if len(exit_locs) == 0:
            return None
        
        # Find closest exit by Manhattan distance
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
        """Check if cell is walkable."""
        if not (0 <= x < self.env.width and 0 <= y < self.env.height):
            return False
        return self.knowledge_grid[x, y] in [EMPTY, EXIT, PERSON]

    def _astar(self, start, goal):
        """A* pathfinding from start to goal with turn cost."""
        # State is now (position, direction) to track orientation
        # Start with current agent direction if at start position
        if start == tuple(self.env.agent_pos):
            start_dir = self.env.agent_dir
        else:
            start_dir = 0  # Default direction if not at current position
        
        open_set = []
        heappush(open_set, (0, start, start_dir))
        came_from = {(start, start_dir): None}
        g_score = {(start, start_dir): 0}
        
        MOVE_COST = 1
        TURN_COST = 0.5  # Cost for changing direction

        while open_set:
            _, current, current_dir = heappop(open_set)
            
            if current == goal:
                # Reconstruct path (just positions, not directions)
                path = []
                state = (current, current_dir)
                while state is not None:
                    path.append(state[0])
                    state = came_from[state]
                path.reverse()
                return path

            for new_dir, (dx, dy) in enumerate(self.DIR_TO_VEC):
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if not self._is_safe(nx, ny):
                    continue

                # Calculate turn cost
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
        """Return action(s) needed to face desired direction."""
        diff = (desired_dir - current_dir) % 4
        
        if diff == 0:
            return []  # Already facing correct direction
        elif diff == 1:
            return [self.env.actions.right]
        elif diff == 2:
            return [self.env.actions.right, self.env.actions.right]
        elif diff == 3:
            return [self.env.actions.left]
        
        return []

    def _move_to(self, next_pos):
        """Move agent to next position (must be adjacent)."""
        agent_pos = tuple(self.env.agent_pos)
        
        # Convert next_pos to tuple of regular ints to avoid numpy types
        next_pos = (int(next_pos[0]), int(next_pos[1]))
        
        dx = next_pos[0] - agent_pos[0]
        dy = next_pos[1] - agent_pos[1]

        # Determine desired direction
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
            print(f"Error: Cannot move from {agent_pos} to {next_pos} - not adjacent!")
            print(f"  Delta: dx={dx}, dy={dy}")
            return False

        # Check if target cell is actually walkable
        target_cell = self.env.grid.get(*next_pos)
        if target_cell is not None:
            if target_cell.type == 'wall':
                print(f"Error: Target position {next_pos} is a wall!")
                return False
            elif target_cell.type == 'lava':
                print(f"Error: Target position {next_pos} is lava!")
                return False

        # Turn to face the desired direction
        turn_actions = self._get_action_to_face(self.env.agent_dir, desired_dir)
        for action in turn_actions:
            obs, reward, term, trunc, info = self.env.step(action)
            self.env.render()
            # Update map during turns
            self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)

        # Check what's in front before moving
        fwd_pos = self.env.front_pos
        fwd_cell = self.env.grid.get(*fwd_pos)
        if fwd_cell is not None and fwd_cell.type in ['wall', 'lava']:
            print(f"Warning: Obstacle in front at {fwd_pos}: {fwd_cell.type}")
            print(f"  Agent at {tuple(self.env.agent_pos)}, facing dir {self.env.agent_dir}")
            return False

        # Move forward
        obs, reward, term, trunc, info = self.env.step(self.env.actions.forward)
        self.env.render()
        
        # Update map after move
        self.search_agent.update_map(obs, self.env.agent_pos, self.env.agent_dir)
        
        # Check if we actually moved
        actual_pos = tuple(self.env.agent_pos)
        if actual_pos != next_pos:
            print(f"Warning: Expected to be at {next_pos}, but at {actual_pos}")
            print(f"  This might be due to auto-pickup or collision")
            # Check if we're at least close
            if abs(actual_pos[0] - next_pos[0]) <= 1 and abs(actual_pos[1] - next_pos[1]) <= 1:
                print(f"  Position is close enough, continuing...")
                return True
            return False
        
        return True

    def _pickup_person(self):
        """Attempt to pick up person at current location."""
        # Check if there's a person at current location
        agent_pos = tuple(self.env.agent_pos)
        cell = self.env.grid.get(*agent_pos)
        
        if cell is not None and cell.type == "ball":
            # The environment should auto-pickup when we step onto the person
            # But let's make sure by using the pickup action if needed
            if self.env.carrying is None:
                obs, reward, term, trunc, info = self.env.step(self.env.actions.pickup)
                self.env.render()
                print(f"Picked up person at {agent_pos}")
            return True
        else:
            print(f"No person at {agent_pos} to pick up")
            return False

    def run_rescue(self, target_pos=None):
        """Execute the full rescue operation."""
        if target_pos is None:
            person_pos = self._find_person()
        else:
            person_pos = target_pos
        
        if person_pos is None:
            print("Cannot run rescue: no person found in knowledge grid.")
            return False

        print(f"\n=== Starting Rescue Operation ===")
        print(f"Person location (from knowledge): {person_pos}")
        print(f"Agent starting at: {tuple(self.env.agent_pos)}")

        # Step 1: Navigate to person (adjacent, not on top of)
        agent_pos = tuple(self.env.agent_pos)
        
        # Find an adjacent walkable cell to the person
        target_pos = None
        for dx, dy in self.DIR_TO_VEC:
            adj_x = person_pos[0] + dx
            adj_y = person_pos[1] + dy
            if 0 <= adj_x < self.env.width and 0 <= adj_y < self.env.height:
                if self._is_safe(adj_x, adj_y):
                    target_pos = (adj_x, adj_y)
                    break
        
        if target_pos is None:
            print("ERROR: No walkable cell adjacent to person!")
            return False
        
        print(f"Target position (adjacent to person): {target_pos}")
        
        # Check if person is already adjacent
        dx_to_person = abs(person_pos[0] - agent_pos[0])
        dy_to_person = abs(person_pos[1] - agent_pos[1])
        
        if dx_to_person + dy_to_person == 1:
            print(f"Already adjacent to person at {person_pos}")
        else:
            # Find path to adjacent cell
            path_to_person = self._astar(agent_pos, target_pos)
            
            if path_to_person is None:
                print("ERROR: No path to person!")
                return False

            print(f"Path to person: {len(path_to_person)} steps")
            
            # Follow path to person
            for i, next_pos in enumerate(path_to_person[1:], 1):
                print(f"Step {i}/{len(path_to_person)-1}: Moving to {next_pos}")
                current_pos = tuple(self.env.agent_pos)
                
                if not self._move_to(next_pos):
                    print(f"Failed to move from {current_pos} to {next_pos}!")
                    # Check if we're already at the person's location (auto-pickup might have happened)
                    if self.env.carrying is not None and self.env.carrying.type == "ball":
                        print("Person was picked up automatically during movement!")
                        break
                    # Try to recompute path from current position
                    print("Attempting to recompute path...")
                    current_pos = tuple(self.env.agent_pos)
                    new_path = self._astar(current_pos, person_pos)
                    if new_path and len(new_path) > 1:
                        print(f"Found new path with {len(new_path)} steps")
                        path_to_person = new_path
                    else:
                        print("Could not find alternate path!")
                        return False
        
        # Step 2: Pick up person using pickup action (must be adjacent)
        # Agent should now be adjacent to the person
        if self.env.carrying is None:
            print("Attempting to pick up person...")
            
            # Find the person in adjacent cells
            person_found = False
            person_dir = None
            
            for i, (dx, dy) in enumerate(self.DIR_TO_VEC):
                check_x = int(self.env.agent_pos[0] + dx)
                check_y = int(self.env.agent_pos[1] + dy)
                if 0 <= check_x < self.env.width and 0 <= check_y < self.env.height:
                    cell = self.env.grid.get(check_x, check_y)
                    if cell is not None and cell.type == "ball":
                        print(f"Person found adjacent at ({check_x}, {check_y})")
                        person_found = True
                        person_dir = i
                        break
            
            if not person_found:
                print("ERROR: Person not adjacent to agent!")
                print(f"Agent at: {tuple(self.env.agent_pos)}")
                # Search nearby
                for x in range(max(0, self.env.agent_pos[0]-2), min(self.env.width, self.env.agent_pos[0]+3)):
                    for y in range(max(0, self.env.agent_pos[1]-2), min(self.env.height, self.env.agent_pos[1]+3)):
                        cell = self.env.grid.get(x, y)
                        if cell is not None and cell.type == "ball":
                            print(f"  Found person at ({x}, {y})")
                return False
            
            # Face the person
            turn_actions = self._get_action_to_face(self.env.agent_dir, person_dir)
            for action in turn_actions:
                self.env.step(action)
                self.env.render()
            
            # Use pickup action
            print("Using pickup action...")
            obs, reward, term, trunc, info = self.env.step(self.env.actions.pickup)
            self.env.render()
        
        if self.env.carrying is None or self.env.carrying.type != "ball":
            print("ERROR: Failed to pick up person!")
            print(f"Current position: {tuple(self.env.agent_pos)}")
            print(f"Carrying: {self.env.carrying}")
            
            # Debug: scan entire grid for person
            print("\nDebug: Scanning entire grid for people...")
            for x in range(self.env.width):
                for y in range(self.env.height):
                    cell = self.env.grid.get(x, y)
                    if cell is not None and cell.type == "ball":
                        print(f"  Found person at ({x}, {y})")
            
            return False
        
        print(f"[OK] Successfully picked up person!")
        
        # Step 3: Find nearest exit
        agent_pos = tuple(self.env.agent_pos)
        exit_pos = self._find_nearest_exit(agent_pos)
        
        if exit_pos is None:
            print("ERROR: No exit found in knowledge grid!")
            return False
        
        print(f"Nearest exit: {exit_pos}")
        
        # Step 4: Navigate to exit
        path_to_exit = self._astar(agent_pos, exit_pos)
        
        if path_to_exit is None:
            print("ERROR: No path to exit!")
            return False

        print(f"Path to exit: {len(path_to_exit)} steps")
        
        # Follow path to exit
        for i, next_pos in enumerate(path_to_exit[1:], 1):
            print(f"Step {i}/{len(path_to_exit)-1}: Moving to {next_pos}")
            current_pos = tuple(self.env.agent_pos)
            
            if not self._move_to(next_pos):
                print(f"Failed to move from {current_pos} to {next_pos}!")
                # Try to recompute path from current position
                print("Attempting to recompute path to exit...")
                current_pos = tuple(self.env.agent_pos)
                new_path = self._astar(current_pos, exit_pos)
                if new_path and len(new_path) > 1:
                    print(f"Found new path with {len(new_path)} steps")
                    path_to_exit = new_path
                else:
                    print("Could not find alternate path to exit!")
                    return False
        
        print(f"[OK] Reached exit at {exit_pos}")
        print(f"[OK] Rescue complete! Person delivered to safety.")
        
        return True


def run_rescue_demo():
    """Standalone demo of rescue agent."""
    from SAREnv import SAREnv
    
    # Create environment
    env = SAREnv(
        room_size=5, 
        num_rows=2, 
        num_cols=2, 
        num_people=1, 
        num_exits=1,
        num_collapsed_floors=2,
        agent_view_size=7,
        render_mode="human"
    )
    
    obs, _ = env.reset()
    
    # Create a simple knowledge grid for demo
    # In practice, this would come from SearchAgent
    knowledge_grid = np.zeros((env.width, env.height), dtype=int)
    
    # Mark everything as empty (simplified - normally from search)
    knowledge_grid[:] = EMPTY
    
    # Mark walls
    for x in range(env.width):
        for y in range(env.height):
            cell = env.grid.get(x, y)
            if cell is not None:
                if cell.type == 'wall':
                    knowledge_grid[x, y] = WALL
                elif cell.type == 'lava':
                    knowledge_grid[x, y] = LAVA
                elif cell.type == 'ball':
                    knowledge_grid[x, y] = PERSON
                elif cell.type == 'goal':
                    knowledge_grid[x, y] = EXIT
    
    # Create and run rescue agent
    # Note: In standalone demo, we mock the search agent
    class MockSearchAgent:
        def __init__(self, grid):
            self.knowledge_grid = grid
        def update_map(self, obs, pos, dir):
            pass
            
    mock_search = MockSearchAgent(knowledge_grid)
    rescue_agent = RescueAgent(env, mock_search)
    success = rescue_agent.run_rescue()
    
    if success:
        print("\n*** RESCUE MISSION SUCCESS ***")
    else:
        print("\n*** RESCUE MISSION FAILED ***")
    
    return success


if __name__ == "__main__":
    run_rescue_demo()