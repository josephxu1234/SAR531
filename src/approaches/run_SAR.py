import numpy as np
from SAREnv import SAREnv
from search_agent import SearchAgent, UNKNOWN, EMPTY, WALL, LAVA, EXIT, PERSON
from rescue_agent import RescueAgent

def run_integrated_search_and_rescue():
    """
    Integrated demo that runs search phase, then rescue phase on the same environment.
    """
    # 1. Setup environment with consistent parameters
    print("=" * 60)
    print("SEARCH AND RESCUE MISSION")
    print("=" * 60)
    
    env = SAREnv(
        room_size=5,
        num_rows=2,
        num_cols=2,
        num_people=1,
        num_exits=1,
        num_collapsed_floors=6,
        agent_view_size=3,
        # agent_view_size uses default from SAREnv
        render_mode="human"
    )

    obs, _ = env.reset()
    
    # 2. SEARCH PHASE - Find the person
    print("\n" + "=" * 60)
    print("PHASE 1: SEARCH")
    print("=" * 60)
    print("Objective: Locate person in need of rescue\n")
    
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    search_complete = False
    max_search_steps = 500
    
    for step in range(max_search_steps):
        env.render()
        
        action = search_agent.get_action(env.agent_pos, env.agent_dir)
        
        if action is None:
            print(f"\nSearch completed at step {step}!")
            search_complete = True
            break
        
        obs, reward, term, trunc, info = env.step(action)
        search_agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        # Check if we accidentally picked up the person during search
        if info.get('carrying_person', False):
            print("\nWARNING: Person picked up during search phase!")
            print("Dropping person at current location for rescue phase...")
            # Drop the person (they become a ball object on the ground)
            if env.carrying is not None:
                env.grid.set(*env.agent_pos, env.carrying)
                env.carrying = None
        
        if term or trunc:
            print("\nSearch terminated early!")
            break
    
    if not search_complete:
        print(f"\nSearch phase ended after {max_search_steps} steps without finding person.")
    
    # Display search results
    print("\n" + "-" * 60)
    print("SEARCH PHASE COMPLETE")
    print("-" * 60)
    
    knowledge_grid = search_agent.knowledge_grid
    people_found = np.sum(knowledge_grid == PERSON)
    exits_found = np.sum(knowledge_grid == EXIT)
    cells_explored = np.sum(knowledge_grid != UNKNOWN)
    total_cells = knowledge_grid.shape[0] * knowledge_grid.shape[1]
    
    print(f"People found: {people_found}")
    print(f"Exits found: {exits_found}")
    print(f"Area explored: {cells_explored}/{total_cells} cells ({100*cells_explored/total_cells:.1f}%)")
    
    # Print the knowledge map
    print("\nKnowledge Map:")
    chars = {UNKNOWN: '?', EMPTY: '.', WALL: '#', LAVA: 'X', EXIT: 'E', PERSON: 'P'}
    for y in range(knowledge_grid.shape[1]):
        line = ""
        for x in range(knowledge_grid.shape[0]):
            line += chars[knowledge_grid[x, y]]
        print(line)
    
    # Check if we can proceed to rescue
    if people_found == 0:
        print("\n[X] Cannot proceed to rescue: No person found!")
        return env, knowledge_grid, False
    
    if exits_found == 0:
        print("\n[X] Cannot proceed to rescue: No exit found!")
        return env, knowledge_grid, False
    
    # 3. RESCUE PHASE - Navigate to person and bring them to exit
    print("\n" + "=" * 60)
    print("PHASE 2: RESCUE")
    print("=" * 60)
    print("Objective: Retrieve person and evacuate to exit\n")
    
    # Ensure person is not already being carried
    if env.carrying is not None:
        print("Dropping carried object before rescue phase...")
        env.carrying = None
    
    # Initialize rescue agent with knowledge from search
    rescue_agent = RescueAgent(env, knowledge_grid)
    
    # Run rescue operation
    rescue_success = rescue_agent.run_rescue()
    
    # 4. Final Results
    print("\n" + "=" * 60)
    print("MISSION COMPLETE")
    print("=" * 60)
    
    if rescue_success:
        print("[SUCCESS] Person successfully rescued and evacuated!")
        print(f"[SUCCESS] People rescued: {env.people_rescued}/{env.num_people}")
    else:
        print("[FAILURE] Rescue operation failed")
        print(f"  People rescued: {env.people_rescued}/{env.num_people}")
    
    return env, knowledge_grid, rescue_success


def run_search_only_demo():
    """
    Demo that only runs the search phase (useful for debugging).
    """
    print("=" * 60)
    print("SEARCH ONLY DEMO")
    print("=" * 60)
    
    env = SAREnv(
        room_size=5,
        num_rows=2,
        num_cols=2,
        num_people=1,
        num_exits=1,
        num_collapsed_floors=6,
        agent_view_size=3,
        # agent_view_size uses default from SAREnv
        render_mode="human"
    )
    
    obs, _ = env.reset()
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    for step in range(500):
        env.render()
        action = search_agent.get_action(env.agent_pos, env.agent_dir)
        
        if action is None:
            print(f"\nSearch completed at step {step}!")
            break
        
        obs, reward, term, trunc, info = env.step(action)
        search_agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        if term or trunc:
            break
    
    # Display results
    knowledge_grid = search_agent.knowledge_grid
    print("\nFinal Knowledge Map:")
    chars = {UNKNOWN: '?', EMPTY: '.', WALL: '#', LAVA: 'X', EXIT: 'E', PERSON: 'P'}
    for y in range(knowledge_grid.shape[1]):
        line = ""
        for x in range(knowledge_grid.shape[0]):
            line += chars[knowledge_grid[x, y]]
        print(line)
    
    people_found = np.sum(knowledge_grid == PERSON)
    print(f"\nPeople found: {people_found}")
    
    return env, knowledge_grid


def run_rescue_only_demo():
    """
    Demo that only runs the rescue phase with perfect knowledge (useful for debugging).
    """
    print("=" * 60)
    print("RESCUE ONLY DEMO (with perfect knowledge)")
    print("=" * 60)
    
    env = SAREnv(
        room_size=5,
        num_rows=2,
        num_cols=2,
        num_people=1,
        num_exits=1,
        num_collapsed_floors=2,
        # agent_view_size uses default from SAREnv
        render_mode="human"
    )
    
    obs, _ = env.reset()
    
    # Create perfect knowledge grid
    knowledge_grid = np.zeros((env.width, env.height), dtype=int)
    knowledge_grid[:] = EMPTY
    
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
    
    # Run rescue
    rescue_agent = RescueAgent(env, knowledge_grid)
    rescue_success = rescue_agent.run_rescue()
    
    if rescue_success:
        print("\n[OK] Rescue successful!")
    else:
        print("\n[X] Rescue failed!")
    
    return env, rescue_success


if __name__ == "__main__":
    import sys
    
    # Choose which demo to run based on command line argument
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "search":
            env, knowledge_grid = run_search_only_demo()
        elif mode == "rescue":
            env, success = run_rescue_only_demo()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python run_rescue.py [search|rescue|full]")
    else:
        # Default: run full integrated demo
        env, knowledge_grid, success = run_integrated_search_and_rescue()