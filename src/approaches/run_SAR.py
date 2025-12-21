import numpy as np
from SAREnv import SAREnv
from search_agent import SearchAgent, UNKNOWN, EMPTY, WALL, LAVA, EXIT, PERSON
from rescue_agent import RescueAgent

def run_multi_person_mission():
    """
    Integrated demo that runs search and rescue for multiple people.
    """
    # 1. Setup environment with consistent parameters
    print("=" * 60)
    print("MULTI-PERSON SEARCH AND RESCUE MISSION")
    print("=" * 60)
    
    TOTAL_PEOPLE = 3
    
    env = SAREnv(
        room_size=5,
        num_rows=3,
        num_cols=3,
        num_people=TOTAL_PEOPLE,
        num_exits=2,
        num_collapsed_floors=5,
        agent_view_size=3,
        render_mode="human"
    )

    obs, _ = env.reset()
    
    # Initialize Search Agent (The Map Keeper)
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    max_steps = 2000
    step_count = 0
    
    while env.people_rescued < TOTAL_PEOPLE and step_count < max_steps:
        print(f"\n--- Status: Rescued {env.people_rescued}/{TOTAL_PEOPLE} ---")
        
        # --- Step 1: Check Knowledge Base ---
        # Do we already know where a person is?
        known_people = search_agent.get_known_people_locations()
        
        target_person = None
        if len(known_people) > 0:
            # Pick the first known person
            target_person = known_people[0]
            
        # --- Step 2: Decide Mode ---
        if target_person:
            print(f"\n[MODE SWITCH] Person known at {target_person}. Switching to RESCUE mode.")
            
            # Initialize Rescue Agent with current knowledge (shared search_agent)
            rescue_bot = RescueAgent(env, search_agent)
            
            # Execute rescue
            success = rescue_bot.run_rescue(target_pos=target_person)
            
            if success:
                print(f"[SUCCESS] Person at {target_person} rescued!")
                # CRITICAL: Update map to reflect person is gone
                search_agent.remove_person_from_memory(target_person)
            else:
                print(f"[FAILURE] Could not rescue person at {target_person}.")
                # Mark as unreachable so we don't keep trying immediately
                # For now, we'll just remove them from memory to avoid infinite loop
                # In a real system, we'd mark as "unreachable" and try others
                print("Removing unreachable person from memory to continue search.")
                search_agent.remove_person_from_memory(target_person)
                
        else:
            print("\n[MODE SWITCH] No people known. Switching to SEARCH mode.")
            
            # Run search step-by-step until something interesting happens
            status = "CONTINUE"
            while status == "CONTINUE" and step_count < max_steps:
                env.render()
                status = search_agent.search_until_new_discovery()
                
                # Update step count (approximate, search_until_new_discovery runs 1 step)
                step_count += 1
                
                # Check for termination
                if env.people_rescued >= TOTAL_PEOPLE:
                    break
            
            if status == "EXPLORED_ALL":
                print("Map fully explored. No more people found.")
                break
            elif status == "FOUND_NEW_PERSON":
                print("New person discovered! Switching to rescue logic.")
                
    print("\n" + "=" * 60)
    print("MISSION END")
    print("=" * 60)
    print(f"Total People Rescued: {env.people_rescued}/{TOTAL_PEOPLE}")
    
    return env, search_agent.knowledge_grid, env.people_rescued == TOTAL_PEOPLE


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
        env, knowledge_grid, success = run_multi_person_mission()