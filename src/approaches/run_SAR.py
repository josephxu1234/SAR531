import numpy as np
from SAREnv import SAREnv
from search_agent import SearchAgent, UNKNOWN, EMPTY, WALL, LAVA, EXIT, PERSON
from rescue_agent import RescueAgent

def run_multi_person_mission():
    """
    Integrated demo that runs search and rescue for multiple people.
    """
    
    TOTAL_PEOPLE = 5
    
    env = SAREnv(
        room_size=8,
        num_rows=4,
        num_cols=4,
        num_people=TOTAL_PEOPLE,
        num_exits=3,
        num_collapsed_floors=20,
        agent_view_size=3,
        render_mode="human"
    )

    obs, _ = env.reset()
    
    # Initialize Search Agent 
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    max_steps = 2000
    step_count = 0
    
    while env.people_rescued < TOTAL_PEOPLE and step_count < max_steps:        
        # Do we already know where a person is?
        known_people = search_agent.get_known_people_locations()
        
        target_person = None
        if len(known_people) > 0:
            # Pick the first known person
            target_person = known_people[0]
            
        #  choose mode
        if target_person:
            print("Switching to RESCUE mode.")
            
            # Initialize Rescue Agent with current knowledge
            rescue_bot = RescueAgent(env, search_agent)
            
            # Execute rescue
            success = rescue_bot.run_rescue(target_pos=target_person)
            
            if success:
                print(f"Person at {target_person} rescued!")
                # Update map to reflect person is gone
                search_agent.remove_person_from_memory(target_person)
            else:
                print(f"Could not rescue person at {target_person}.")
                # Mark as unreachable so we don't keep trying immediately
                print("Removing unreachable person from memory to continue search.")
                search_agent.remove_person_from_memory(target_person)
                
        else:
            print("Switching to SEARCH mode.")
            
            # Run search until something interesting happens
            status = "CONTINUE"
            while status == "CONTINUE" and step_count < max_steps:
                env.render()
                status = search_agent.search_until_new_discovery()
                
                # Update step count
                step_count += 1
                
                # Check for termination
                if env.people_rescued >= TOTAL_PEOPLE:
                    break
            
            if status == "EXPLORED_ALL":
                print("Map fully explored. No more people found.")
                break
            elif status == "FOUND_NEW_PERSON":
                print("New person discovered!")
                
    print(f"Total People Rescued: {env.people_rescued}/{TOTAL_PEOPLE}")
    
    return env, search_agent.knowledge_grid, env.people_rescued == TOTAL_PEOPLE


if __name__ == "__main__":
    import sys
    
    env, knowledge_grid, success = run_multi_person_mission()