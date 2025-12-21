"""
Comparison script that compares Baseline SearchAgent vs PPO Frontier Selection Agent.

This script:
- Runs both agents (Baseline and PPO) on the same set of environments
- Collects metrics (rewards, steps, success rate, people rescued)
- Tracks how often PPO picks nearest frontier vs farther ones
- Assuming you have a ppo_frontier.zip in the same directory containning the ppo model:
- run: python compare_agents.py --ppo-model ppo_frontier.zip --num-episodes 100
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from SAREnv import SAREnv
from ppo_frontier_wrapper import FrontierSelectionEnv
from search_agent import SearchAgent
from rescue_agent import RescueAgent


@dataclass
class EpisodeResult:
    """Container for single episode results."""
    episode_id: int
    agent_type: str  # "baseline" or "ppo"
    total_reward: float
    episode_length: int
    people_rescued: int
    success: bool
    lava_hits: int
    seed: int
    nearest_frontier_picks: int = 0  # Only for PPO
    total_frontier_picks: int = 0     # Only for PPO


def _adjacent_walkable(rescue_agent, pos):
    """Helper to find adjacent walkable cells."""
    adj = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < rescue_agent.env.width and 0 <= ny < rescue_agent.env.height:
            if rescue_agent._is_safe(nx, ny):
                adj.append((nx, ny))
    return adj


def _action_toward(env, next_pos):
    """Helper to get action toward next position."""
    curr = tuple(env.agent_pos)
    dx = next_pos[0] - curr[0]
    dy = next_pos[1] - curr[1]
    
    desired_dir = -1
    if dx == 1 and dy == 0:
        desired_dir = 0
    elif dx == 0 and dy == 1:
        desired_dir = 1
    elif dx == -1 and dy == 0:
        desired_dir = 2
    elif dx == 0 and dy == -1:
        desired_dir = 3
    
    if desired_dir == env.agent_dir:
        return env.actions.forward
    
    # Turn toward desired direction
    diff = (desired_dir - env.agent_dir) % 4
    if diff == 1:
        return env.actions.right
    return env.actions.left


def run_baseline_episode(env_config: Dict, seed: int, episode_id: int, 
                        max_steps: int = 2000, verbose: bool = False) -> EpisodeResult:
    """
    Run one episode with baseline search+rescue agent.
    Uses greedy nearest-frontier selection during search.
    """
    env = SAREnv(**env_config, render_mode=None)
    obs, _ = env.reset(seed=seed)
    
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    
    rescue_agent = RescueAgent(env, search_agent)
    
    mode = "search"  # "search" or "rescue"
    total_reward = 0.0
    episode_length = 0
    lava_hits = 0
    terminated = False
    truncated = False
    
    for step in range(max_steps):
        # Check if mission complete
        if env.people_rescued >= env.num_people:
            terminated = True
            break
        
        # Decide mode based on knowledge
        known_people = search_agent.get_known_people_locations()
        
        if known_people and mode == "search":
            # Found a person, switch to rescue
            mode = "rescue"
            if verbose:
                print(f"  Step {step}: Switching to RESCUE mode")
        
        # Get action based on mode
        if mode == "rescue" and known_people:
            # Run one step of rescue toward first known person
            person_pos = known_people[0]
            
            # Check if we're carrying someone
            if env.carrying is not None and env.carrying.type == "ball":
                # Navigate to exit
                exit_pos = rescue_agent._find_nearest_exit(tuple(env.agent_pos))
                if exit_pos:
                    path = rescue_agent._astar(tuple(env.agent_pos), exit_pos)
                    if path and len(path) > 1:
                        action = _action_toward(env, path[1])
                    else:
                        action = env.actions.forward
                else:
                    mode = "search"
                    continue
            else:
                # Check if adjacent to person for pickup
                dx = abs(person_pos[0] - env.agent_pos[0])
                dy = abs(person_pos[1] - env.agent_pos[1])
                
                if dx + dy == 1:
                    # Adjacent - pickup
                    # Face the person
                    dx_dir = person_pos[0] - env.agent_pos[0]
                    dy_dir = person_pos[1] - env.agent_pos[1]
                    
                    person_dir = -1
                    if dx_dir == 1:
                        person_dir = 0
                    elif dy_dir == 1:
                        person_dir = 1
                    elif dx_dir == -1:
                        person_dir = 2
                    elif dy_dir == -1:
                        person_dir = 3
                    
                    if person_dir != -1 and person_dir != env.agent_dir:
                        diff = (person_dir - env.agent_dir) % 4
                        action = env.actions.right if diff == 1 else env.actions.left
                    else:
                        action = env.actions.pickup
                else:
                    # Navigate to adjacent cell
                    adjacent_cells = _adjacent_walkable(rescue_agent, person_pos)
                    if adjacent_cells:
                        target = min(adjacent_cells, 
                                   key=lambda p: abs(p[0]-env.agent_pos[0])+abs(p[1]-env.agent_pos[1]))
                        path = rescue_agent._astar(tuple(env.agent_pos), target)
                        if path and len(path) > 1:
                            action = _action_toward(env, path[1])
                        else:
                            mode = "search"
                            continue
                    else:
                        mode = "search"
                        continue
        else:
            # Search mode - use search agent
            mode = "search"
            action = search_agent.get_action(env.agent_pos, env.agent_dir)
            
            if action is None:
                # Search exhausted
                break
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        search_agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        total_reward += reward
        episode_length += 1
        
        if reward <= -50:
            lava_hits += 1
        
        if terminated or truncated:
            break
        
        if verbose and episode_length % 100 == 0:
            print(f"  Step {episode_length}: mode={mode}, reward={total_reward:.1f}, rescued={env.people_rescued}")
    
    success = env.people_rescued >= env.num_people
    
    env.close()
    
    return EpisodeResult(
        episode_id=episode_id,
        agent_type="baseline",
        total_reward=total_reward,
        episode_length=episode_length,
        people_rescued=env.people_rescued,
        success=success,
        lava_hits=lava_hits,
        seed=seed
    )


def run_ppo_episode(ppo_model, env_config: Dict, seed: int, episode_id: int,
                   max_steps: int = 2000, verbose: bool = False) -> EpisodeResult:
    """Run one episode with PPO agent, tracking frontier selection patterns."""
    base_env = SAREnv(**env_config, render_mode=None)
    env = FrontierSelectionEnv(base_env, frontier_limit=64)
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    episode_length = 0
    lava_hits = 0
    terminated = False
    truncated = False
    nearest_frontier_picks = 0
    total_frontier_picks = 0
    
    # Anti-stall: track if making progress
    last_rescued_count = 0
    steps_since_rescue = 0
    max_steps_without_progress = 500
    
    for step in range(max_steps):
        # Get PPO action
        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)
        
        # Check if this was the nearest frontier
        frontiers = obs['frontiers']
        valid_mask = frontiers[:, 0] >= 0
        
        if np.any(valid_mask):
            total_frontier_picks += 1
            # Frontier 0 is always nearest (sorted by distance in wrapper)
            if action == 0:
                nearest_frontier_picks += 1
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        if reward <= -50:
            lava_hits += 1
        
        # Check for progress
        current_rescued = base_env.people_rescued
        if current_rescued > last_rescued_count:
            last_rescued_count = current_rescued
            steps_since_rescue = 0
        else:
            steps_since_rescue += 1
        
        # Force termination if stuck
        if steps_since_rescue > max_steps_without_progress:
            if verbose:
                print(f"  TIMEOUT: No progress for {max_steps_without_progress} steps")
            truncated = True
        
        if terminated or truncated:
            break
        
        if verbose and episode_length % 100 == 0:
            rescued = base_env.people_rescued
            print(f"  Step {episode_length}: reward={total_reward:.1f}, rescued={rescued}")
    
    success = base_env.people_rescued >= base_env.num_people
    
    env.close()
    
    return EpisodeResult(
        episode_id=episode_id,
        agent_type="ppo",
        total_reward=total_reward,
        episode_length=episode_length,
        people_rescued=base_env.people_rescued,
        success=success,
        lava_hits=lava_hits,
        seed=seed,
        nearest_frontier_picks=nearest_frontier_picks,
        total_frontier_picks=total_frontier_picks
    )


def compute_statistics(results: List[EpisodeResult]) -> Dict:
    """Compute statistical summary."""
    baseline_results = [r for r in results if r.agent_type == "baseline"]
    ppo_results = [r for r in results if r.agent_type == "ppo"]
    
    def stats_for_group(group: List[EpisodeResult]) -> Dict:
        if not group:
            return {}
        
        rewards = [r.total_reward for r in group]
        lengths = [r.episode_length for r in group]
        rescued = [r.people_rescued for r in group]
        successes = [r.success for r in group]
        lava_hits = [r.lava_hits for r in group]
        
        stats = {
            'count': len(group),
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'reward_min': float(np.min(rewards)),
            'reward_max': float(np.max(rewards)),
            'length_mean': float(np.mean(lengths)),
            'length_std': float(np.std(lengths)),
            'rescued_mean': float(np.mean(rescued)),
            'rescued_std': float(np.std(rescued)),
            'success_rate': float(np.mean(successes) * 100),
            'lava_hits_mean': float(np.mean(lava_hits)),
        }
        
        # PPO-specific stats
        if group and group[0].agent_type == "ppo":
            nearest_picks = sum(r.nearest_frontier_picks for r in group)
            total_picks = sum(r.total_frontier_picks for r in group)
            if total_picks > 0:
                stats['nearest_frontier_percentage'] = float(nearest_picks / total_picks * 100)
            else:
                stats['nearest_frontier_percentage'] = 0.0
        
        return stats
    
    return {
        'baseline': stats_for_group(baseline_results),
        'ppo': stats_for_group(ppo_results)
    }


def print_statistics(stats: Dict):
    """Print formatted statistics."""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    if stats.get('baseline'):
        print("\nBASELINE AGENT:")
        b = stats['baseline']
        print(f"  Episodes: {b['count']}")
        print(f"  Reward: {b['reward_mean']:.2f} ± {b['reward_std']:.2f}")
        print(f"  Episode Length: {b['length_mean']:.1f} ± {b['length_std']:.1f}")
        print(f"  People Rescued: {b['rescued_mean']:.2f} ± {b['rescued_std']:.2f}")
        print(f"  Success Rate: {b['success_rate']:.1f}%")
        print(f"  Lava Hits: {b['lava_hits_mean']:.2f}")
    
    if stats.get('ppo'):
        print("\nPPO AGENT:")
        p = stats['ppo']
        print(f"  Episodes: {p['count']}")
        print(f"  Reward: {p['reward_mean']:.2f} ± {p['reward_std']:.2f}")
        print(f"  Episode Length: {p['length_mean']:.1f} ± {p['length_std']:.1f}")
        print(f"  People Rescued: {p['rescued_mean']:.2f} ± {p['rescued_std']:.2f}")
        print(f"  Success Rate: {p['success_rate']:.1f}%")
        print(f"  Lava Hits: {p['lava_hits_mean']:.2f}")
        if 'nearest_frontier_percentage' in p:
            print(f"  Nearest Frontier Selection: {p['nearest_frontier_percentage']:.1f}%")
    
    # Comparison
    if stats.get('baseline') and stats.get('ppo'):
        print("\nCOMPARISON:")
        b, p = stats['baseline'], stats['ppo']
        reward_diff = p['reward_mean'] - b['reward_mean']
        success_diff = p['success_rate'] - b['success_rate']
        
        print(f"  Reward Difference: {reward_diff:+.2f} (PPO - Baseline)")
        print(f"  Success Rate Difference: {success_diff:+.1f}% (PPO - Baseline)")
    
    print("="*70)


def plot_results(results: List[EpisodeResult], save_path: str = "comparison.png"):
    """Generate comparison plots."""
    baseline = [r for r in results if r.agent_type == "baseline"]
    ppo = [r for r in results if r.agent_type == "ppo"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs PPO Agent Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward distribution
    if baseline and ppo:
        b_rewards = [r.total_reward for r in baseline]
        p_rewards = [r.total_reward for r in ppo]
        axes[0, 0].hist([b_rewards, p_rewards], label=['Baseline', 'PPO'],
                       alpha=0.7, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Total Reward')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Reward Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate
    if baseline or ppo:
        rates = []
        labels = []
        if baseline:
            rates.append(np.mean([r.success for r in baseline]) * 100)
            labels.append('Baseline')
        if ppo:
            rates.append(np.mean([r.success for r in ppo]) * 100)
            labels.append('PPO')
        
        bars = axes[0, 1].bar(labels, rates, color=['#FF6B6B', '#4ECDC4'], 
                             edgecolor='black', linewidth=1.5)
        axes[0, 1].set_ylabel('Success Rate (%)')
        axes[0, 1].set_title('Success Rate')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                          f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: People rescued
    if baseline and ppo:
        b_rescued = [r.people_rescued for r in baseline]
        p_rescued = [r.people_rescued for r in ppo]
        axes[1, 0].hist([b_rescued, p_rescued], label=['Baseline', 'PPO'],
                       alpha=0.7, bins=range(max(b_rescued + p_rescued) + 2),
                       edgecolor='black')
        axes[1, 0].set_xlabel('People Rescued')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('People Rescued Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: PPO nearest frontier selection
    if ppo:
        nearest_pct = []
        for r in ppo:
            if r.total_frontier_picks > 0:
                nearest_pct.append(r.nearest_frontier_picks / r.total_frontier_picks * 100)
        
        if nearest_pct:
            axes[1, 1].hist(nearest_pct, bins=20, alpha=0.7, 
                          edgecolor='black', color='#4ECDC4')
            axes[1, 1].axvline(np.mean(nearest_pct), color='red', 
                             linestyle='--', linewidth=2, label=f'Mean: {np.mean(nearest_pct):.1f}%')
            axes[1, 1].set_xlabel('% Nearest Frontier Selected')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('PPO: Nearest Frontier Selection Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Baseline vs PPO agents")
    parser.add_argument("--ppo-model", type=str, default=None,
                       help="Path to trained PPO model (optional)")
    parser.add_argument("--num-episodes", type=int, default=50,
                       help="Number of episodes per agent")
    parser.add_argument("--max-steps", type=int, default=2000,
                       help="Max steps per episode")
    parser.add_argument("--seed-start", type=int, default=1000,
                       help="Starting seed")
    parser.add_argument("--save-prefix", type=str, default="eval",
                       help="Prefix for output files")
    parser.add_argument("--verbose", action="store_true",
                       help="Print episode progress")
    
    args = parser.parse_args()
    
    # Environment config - matches your PPO training setup
    env_config = {
        'room_size': 6,
        'num_rows': 3,
        'num_cols': 3,
        'num_people': 3,
        'num_exits': 2,
        'num_collapsed_floors': 8,
        'agent_view_size': 3,
    }
    
    print("Environment Configuration:")
    for key, val in env_config.items():
        print(f"  {key}: {val}")
    
    results = []
    seeds = list(range(args.seed_start, args.seed_start + args.num_episodes))
    
    # Evaluate baseline
    print("\n" + "="*70)
    print("EVALUATING BASELINE AGENT")
    print("="*70)
    
    for i, seed in enumerate(seeds):
        if args.verbose or i % 10 == 0:
            print(f"\nBaseline Episode {i+1}/{args.num_episodes} (seed={seed})")
        
        try:
            result = run_baseline_episode(env_config, seed, i, args.max_steps, args.verbose)
            results.append(result)
            
            if args.verbose or i % 10 == 0:
                print(f"  Result: reward={result.total_reward:.1f}, "
                      f"rescued={result.people_rescued}, success={result.success}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Evaluate PPO if model provided
    if args.ppo_model:
        print(f"\nChecking for PPO model at: {args.ppo_model}")
        print(f"Current directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(args.ppo_model)}")
        
        if not os.path.exists(args.ppo_model):
            print("\n" + "="*70)
            print(f"ERROR: PPO model not found at {args.ppo_model}")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("EVALUATING PPO AGENT")
            print("="*70)
            print(f"Loading model from {args.ppo_model}")
            
            try:
                ppo_model = PPO.load(args.ppo_model)
                
                for i, seed in enumerate(seeds):
                    if args.verbose or i % 10 == 0:
                        print(f"\nPPO Episode {i+1}/{args.num_episodes} (seed={seed})")
                    
                    try:
                        result = run_ppo_episode(ppo_model, env_config, seed, i, 
                                                args.max_steps, args.verbose)
                        results.append(result)
                        
                        if args.verbose or i % 10 == 0:
                            nearest_pct = 0
                            if result.total_frontier_picks > 0:
                                nearest_pct = result.nearest_frontier_picks / result.total_frontier_picks * 100
                            print(f"  Result: reward={result.total_reward:.1f}, "
                                  f"rescued={result.people_rescued}, success={result.success}, "
                                  f"nearest={nearest_pct:.1f}%")
                    except Exception as e:
                        print(f"  ERROR in episode: {e}")
                        import traceback
                        traceback.print_exc()
            except Exception as e:
                print(f"ERROR loading PPO model: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("\n" + "="*70)
        print("No PPO model specified (use --ppo-model to evaluate PPO)")
        print("="*70)
    
    # Compute statistics
    stats = compute_statistics(results)
    print_statistics(stats)
    
    # Generate plots
    plot_file = f"{args.save_prefix}_comparison.png"
    plot_results(results, save_path=plot_file)
    
    # Save results
    json_file = f"{args.save_prefix}_results.json"
    data = {
        'env_config': env_config,
        'timestamp': datetime.now().isoformat(),
        'results': [asdict(r) for r in results],
        'statistics': stats
    }
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nResults saved to {json_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()