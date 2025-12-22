import argparse
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from SAREnv import SAREnv
from ppo_frontier_wrapper import FrontierSelectionEnv
from search_agent import SearchAgent
from rescue_agent import RescueAgent
from dataclasses import dataclass, asdict
from typing import Dict, List
from datetime import datetime


@dataclass
class EpisodeResult:
    """Episode results with frontier selection tracking."""
    episode_id: int
    agent_type: str
    total_reward: float
    episode_length: int
    people_rescued: int
    success: bool
    seed: int
    # PPO-specific
    valid_picks: int = 0
    valid_nearest_picks: int = 0
    fallback_picks: int = 0


def _action_toward(env, next_pos):
    """Helper to determine action toward next position."""
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
    
    diff = (desired_dir - env.agent_dir) % 4
    return env.actions.right if diff == 1 else env.actions.left


def _adjacent_walkable(rescue_agent, pos):
    """Helper to find adjacent walkable cells."""
    adj = []
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < rescue_agent.env.width and 0 <= ny < rescue_agent.env.height:
            if rescue_agent._is_safe(nx, ny):
                adj.append((nx, ny))
    return adj


def run_baseline_episode(env_config: Dict, seed: int, episode_id: int, 
                        max_steps: int = 2000) -> EpisodeResult:
    """Run baseline greedy agent with search and rescue."""
    env = SAREnv(**env_config, render_mode=None)
    obs, _ = env.reset(seed=seed)
    
    search_agent = SearchAgent(env)
    search_agent.update_map(obs, env.agent_pos, env.agent_dir)
    rescue_agent = RescueAgent(env, search_agent)
    
    total_reward = 0.0
    episode_length = 0
    mode = "search"
    
    for _ in range(max_steps):
        if env.people_rescued >= env.num_people:
            break
        
        # Check for known people
        known_people = search_agent.get_known_people_locations()
        
        if known_people and mode == "search":
            mode = "rescue"
        
        # Get action based on mode
        if mode == "rescue" and known_people:
            person_pos = known_people[0]
            
            # If carrying, go to exit
            if env.carrying is not None and env.carrying.type == "ball":
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
                # Navigate to person and pickup
                dx = abs(person_pos[0] - env.agent_pos[0])
                dy = abs(person_pos[1] - env.agent_pos[1])
                
                if dx + dy == 1:
                    # Adjacent - face and pickup
                    dx_dir = person_pos[0] - env.agent_pos[0]
                    dy_dir = person_pos[1] - env.agent_pos[1]
                    
                    desired_dir = -1
                    if dx_dir == 1:
                        desired_dir = 0
                    elif dy_dir == 1:
                        desired_dir = 1
                    elif dx_dir == -1:
                        desired_dir = 2
                    elif dy_dir == -1:
                        desired_dir = 3
                    
                    if desired_dir != -1 and desired_dir != env.agent_dir:
                        diff = (desired_dir - env.agent_dir) % 4
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
            # Search mode
            mode = "search"
            action = search_agent.get_action(env.agent_pos, env.agent_dir)
            if action is None:
                break
        
        obs, reward, terminated, truncated, _ = env.step(action)
        search_agent.update_map(obs, env.agent_pos, env.agent_dir)
        
        total_reward += reward
        episode_length += 1
        
        if terminated or truncated:
            break
    
    success = env.people_rescued >= env.num_people
    people_rescued = env.people_rescued
    env.close()
    
    return EpisodeResult(
        episode_id=episode_id,
        agent_type="baseline",
        total_reward=total_reward,
        episode_length=episode_length,
        people_rescued=people_rescued,
        success=success,
        seed=seed
    )


def run_ppo_episode(ppo_model, env_config: Dict, seed: int, episode_id: int,
                   max_steps: int = 2000, verbose: bool = False) -> EpisodeResult:
    """Run PPO agent with frontier tracking."""
    base_env = SAREnv(**env_config, render_mode=None)
    env = FrontierSelectionEnv(base_env, frontier_limit=64)
    obs, _ = env.reset(seed=seed)
    
    total_reward = 0.0
    episode_length = 0
    valid_picks = 0
    valid_nearest_picks = 0
    fallback_picks = 0
    
    last_rescued = 0
    steps_since_rescue = 0
    
    for step in range(max_steps):
        action, _ = ppo_model.predict(obs, deterministic=True)
        action = int(action)
        
        # Track frontier selection
        frontiers = obs['frontiers']
        num_valid = int(np.sum(frontiers[:, 0] >= 0))
        
        if num_valid > 0:
            is_valid = action < num_valid
            
            if is_valid:
                valid_picks += 1
                if action == 0:  # Nearest frontier
                    valid_nearest_picks += 1
            else:
                fallback_picks += 1
            
            if verbose and (valid_picks + fallback_picks) <= 5:
                print(f"Pick: action={action}, valid_frontiers={num_valid}, "
                      f"is_valid={is_valid}, nearest={action==0}")
        
        obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward += reward
        episode_length += 1
        
        # Timeout if stuck
        if base_env.people_rescued > last_rescued:
            last_rescued = base_env.people_rescued
            steps_since_rescue = 0
        else:
            steps_since_rescue += 1
        
        if steps_since_rescue > 500:
            truncated = True
        
        if terminated or truncated:
            break
    
    success = base_env.people_rescued >= base_env.num_people
    people_rescued = base_env.people_rescued
    
    if verbose:
        total = valid_picks + fallback_picks
        if total > 0:
            print(f"Summary: Valid={valid_picks}/{total} ({100*valid_picks/total:.1f}%), "
                  f"Valid_Nearest={valid_nearest_picks}/{valid_picks if valid_picks else 1} "
                  f"({100*valid_nearest_picks/max(1,valid_picks):.1f}%)")
    
    env.close()
    
    return EpisodeResult(
        episode_id=episode_id,
        agent_type="ppo",
        total_reward=total_reward,
        episode_length=episode_length,
        people_rescued=people_rescued,
        success=success,
        seed=seed,
        valid_picks=valid_picks,
        valid_nearest_picks=valid_nearest_picks,
        fallback_picks=fallback_picks
    )


def compute_metrics(results: List[EpisodeResult]) -> Dict:
    """Compute metrics for agents."""
    baseline = [r for r in results if r.agent_type == "baseline"]
    ppo = [r for r in results if r.agent_type == "ppo"]
    
    def metrics(group):
        if not group:
            return {}
        return {
            'count': len(group),
            'reward_mean': float(np.mean([r.total_reward for r in group])),
            'reward_std': float(np.std([r.total_reward for r in group])),
            'length_mean': float(np.mean([r.episode_length for r in group])),
            'length_std': float(np.std([r.episode_length for r in group])),
            'rescued_mean': float(np.mean([r.people_rescued for r in group])),
            'success_rate': float(np.mean([r.success for r in group]) * 100),
        }
    
    result = {
        'baseline': metrics(baseline),
        'ppo': metrics(ppo)
    }
    
    # Add PPO frontier analysis
    if ppo:
        total_valid = sum(r.valid_picks for r in ppo)
        total_nearest = sum(r.valid_nearest_picks for r in ppo)
        total_fallback = sum(r.fallback_picks for r in ppo)
        
        result['ppo']['valid_picks'] = total_valid
        result['ppo']['valid_nearest_picks'] = total_nearest
        result['ppo']['fallback_picks'] = total_fallback
        if total_valid > 0:
            result['ppo']['nearest_percentage'] = float(100 * total_nearest / total_valid)
    
    return result


def print_metrics(metrics: Dict):
    """Print formatted metrics after calcuation."""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    if metrics.get('baseline'):
        b = metrics['baseline']
        print("\nBASELINE AGENT:")
        print(f"Episodes: {b['count']}")
        print(f"Reward: {b['reward_mean']:.1f} +/- {b['reward_std']:.1f}")
        print(f"Episode Length: {b['length_mean']:.1f} +/- {b['length_std']:.1f}")
        print(f"People Rescued: {b['rescued_mean']:.2f}")
        print(f"Success Rate: {b['success_rate']:.1f}%")
    
    if metrics.get('ppo'):
        p = metrics['ppo']
        print("\nPPO AGENT:")
        print(f"Episodes: {p['count']}")
        print(f"Reward: {p['reward_mean']:.1f} +/- {p['reward_std']:.1f}")
        print(f"Episode Length: {p['length_mean']:.1f} +/- {p['length_std']:.1f}")
        print(f"People Rescued: {p['rescued_mean']:.2f}")
        print(f"Success Rate: {p['success_rate']:.1f}%")
        
        if 'valid_picks' in p:
            total = p['valid_picks'] + p['fallback_picks']
            print(f"\n  FRONTIER SELECTION:")
            print(f"Total: {total}")
            print(f"Valid: {p['valid_picks']} ({100*p['valid_picks']/total:.1f}%)")
            print(f"Fallback: {p['fallback_picks']} ({100*p['fallback_picks']/total:.1f}%)")
            if p['valid_picks'] > 0:
                print(f"Valid to Nearest: {p['valid_nearest_picks']}/{p['valid_picks']} "
                      f"({p['nearest_percentage']:.1f}%)")
    
    if metrics.get('baseline') and metrics.get('ppo'):
        print("\nCOMPARISON:")
        reward_diff = metrics['ppo']['reward_mean'] - metrics['baseline']['reward_mean']
        success_diff = metrics['ppo']['success_rate'] - metrics['baseline']['success_rate']
        print(f"Reward Difference: {reward_diff:+.1f} (PPO - Baseline)")
        print(f"Success Rate Difference: {success_diff:+.1f}% (PPO - Baseline)")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline vs PPO agents")
    parser.add_argument("--ppo-model", type=str, default="ppo_frontier.zip")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--save-prefix", type=str, default="eval")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    # Environment configuration
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
    
    # Evaluate Baseline
    print("\n" + "="*70)
    print("Evaluating Baseline Agent")
    print("="*70)
    
    for i, seed in enumerate(seeds):
        if i % 10 == 0:
            print(f"Episode {i+1}/{args.num_episodes}")
        
        result = run_baseline_episode(env_config, seed, i, args.max_steps)
        results.append(result)
    
    # Evaluate PPO
    if os.path.exists(args.ppo_model):
        print("\n" + "="*70)
        print("Evaluating PPO Agent")
        print("="*70)
        
        ppo_model = PPO.load(args.ppo_model)
        
        for i, seed in enumerate(seeds):
            if args.verbose or i % 10 == 0:
                print(f"\nEpisode {i+1}/{args.num_episodes} (seed={seed})")
            
            result = run_ppo_episode(ppo_model, env_config, seed, i, 
                                    args.max_steps, args.verbose)
            results.append(result)
    else:
        print(f"\nMissing PPO model, not found at {args.ppo_model}")
    
    # Compute and print metrics
    metrics = compute_metrics(results)
    print_metrics(metrics)

    # Generate plots
    plot_file = f"{args.save_prefix}_comparison.png"
    plot_results(results, save_path=plot_file)
    
    # Save results
    output_data = {
        'env_config': env_config,
        'timestamp': datetime.now().isoformat(),
        'results': [asdict(r) for r in results],
        'metrics': metrics
    }
    
    json_file = f"{args.save_prefix}_results.json"
    with open(json_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {json_file}")
    print("\nEvaluation complete")

def plot_results(results: List[EpisodeResult], save_path: str = "comparison.png"):
    """Generate comparison plots with detailed frontier breakdown."""
    baseline = [r for r in results if r.agent_type == "baseline"]
    ppo = [r for r in results if r.agent_type == "ppo"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline vs PPO Agent Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Reward distribution
    if baseline and ppo:
        b_rewards = [r.total_reward for r in baseline]
        p_rewards = [r.total_reward for r in ppo]
        
        all_rewards = b_rewards + p_rewards
        min_reward, max_reward = min(all_rewards), max(all_rewards)
        
        num_bins = 10
        bin_edges = np.linspace(min_reward, max_reward, num_bins + 1)
        b_counts, _ = np.histogram(b_rewards, bins=bin_edges)
        p_counts, _ = np.histogram(p_rewards, bins=bin_edges)
        
        x = np.arange(num_bins)
        width = 0.35
        
        axes[0, 0].bar(x - width/2, b_counts, width=width, 
                      label='Baseline', color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[0, 0].bar(x + width/2, p_counts, width=width,
                      label='PPO', color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        bin_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(num_bins)]
        axes[0, 0].set_xlabel('Total Reward Range', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Number of Episodes', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Reward Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: People rescued distribution
    if baseline and ppo:
        max_people = 3
        b_counts = [0] * (max_people + 1)
        p_counts = [0] * (max_people + 1)
        
        for r in baseline:
            if 0 <= r.people_rescued <= max_people:
                b_counts[r.people_rescued] += 1
        for r in ppo:
            if 0 <= r.people_rescued <= max_people:
                p_counts[r.people_rescued] += 1
        
        x = np.arange(max_people + 1)
        width = 0.35
        
        axes[1, 0].bar(x - width/2, b_counts, width, label='Baseline', 
                      color='#FF6B6B', edgecolor='black', linewidth=1.5, alpha=0.8)
        axes[1, 0].bar(x + width/2, p_counts, width, label='PPO',
                      color='#4ECDC4', edgecolor='black', linewidth=1.5, alpha=0.8)
        
        axes[1, 0].set_xlabel('Number of People Rescued', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Number of Episodes', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('People Rescued Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(['0', '1', '2', '3 (Success)'])
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")

if __name__ == "__main__":
    main()