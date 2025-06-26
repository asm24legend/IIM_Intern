import numpy as np
from inventory_env import InventoryEnvironment
from q_learning_agent import TDAgent
from dqn_agent import DoubleDQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import pickle
import csv

def convert_to_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def plot_training_progress(rewards, td_errors, metrics, save_dir):
    """Plot training metrics"""
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title('Training Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot TD errors
    ax2.plot(td_errors)
    ax2.set_title('TD Errors')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average TD Error')
    
    # Plot service levels
    service_levels = [m['service_level'] for m in metrics]
    ax3.plot(service_levels)
    ax3.set_title('Service Level')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Service Level (%)')
    
    # Plot stockouts by location
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    for location in locations:
        stockouts = [m['location_stockouts'][location] for m in metrics]
        ax4.plot(stockouts, label=location)
    ax4.set_title('Stockouts per Location')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Stockouts')
    ax4.legend()
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def train(env, agent, num_episodes=100, max_steps=500):
    """Train the agent"""
    rewards_history = []
    td_errors = []
    metrics_history = []
    episode_lengths = []
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training Progress")
    
    for episode in pbar:
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_metrics = {
            'stockouts': {},
            'location_stockouts': {
                'Location_1': 0,
                'Location_2': 0,
                'Location_3': 0,
                'Retail': 0
            },
            'service_level': 0,
            'warehouse_levels': {},
            'retail_levels': {},
            'supplier_reliability': {},
            'transportation_cost': 0.0  # Track total transportation cost for this episode
        }
        
        while not done and episode_steps < max_steps:
            # Get action from agent
            action = agent.get_action(state, env)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            td_error = agent.learn(state, action, reward, next_state)
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            
            # Track stockouts by location
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_metrics['location_stockouts'][location] += stockout
                # Track retail stockouts separately
                if sku.retail_stock <= 0:
                    episode_metrics['location_stockouts']['Retail'] += stockout
            
            # Track inventory levels
            for sku_id in env.skus:
                if sku_id not in episode_metrics['warehouse_levels']:
                    episode_metrics['warehouse_levels'][sku_id] = []
                if sku_id not in episode_metrics['retail_levels']:
                    episode_metrics['retail_levels'][sku_id] = []
                
                episode_metrics['warehouse_levels'][sku_id].append(
                    int(info['warehouse_stock'][sku_id])
                )
                episode_metrics['retail_levels'][sku_id].append(
                    int(info['retail_stock'][sku_id])
                )
            
            # Track supplier reliability
            episode_metrics['supplier_reliability'] = {
                k: float(v) for k, v in info['supplier_reliability'].items()
            }
            
            # Track transportation cost
            episode_metrics['transportation_cost'] += sum(info.get('transportation_costs', {}).values())
            
            state = next_state
        
        # Get final service level from environment info
        episode_metrics['service_level'] = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Store episode results
        rewards_history.append(float(episode_reward))
        td_errors.append(float(agent.get_average_td_error()))
        metrics_history.append(episode_metrics)
        episode_lengths.append(int(episode_steps))
        
        # Update progress bar with location-specific stockouts
        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(rewards_history[-10:]))
            avg_service_level = float(np.mean([
                m['service_level'] for m in metrics_history[-10:]
            ]))
            # Calculate average stockouts by location
            avg_stockouts = {
                location: np.mean([m['location_stockouts'][location] for m in metrics_history[-10:]])
                for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']
            }
            pbar.set_postfix({
                'Reward': f'{avg_reward:.2f}',
                'Service Level': f'{avg_service_level:.1f}%',
                'L1 Stockouts': f'{avg_stockouts["Location_1"]:.1f}',
                'L2 Stockouts': f'{avg_stockouts["Location_2"]:.1f}',
                'L3 Stockouts': f'{avg_stockouts["Location_3"]:.1f}',
                'Retail Stockouts': f'{avg_stockouts["Retail"]:.1f}'
            })
    
    return rewards_history, metrics_history, episode_lengths, td_errors

def evaluate(env, agent, num_episodes=100, max_steps=500, episode_lengths_override=None):
    """Evaluate the trained agent (optionally using provided episode lengths per episode)"""
    eval_metrics = {
        'rewards': [],
        'service_levels': [],
        'location_stockouts': {
            'Location_1': [],
            'Location_2': [],
            'Location_3': [],
            'Retail': []
        },
        'episode_lengths': [],
        'transportation_costs': []  # Track transportation cost per episode
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_stockouts = {
            'Location_1': 0,
            'Location_2': 0,
            'Location_3': 0,
            'Retail': 0
        }
        # Determine the max steps for this episode
        this_max_steps = max_steps
        if episode_lengths_override is not None and episode < len(episode_lengths_override):
            this_max_steps = episode_lengths_override[episode]
        while not done and episode_steps < this_max_steps:
            action = agent.get_action(state, env, greedy=True)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_stockouts[location] += stockout
                if sku.retail_stock <= 0:
                    episode_stockouts['Retail'] += stockout
            if 'transportation_costs' in info:
                if len(eval_metrics['transportation_costs']) <= episode:
                    eval_metrics['transportation_costs'].append(0.0)
                eval_metrics['transportation_costs'][episode] += sum(info['transportation_costs'].values())
            state = next_state
        service_level = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        eval_metrics['rewards'].append(float(episode_reward))
        eval_metrics['service_levels'].append(float(service_level))
        for location in episode_stockouts:
            eval_metrics['location_stockouts'][location].append(episode_stockouts[location])
        eval_metrics['episode_lengths'].append(int(episode_steps))
    return eval_metrics

def plot_cumulative_reward(eval_metrics, save_dir):
    """Plot the cumulative reward as a function of the number of steps."""
    rewards = eval_metrics['rewards']
    cumulative_rewards = np.cumsum(rewards)
    steps = np.arange(1, len(cumulative_rewards) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_rewards, label='Cumulative Reward')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'cumulative_reward.png'))
    plt.close()

def plot_location_stockouts(eval_metrics, save_dir):
    """Plot stockouts by location"""
    plt.figure(figsize=(12, 6))
    
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    for location in locations:
        stockouts = eval_metrics['location_stockouts'][location]
        plt.plot(stockouts, label=location)
    
    plt.xlabel('Episode')
    plt.ylabel('Number of Stockouts')
    plt.title('Stockouts by Location')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'location_stockouts.png'))
    plt.close()

def export_q_table_to_csv(q_table_path, csv_path):
    """
    Load a Q-table from a .npy file and export it to a readable CSV file.
    """
    q_table = np.load(q_table_path, allow_pickle=True).item()
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['State', 'Action', 'Q-value'])
        for key, value in q_table.items():
            # key is usually a tuple: (state, action)
            if isinstance(key, (tuple, list)) and len(key) == 2:
                state, action = key
            else:
                state, action = key, ""
            writer.writerow([str(state), str(action), value])

class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
    def get_action(self, state, env=None, greedy=False):
        # Basic EOQ policy: if warehouse stock < reorder point, order up to EOQ, else order 0. Lead time reduction = 0.
        num_skus = len(self.env.skus)
        order_quantities = np.zeros(num_skus, dtype=np.int32)
        lead_time_reductions = np.zeros(num_skus, dtype=np.int32)
        for i, sku_id in enumerate(self.env.skus):
            sku = self.env.skus[sku_id]
            warehouse_stock = state[2*i]
            reorder_point = sku.reorder_point
            eoq = sku.eoq
            if warehouse_stock < reorder_point:
                order_quantities[i] = eoq
            else:
                order_quantities[i] = 0
            lead_time_reductions[i] = 0
        return np.concatenate([order_quantities, lead_time_reductions])
    def learn(self, *args, **kwargs):
        return 0.0
    def get_average_td_error(self):
        return 0.0
    def save(self, path):
        pass

def main():
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment and agents
    env = InventoryEnvironment()
    
    # Double Q-learning agent
    td_agent = TDAgent(
        action_space=env.action_space,
        discount_factor=0.99
    )
    
    # Double DQN agent
    dqn_agent = DoubleDQNAgent(
        action_space=env.action_space,
        discount_factor=0.99
    )
    
    # Training parameters
    num_episodes = 10000 
    eval_interval = 100
    
    print("Starting training for Double Q-learning agent...")
    print(f"Training will run for {num_episodes} episodes")
    
    # Train Double Q-learning agent
    td_rewards_history, td_metrics_history, td_episode_lengths, td_errors = train(
        env, 
        td_agent,
        num_episodes=num_episodes,
        max_steps=500
    )
    
    # Plot training progress for Double Q-learning
    plot_training_progress(td_rewards_history, td_errors, td_metrics_history, results_dir)
    
    print("\nStarting training for Double DQN agent...")
    print(f"Training will run for {num_episodes} episodes")
    
    # Train Double DQN agent
    dqn_rewards_history, dqn_metrics_history, dqn_episode_lengths, dqn_errors = train(
        env, 
        dqn_agent,
        num_episodes=num_episodes,
        max_steps=500
    )
    
    # Plot training progress for Double DQN
    os.makedirs(results_dir, exist_ok=True)
    plot_training_progress(dqn_rewards_history, dqn_errors, dqn_metrics_history, 
                          os.path.join(results_dir, 'dqn_training_progress.png'))
    
    # Evaluate all agents
    print("\nEvaluating Double Q-learning agent...")
    td_eval_metrics = evaluate(env, td_agent, num_episodes=100, max_steps=500)
    
    print("\nEvaluating Double DQN agent...")
    dqn_eval_metrics = evaluate(env, dqn_agent, num_episodes=100, max_steps=500)
    
    print("\nEvaluating random agent...")
    random_agent = RandomAgent(env)
    random_metrics = evaluate(env, random_agent, num_episodes=100, max_steps=500, 
                             episode_lengths_override=td_eval_metrics['episode_lengths'])
    
    # Plot evaluation results
    plot_cumulative_reward(td_eval_metrics, results_dir)
    plot_location_stockouts(td_eval_metrics, results_dir)
    
    # Plot histogram comparing average rewards for all three agents
    plt.figure(figsize=(10, 6))
    plt.hist(td_eval_metrics['rewards'], bins=20, alpha=0.7, label='Double Q-learning Agent')
    plt.hist(dqn_eval_metrics['rewards'], bins=20, alpha=0.7, label='Double DQN Agent')
    plt.hist(random_metrics['rewards'], bins=20, alpha=0.7, label='Random Agent')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution: Double Q-learning vs Double DQN vs Random Agent')
    plt.legend()
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'reward_histogram.png'))
    plt.close()
    
    # Save evaluation metrics
    td_eval_metrics_serializable = convert_to_serializable(td_eval_metrics)
    dqn_eval_metrics_serializable = convert_to_serializable(dqn_eval_metrics)
    random_metrics_serializable = convert_to_serializable(random_metrics)
    
    with open(os.path.join(results_dir, 'td_evaluation_metrics.json'), 'w') as f:
        json.dump(td_eval_metrics_serializable, f, indent=4)
    with open(os.path.join(results_dir, 'dqn_evaluation_metrics.json'), 'w') as f:
        json.dump(dqn_eval_metrics_serializable, f, indent=4)
    with open(os.path.join(results_dir, 'random_evaluation_metrics.json'), 'w') as f:
        json.dump(random_metrics_serializable, f, indent=4)
    
    print(f"\nResults saved in {results_dir}")
    print("\nEvaluation Results:")
    print(f"Average Reward (Double Q-learning): {np.mean(td_eval_metrics['rewards']):.2f}")
    print(f"Average Reward (Double DQN): {np.mean(dqn_eval_metrics['rewards']):.2f}")
    print(f"Average Reward (Random): {np.mean(random_metrics['rewards']):.2f}")
    print(f"Average Service Level (Double Q-learning): {np.mean(td_eval_metrics['service_levels']):.1f}%")
    print(f"Average Service Level (Double DQN): {np.mean(dqn_eval_metrics['service_levels']):.1f}%")
    print(f"Average Service Level (Random): {np.mean(random_metrics['service_levels']):.1f}%")
    print(f"Average Episode Length (Double Q-learning): {np.mean(td_eval_metrics['episode_lengths']):.2f}")
    print(f"Average Episode Length (Double DQN): {np.mean(dqn_eval_metrics['episode_lengths']):.2f}")
    print(f"Average Episode Length (Random): {np.mean(random_metrics['episode_lengths']):.2f}")
    
    print("\nAverage Stockouts by Location:")
    for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']:
        avg_stockouts_td = np.mean(td_eval_metrics['location_stockouts'][location])
        avg_stockouts_dqn = np.mean(dqn_eval_metrics['location_stockouts'][location])
        avg_stockouts_random = np.mean(random_metrics['location_stockouts'][location])
        print(f"{location} (Double Q-learning): {avg_stockouts_td:.2f}")
        print(f"{location} (Double DQN): {avg_stockouts_dqn:.2f}")
        print(f"{location} (Random): {avg_stockouts_random:.2f}")
    
    if 'transportation_costs' in td_eval_metrics and len(td_eval_metrics['transportation_costs']) > 0:
        print(f"\nAverage Transportation Cost per Episode (Double Q-learning): {np.mean(td_eval_metrics['transportation_costs']):.2f}")
    if 'transportation_costs' in dqn_eval_metrics and len(dqn_eval_metrics['transportation_costs']) > 0:
        print(f"Average Transportation Cost per Episode (Double DQN): {np.mean(dqn_eval_metrics['transportation_costs']):.2f}")
    if 'transportation_costs' in random_metrics and len(random_metrics['transportation_costs']) > 0:
        print(f"Average Transportation Cost per Episode (Random): {np.mean(random_metrics['transportation_costs']):.2f}")

    # Plot bar chart comparing metrics for all three agents
    labels = ['Service Level', 'Transp. Cost', 'Stockouts L1', 'Stockouts L2', 'Stockouts L3', 'Stockouts Retail', 'Ep. Length']
    
    td_values = [
        np.mean(td_eval_metrics['service_levels']),
        np.mean(td_eval_metrics['transportation_costs']) if 'transportation_costs' in td_eval_metrics else 0,
        np.mean(td_eval_metrics['location_stockouts']['Location_1']),
        np.mean(td_eval_metrics['location_stockouts']['Location_2']),
        np.mean(td_eval_metrics['location_stockouts']['Location_3']),
        np.mean(td_eval_metrics['location_stockouts']['Retail']),
        np.mean(td_eval_metrics['episode_lengths'])
    ]
    
    dqn_values = [
        np.mean(dqn_eval_metrics['service_levels']),
        np.mean(dqn_eval_metrics['transportation_costs']) if 'transportation_costs' in dqn_eval_metrics else 0,
        np.mean(dqn_eval_metrics['location_stockouts']['Location_1']),
        np.mean(dqn_eval_metrics['location_stockouts']['Location_2']),
        np.mean(dqn_eval_metrics['location_stockouts']['Location_3']),
        np.mean(dqn_eval_metrics['location_stockouts']['Retail']),
        np.mean(dqn_eval_metrics['episode_lengths'])
    ]
    
    random_values = [
        np.mean(random_metrics['service_levels']),
        np.mean(random_metrics['transportation_costs']) if 'transportation_costs' in random_metrics else 0,
        np.mean(random_metrics['location_stockouts']['Location_1']),
        np.mean(random_metrics['location_stockouts']['Location_2']),
        np.mean(random_metrics['location_stockouts']['Location_3']),
        np.mean(random_metrics['location_stockouts']['Retail']),
        np.mean(random_metrics['episode_lengths'])
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(15, 6))
    plt.bar(x - width, td_values, width, label='Double Q-learning')
    plt.bar(x, dqn_values, width, label='Double DQN')
    plt.bar(x + width, random_values, width, label='Random')
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Comparison of Key Metrics: Double Q-learning vs Double DQN vs Random Agent')
    plt.legend()
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'metrics_comparison_bar.png'))
    plt.close()

    # Save trained agents
    td_agent.save(os.path.join(results_dir, 'trained_td_agent'))
    dqn_agent.save(os.path.join(results_dir, 'trained_dqn_agent'))

    # Export Q-table to CSV (use Q-table A for export)
    q_table_path = os.path.join(results_dir, 'trained_td_agent_A.pkl')
    csv_path = os.path.join(results_dir, 'td_q_table.csv')
    # Load Q-table A from pickle for export
    with open(q_table_path, 'rb') as f:
        q_table_A = pickle.load(f)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['State', 'Action', 'Q-value'])
        for key, value in q_table_A.items():
            if isinstance(key, (tuple, list)) and len(key) == 2:
                state, action = key
            else:
                state, action = key, ""
            writer.writerow([str(state), str(action), value])
    print(f"Q-table (A) exported to {csv_path}")

    # Print average reward per step for all agents
    avg_reward_per_step_td = np.mean(td_eval_metrics['rewards']) / np.mean(td_eval_metrics['episode_lengths'])
    avg_reward_per_step_dqn = np.mean(dqn_eval_metrics['rewards']) / np.mean(dqn_eval_metrics['episode_lengths'])
    avg_reward_per_step_random = np.mean(random_metrics['rewards']) / np.mean(random_metrics['episode_lengths'])
    print(f"Average Reward per Step (Double Q-learning): {avg_reward_per_step_td:.2f}")
    print(f"Average Reward per Step (Double DQN): {avg_reward_per_step_dqn:.2f}")
    print(f"Average Reward per Step (Random): {avg_reward_per_step_random:.2f}")

    # --- BENCHMARK: Fixed Demand ---
    print("\nRunning fixed demand benchmark...")
    benchmark_config = env.config.copy()
    benchmark_config['use_fixed_demand'] = True
    benchmark_config['fixed_daily_demand'] = 5  # You can adjust this value
    env_benchmark = InventoryEnvironment(config=benchmark_config)
    agent_benchmark = TDAgent(
        action_space=env_benchmark.action_space,
        discount_factor=0.99
    )
    rewards_bench, metrics_bench, episode_lengths_bench, td_errors_bench = train(
        env_benchmark,
        agent_benchmark,
        num_episodes=5000,
        max_steps=500
    )
    print("\nEvaluating benchmark agent...")
    eval_metrics_bench = evaluate(env_benchmark, agent_benchmark, num_episodes=100, max_steps=500)
    print("\nBenchmark Results (Fixed Demand):")
    print(f"Average Reward: {np.mean(eval_metrics_bench['rewards']):.2f}")
    print(f"Average Service Level: {np.mean(eval_metrics_bench['service_levels']):.1f}%")
    print(f"Average Episode Length: {np.mean(eval_metrics_bench['episode_lengths']):.2f}")
    print("Average Stockouts by Location:")
    for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']:
        avg_stockouts = np.mean(eval_metrics_bench['location_stockouts'][location])
        print(f"{location}: {avg_stockouts:.2f}")
    # Save benchmark results
    eval_metrics_bench_serializable = convert_to_serializable(eval_metrics_bench)
    with open(os.path.join(results_dir, 'benchmark_evaluation_metrics.json'), 'w') as f:
        json.dump(eval_metrics_bench_serializable, f, indent=4)

    print("Environment initialized with constant lead times per SKU.")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main() 