import numpy as np
from inventory_env import InventoryEnvironment
from q_learning_agent import TDAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime

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

def evaluate(env, agent, num_episodes=100, max_steps=500):
    """Evaluate the trained agent"""
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
        
        while not done and episode_steps < max_steps:
            action = agent.get_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Track stockouts by location
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_stockouts[location] += stockout
                # Track retail stockouts separately
                if sku.retail_stock <= 0:
                    episode_stockouts['Retail'] += stockout
            
            # Track transportation cost for this episode
            if 'transportation_costs' in info:
                if len(eval_metrics['transportation_costs']) <= episode:
                    eval_metrics['transportation_costs'].append(0.0)
                eval_metrics['transportation_costs'][episode] += sum(info['transportation_costs'].values())
            
            state = next_state
        
        # Get service level from environment info
        service_level = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Store episode results
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
    plt.savefig(os.path.join(save_dir, 'location_stockouts.png'))
    plt.close()

def export_q_table_to_csv(q_table_path, csv_path):
    """
    Load a Q-table from a .npy file and export it to a readable CSV file.
    """
    import csv

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
    def __init__(self, action_space):
        self.action_space = action_space
    def get_action(self, state, env=None):
        return self.action_space.sample()
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
    
    # Initialize environment and agent
    env = InventoryEnvironment()
    agent = TDAgent(
        action_space=env.action_space,
        discount_factor=0.99
    )
    
    # Training parameters
    num_episodes = 1000
    eval_interval = 100
    
    print("Starting training...")
    print(f"Training will run for {num_episodes} episodes")
    
    rewards_history, metrics_history, episode_lengths, td_errors = train(
        env, 
        agent,
        num_episodes=num_episodes,
        max_steps=500
    )
    
    # Plot training progress
    plot_training_progress(rewards_history, td_errors, metrics_history, results_dir)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_metrics = evaluate(env, agent, num_episodes=100, max_steps=500)
    # Evaluate the random agent
    print("\nEvaluating random agent...")
    random_agent = RandomAgent(env.action_space)
    random_metrics = evaluate(env, random_agent, num_episodes=100, max_steps=500)
    
    # Plot evaluation results
    plot_cumulative_reward(eval_metrics, results_dir)
    plot_location_stockouts(eval_metrics, results_dir)
    # Plot histogram comparing average rewards
    plt.figure(figsize=(8, 6))
    plt.hist(eval_metrics['rewards'], bins=20, alpha=0.7, label='Q-learning Agent')
    plt.hist(random_metrics['rewards'], bins=20, alpha=0.7, label='Random Agent')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution: Q-learning vs Random Agent')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reward_histogram.png'))
    plt.close()
    
    # Save evaluation metrics
    eval_metrics_serializable = convert_to_serializable(eval_metrics)
    with open(os.path.join(results_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(eval_metrics_serializable, f, indent=4)
    
    print(f"\nResults saved in {results_dir}")
    print("\nEvaluation Results:")
    print(f"Average Reward (Q-learning): {np.mean(eval_metrics['rewards']):.2f}")
    print(f"Average Reward (Random): {np.mean(random_metrics['rewards']):.2f}")
    print(f"Average Service Level (Q-learning): {np.mean(eval_metrics['service_levels']):.1f}%")
    print(f"Average Service Level (Random): {np.mean(random_metrics['service_levels']):.1f}%")
    print(f"Average Episode Length (Q-learning): {np.mean(eval_metrics['episode_lengths']):.2f}")
    print(f"Average Episode Length (Random): {np.mean(random_metrics['episode_lengths']):.2f}")
    print("\nAverage Stockouts by Location:")
    for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']:
        avg_stockouts_q = np.mean(eval_metrics['location_stockouts'][location])
        avg_stockouts_r = np.mean(random_metrics['location_stockouts'][location])
        print(f"{location} (Q-learning): {avg_stockouts_q:.2f}")
        print(f"{location} (Random): {avg_stockouts_r:.2f}")
    if 'transportation_costs' in eval_metrics and len(eval_metrics['transportation_costs']) > 0:
        print(f"\nAverage Transportation Cost per Episode (Q-learning): {np.mean(eval_metrics['transportation_costs']):.2f}")
    if 'transportation_costs' in random_metrics and len(random_metrics['transportation_costs']) > 0:
        print(f"Average Transportation Cost per Episode (Random): {np.mean(random_metrics['transportation_costs']):.2f}")

    # Plot bar chart comparing metrics
    labels = ['Service Level', 'Transp. Cost', 'Stockouts L1', 'Stockouts L2', 'Stockouts L3', 'Stockouts Retail', 'Ep. Length']
    q_values = [
        np.mean(eval_metrics['service_levels']),
        np.mean(eval_metrics['transportation_costs']) if 'transportation_costs' in eval_metrics else 0,
        np.mean(eval_metrics['location_stockouts']['Location_1']),
        np.mean(eval_metrics['location_stockouts']['Location_2']),
        np.mean(eval_metrics['location_stockouts']['Location_3']),
        np.mean(eval_metrics['location_stockouts']['Retail']),
        np.mean(eval_metrics['episode_lengths'])
    ]
    r_values = [
        np.mean(random_metrics['service_levels']),
        np.mean(random_metrics['transportation_costs']) if 'transportation_costs' in random_metrics else 0,
        np.mean(random_metrics['location_stockouts']['Location_1']),
        np.mean(random_metrics['location_stockouts']['Location_2']),
        np.mean(random_metrics['location_stockouts']['Location_3']),
        np.mean(random_metrics['location_stockouts']['Retail']),
        np.mean(random_metrics['episode_lengths'])
    ]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, q_values, width, label='Q-learning')
    plt.bar(x + width/2, r_values, width, label='Random')
    plt.xticks(x, labels)
    plt.ylabel('Value')
    plt.title('Comparison of Key Metrics: Q-learning vs Random Agent')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_comparison_bar.png'))
    plt.close()

    # Save trained agent
    agent.save(os.path.join(results_dir, 'trained_agent.npy'))

    # Export Q-table to CSV
    q_table_path = os.path.join(results_dir, 'trained_agent.npy')
    csv_path = os.path.join(results_dir, 'q_table.csv')
    export_q_table_to_csv(q_table_path, csv_path)
    print(f"Q-table exported to {csv_path}")

    # Print average reward per step for both agents
    avg_reward_per_step_q = np.mean(eval_metrics['rewards']) / np.mean(eval_metrics['episode_lengths'])
    avg_reward_per_step_random = np.mean(random_metrics['rewards']) / np.mean(random_metrics['episode_lengths'])
    print(f"Average Reward per Step (Q-learning): {avg_reward_per_step_q:.2f}")
    print(f"Average Reward per Step (Random): {avg_reward_per_step_random:.2f}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main() 