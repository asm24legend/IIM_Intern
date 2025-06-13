import numpy as np
from inventory_env import InventoryEnvironment
from q_learning_agent import TDAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from datetime import datetime
import pickle

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
    
    # Plot stockouts by location and total stockout percentage
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    for location in locations:
        stockouts = [m['location_stockouts'][location] for m in metrics]
        ax4.plot(stockouts, label=location)
    
    # Add stockout percentage line
    stockout_percentages = [m['stockout_percentage'] for m in metrics]
    ax4_twin = ax4.twinx()
    ax4_twin.plot(stockout_percentages, 'k--', label='Stockout %')
    ax4_twin.set_ylabel('Stockout Percentage (%)')
    
    ax4.set_title('Stockouts per Location and Overall Percentage')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Stockouts')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def train(env, agent, num_episodes=10000):
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
            'total_demand': 0,  # Track total demand
            'total_stockouts': 0,  # Track total stockouts
            'service_level': 0,
            'warehouse_levels': {},
            'retail_levels': {},
            'supplier_reliability': {}
        }
        
        while not done:
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
                episode_metrics['total_stockouts'] += stockout
                # Track retail stockouts separately
                if sku.retail_stock <= 0:
                    episode_metrics['location_stockouts']['Retail'] += stockout
            
            # Track total demand
            for sku_id, sku in env.skus.items():
                episode_metrics['total_demand'] += sku.total_demand
            
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
            
            state = next_state
        
        # Get final service level from environment info
        episode_metrics['service_level'] = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Calculate stockout percentage based on actual demand and stockouts
        if episode_metrics['total_demand'] > 0:
            episode_metrics['stockout_percentage'] = (episode_metrics['total_stockouts'] / episode_metrics['total_demand']) * 100
        else:
            episode_metrics['stockout_percentage'] = 0.0
        
        # Store episode results
        rewards_history.append(float(episode_reward))
        td_errors.append(float(agent.get_average_td_error()))
        metrics_history.append(episode_metrics)
        episode_lengths.append(int(episode_steps))
        
        # Update progress bar with location-specific stockouts and percentages
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
            # Calculate average stockout percentage
            total_stockouts = sum(m['total_stockouts'] for m in metrics_history[-10:])
            total_demand = sum(m['total_demand'] for m in metrics_history[-10:])
            avg_stockout_percentage = (total_stockouts / total_demand * 100) if total_demand > 0 else 0.0
            
            pbar.set_postfix({
                'Reward': f'{avg_reward:.2f}',
                'Service Level': f'{avg_service_level:.1f}%',
                'Stockout %': f'{avg_stockout_percentage:.1f}%',
                'L1 Stockouts': f'{avg_stockouts["Location_1"]:.1f}',
                'L2 Stockouts': f'{avg_stockouts["Location_2"]:.1f}',
                'L3 Stockouts': f'{avg_stockouts["Location_3"]:.1f}',
                'Retail Stockouts': f'{avg_stockouts["Retail"]:.1f}'
            })
    
    return rewards_history, metrics_history, episode_lengths, td_errors

def evaluate(env, agent, num_episodes=100):
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
        'total_stockouts': [],
        'total_demand': [],
        'stockout_percentages': [],
        'episode_lengths': []
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
        total_stockouts = 0
        total_demand = 0
        
        while not done:
            action = agent.get_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Track stockouts by location
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_stockouts[location] += stockout
                total_stockouts += stockout
                # Track retail stockouts separately
                if sku.retail_stock <= 0:
                    episode_stockouts['Retail'] += stockout
            
            # Track total demand
            for sku_id, sku in env.skus.items():
                total_demand += sku.total_demand
            
            state = next_state
        
        # Get service level from environment info
        service_level = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Calculate stockout percentage based on actual demand and stockouts
        stockout_percentage = (total_stockouts / total_demand * 100) if total_demand > 0 else 0.0
        
        # Store episode results
        eval_metrics['rewards'].append(float(episode_reward))
        eval_metrics['service_levels'].append(float(service_level))
        for location in episode_stockouts:
            eval_metrics['location_stockouts'][location].append(episode_stockouts[location])
        eval_metrics['total_stockouts'].append(total_stockouts)
        eval_metrics['total_demand'].append(total_demand)
        eval_metrics['stockout_percentages'].append(stockout_percentage)
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
    
    # Create two y-axes
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    
    # Plot stockouts by location
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    for location in locations:
        stockouts = eval_metrics['location_stockouts'][location]
        ax1.plot(stockouts, label=location)
    
    # Plot stockout percentage
    ax2.plot(eval_metrics['stockout_percentages'], 'k--', label='Stockout %')
    
    # Set labels and title
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Number of Stockouts')
    ax2.set_ylabel('Stockout Percentage (%)')
    plt.title('Stockouts by Location and Overall Percentage')
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
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

def print_training_summary(rewards_history, metrics_history, episode_lengths, td_errors):
    """Print summary of training results"""
    # Calculate total demand and stockouts
    total_demand = sum(m['total_demand'] for m in metrics_history)
    total_stockouts = sum(m['total_stockouts'] for m in metrics_history)
    stockout_percentage = (total_stockouts / total_demand) * 100
    
    # Calculate episode statistics
    avg_episode_length = np.mean(episode_lengths)
    avg_stockouts_per_episode = total_stockouts / len(metrics_history)
    
    print("\nEpisode Information:")
    print(f"Number of Episodes: {len(metrics_history)}")
    print(f"Average Episode Length: {avg_episode_length:.1f} days")
    print(f"Average Stockouts per Episode: {avg_stockouts_per_episode:.2f}")
    print("\nNote: Each episode represents a complete inventory cycle, starting from initial stock levels")
    print("and continuing until either a terminal condition is met (e.g., retail stockout) or")
    print("the maximum episode length is reached. Each step within an episode represents one day.")
    
    print("\nOverall Metrics:")
    print(f"Average Reward: {np.mean(rewards_history):.2f}")
    print(f"Average Service Level: {np.mean([m['service_level'] for m in metrics_history]):.1f}%")
    print(f"Stockout Percentage: {stockout_percentage:.5f}%")
    print(f"Total Demand: {total_demand:.0f}")
    print(f"Total Stockouts: {total_stockouts:.0f}")


def main():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = InventoryEnvironment()
    agent = TDAgent(env.action_space)
    
    # Train the agent
    print("\nStarting Training...")
    rewards_history, metrics_history, episode_lengths, td_errors = train(env, agent)
    
    # Print training summary
    print_training_summary(rewards_history, metrics_history, episode_lengths, td_errors)
    
    # Evaluate the trained agent
    print("\nStarting Evaluation...")
    eval_metrics = evaluate(env, agent)
    
    
    # Plot training progress
    plot_training_progress(rewards_history, td_errors, metrics_history, results_dir)
    
    # Plot evaluation results
    plot_location_stockouts(eval_metrics, results_dir)
    
    # Save results
    results = {
        'training': {
            'rewards': rewards_history,
            'metrics': metrics_history,
            'episode_lengths': episode_lengths,
            'td_errors': td_errors
        },
        'evaluation': eval_metrics
    }
    
    with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main() 