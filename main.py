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
    
    # Plot stockouts
    stockouts = [m['total_stockouts'] for m in metrics]
    ax4.plot(stockouts)
    ax4.set_title('Stockouts per Episode')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Number of Stockouts')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_progress.png'))
    plt.close()

def train(env, agent, num_episodes=100):
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
            'total_stockouts': 0,
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
            
            # Track stockouts and service level
            for sku_id in env.skus:
                stockout = info['stockouts'][sku_id]
                episode_metrics['total_stockouts'] += stockout
            
            # Track inventory levels
            for sku_id in env.skus:
                if sku_id not in episode_metrics['warehouse_levels']:
                    episode_metrics['warehouse_levels'][sku_id] = []
                if sku_id not in episode_metrics['retail_levels']:
                    episode_metrics['retail_levels'][sku_id] = []
                
                episode_metrics['warehouse_levels'][sku_id].append(
                    int(info['warehouse_stock'][sku_id])  # Convert to native int
                )
                episode_metrics['retail_levels'][sku_id].append(
                    int(info['retail_stock'][sku_id])  # Convert to native int
                )
            
            # Track supplier reliability
            episode_metrics['supplier_reliability'] = {
                k: float(v) for k, v in info['supplier_reliability'].items()  # Convert to native float
            }
            
            state = next_state
        
        # Get final service level from environment info
        episode_metrics['service_level'] = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Store episode results (convert to native types)
        rewards_history.append(float(episode_reward))
        td_errors.append(float(agent.get_average_td_error()))
        metrics_history.append(episode_metrics)
        episode_lengths.append(int(episode_steps))
        
        # Update progress bar
        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(rewards_history[-10:]))
            avg_service_level = float(np.mean([m['service_level'] for m in metrics_history[-10:]]))
            pbar.set_postfix({
                'Reward': f'{avg_reward:.2f}',
                'Service Level': f'{avg_service_level:.1f}%'
            })
    
    return rewards_history, metrics_history, episode_lengths, td_errors

def evaluate(env, agent, num_episodes=100):
    """Evaluate the trained agent"""
    eval_metrics = {
        'rewards': [],
        'service_levels': [],
        'stockouts': [],
        'episode_lengths': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        total_stockouts = 0
        
        while not done:
            action = agent.get_action(state, env)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            # Track metrics
            total_stockouts += sum(info['stockouts'].values())
            
            state = next_state
        
        # Get service level from environment info
        service_level = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Store episode results (convert to native types)
        eval_metrics['rewards'].append(float(episode_reward))
        eval_metrics['service_levels'].append(float(service_level))
        eval_metrics['stockouts'].append(int(total_stockouts))
        eval_metrics['episode_lengths'].append(int(episode_steps))
    
    return eval_metrics

def main():
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = InventoryEnvironment()
    agent = TDAgent(
        action_space=env.action_space,
        learning_rate=0.05,
        discount_factor=0.99,
        epsilon=1.0
    )
    
    # Training parameters
    num_episodes = 2000
    eval_interval = 100
    
    print("Starting training...")
    print(f"Training will run for {num_episodes} episodes")
    
    rewards_history, metrics_history, episode_lengths, td_errors = train(
        env, 
        agent,
        num_episodes=num_episodes
    )
    
    # Plot training progress with seasonal analysis
    plot_training_progress(rewards_history, td_errors, metrics_history, results_dir)
    
    # Additional seasonal analysis plots
    plot_seasonal_analysis(metrics_history, results_dir)
    
    # Save training results (convert to serializable format)
    training_results = {
        'rewards': [float(r) for r in rewards_history],
        'td_errors': [float(e) for e in td_errors],
        'metrics': convert_to_serializable(metrics_history),
        'episode_lengths': [int(l) for l in episode_lengths]
    }
    with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
        json.dump(training_results, f, indent=4)
    
    print("\nEvaluating agent...")
    print("Running extended evaluation over multiple seasonal cycles...")
    eval_metrics = evaluate(env, agent, num_episodes=500)  # Longer evaluation period

    
    
    # Save evaluation results
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(convert_to_serializable(eval_metrics), f, indent=4)
    
    # Print final performance metrics
    print("\nFinal Performance Metrics:")
    print(f"Average Reward: {float(np.mean(eval_metrics['rewards'])):.2f}")
    print(f"Average Service Level: {float(np.mean(eval_metrics['service_levels'])):.2f}%")
    print(f"Average Stockouts per Episode: {float(np.mean(eval_metrics['stockouts'])):.2f}")
    print(f"Average Episode Length: {float(np.mean(eval_metrics['episode_lengths'])):.2f}")
    
    # Save trained agent
    agent.save(os.path.join(results_dir, 'trained_agent.npy'))

def plot_seasonal_analysis(metrics_history, save_dir):
    """Plot additional analysis of seasonal patterns"""
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    episodes = range(len(metrics_history))
    
    # Plot warehouse stock levels for each SKU type
    for sku_type in ['Type_A', 'Type_B', 'Type_C']:
        warehouse_levels = [m['warehouse_levels'][sku_type][-1] for m in metrics_history]
        ax1.plot(episodes, warehouse_levels, label=f'{sku_type} Warehouse')
    ax1.set_title('Warehouse Stock Levels')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Stock Level')
    ax1.legend()
    
    
    # Plot retail stock levels for each SKU type
    for sku_type in ['Type_A', 'Type_B', 'Type_C']:
        retail_levels = [m['retail_levels'][sku_type][-1] for m in metrics_history]
        ax2.plot(episodes, retail_levels, label=f'{sku_type} Retail')
    ax2.set_title('Retail Stock Levels')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Stock Level')
    ax2.legend()
    
    # Plot service levels over time
    service_levels = [m['service_level'] for m in metrics_history]
    ax3.plot(episodes, service_levels)
    ax3.set_title('Service Level Over Time')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Service Level (%)')
    
    # Plot supplier reliability
    for supplier in ['Supplier_X', 'Supplier_Y']:
        reliability = [m['supplier_reliability'].get(supplier, 0) for m in metrics_history]
        ax4.plot(episodes, reliability, label=supplier)
    ax4.set_title('Supplier Reliability')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reliability')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seasonal_analysis.png'))
    plt.close()

#Call the analyze_performance.py script to generate additional performance metrics
def analyze_performance(metrics, save_dir):
    """Analyze performance metrics and generate visualizations"""
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot 6-month analysis
    # plot_six_month_analysis(metrics, save_dir)  # Removed because function is not defined
    
    # Save metrics to JSON
    with open(os.path.join(save_dir, 'performance_metrics.json'), 'w') as f:
        json.dump(convert_to_serializable(metrics), f, indent=4)

def evaluate_rewards_by_product_type(env, agent, num_episodes=100):
    """
    Evaluate and aggregate rewards per product type (A, B, C) for each episode.
    Assumes env.step()'s info dict contains 'sku_rewards' as {sku_type: reward}.
    Returns a dict: { 'Type_A': [...], 'Type_B': [...], 'Type_C': [...] }
    """
    rewards_by_type = {'Type_A': [], 'Type_B': [], 'Type_C': []}

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_rewards = {'Type_A': 0.0, 'Type_B': 0.0, 'Type_C': 0.0}

        while not done:
            action = agent.get_action(state, env)
            next_state, reward, done, info = env.step(action)
            # Aggregate rewards per SKU type
            sku_rewards = info.get('sku_rewards', {})
            for sku_type in ['Type_A', 'Type_B', 'Type_C']:
                episode_rewards[sku_type] += float(sku_rewards.get(sku_type, 0.0))
            state = next_state

        for sku_type in ['Type_A', 'Type_B', 'Type_C']:
            rewards_by_type[sku_type].append(episode_rewards[sku_type])

    return rewards_by_type
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main() 