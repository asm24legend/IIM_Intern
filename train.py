import numpy as np
import matplotlib.pyplot as plt
from inventory_env import InventoryEnvironment
from q_learning_agent import TDAgent
import json
from datetime import datetime
import os
def plot_metrics(episode_rewards, td_errors, stockouts, service_levels, filename):
    """
    Plot and save training metrics.
    
    Args:
        episode_rewards: List of rewards per episode
        td_errors: List of TD errors per episode
        stockouts: List of stockouts per episode
        service_levels: List of service levels per episode
        filename: Base filename to save plots
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, label='Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(td_errors, label='TD Errors', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('TD Error')
    plt.title('TD Errors')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(stockouts, label='Stockouts', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Stockouts')
    plt.title('Stockouts')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(service_levels, label='Service Levels', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Service Level (%)')
    plt.title('Service Levels')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{filename}_metrics.png')
    plt.close()
def save_metrics(metrics, filename):
    """
    Save training metrics to a JSON file.
    
    Args:
        metrics: Dictionary containing training metrics
        filename: Base filename to save metrics
    """
    with open(f'{filename}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
def train(num_episodes=1000, eval_interval=50):
    # Initialize environment and agent
    env = InventoryEnvironment()
    agent = TDAgent(
        action_space=env.action_space,
        learning_rate=0.05,
        discount_factor=0.99,
        epsilon=1.0
    )
    
    # Initialize metrics tracking
    metrics = {
        'episode_rewards': [],
        'td_errors': [],
        'stockouts': [],
        'service_levels': [],
        'warehouse_stock_levels': [],
        'retail_stock_levels': [],
        'supplier_reliability': []
    }
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'training_results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("Starting training...")
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_stockouts = 0
        total_demand = 0
        fulfilled_demand = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state, env)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Q-learning update (no next_action needed)
            td_error = agent.learn(state, action, reward, next_state)
            
            # Update metrics
            episode_reward += reward
            episode_stockouts += sum(info['stockouts'].values())
            
            for sku_id in env.skus:
                demand = info['stockouts'][sku_id]
                fulfilled = max(0, demand - info['stockouts'][sku_id])
                total_demand += demand
                fulfilled_demand += fulfilled
            
            state = next_state
        
        # Calculate service level
        service_level = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
        
        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['td_errors'].append(agent.get_average_td_error())
        metrics['stockouts'].append(episode_stockouts)
        metrics['service_levels'].append(service_level)
        metrics['warehouse_stock_levels'].append({sku_id: level for sku_id, level in info['warehouse_stock'].items()})
        metrics['retail_stock_levels'].append({sku_id: level for sku_id, level in info['retail_stock'].items()})
        metrics['supplier_reliability'].append(info['supplier_reliability'])
        
        # Print progress
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-eval_interval:])
            avg_service_level = np.mean(metrics['service_levels'][-eval_interval:])
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Service Level: {avg_service_level:.2f}%")
            print(f"Current Epsilon: {agent.epsilon:.3f}")
            print("---")
            
            plot_metrics(
                metrics['episode_rewards'],
                metrics['td_errors'],
                metrics['stockouts'],
                metrics['service_levels'],
                f'{results_dir}/intermediate_{episode + 1}'
            )
            save_metrics(metrics, f'{results_dir}/intermediate_{episode + 1}')
    
    print("Training completed!")
    
    plot_metrics(
        metrics['episode_rewards'],
        metrics['td_errors'],
        metrics['stockouts'],
        metrics['service_levels'],
        f'{results_dir}/final'
    )
    save_metrics(metrics, f'{results_dir}/final')
    
    return agent, metrics

if __name__ == "__main__":
    np.random.seed(42)
    agent, metrics = train(num_episodes=1000)
    
    print("\nFinal Performance Metrics:")
    print(f"Average Reward (last 1000 episodes): {np.mean(metrics['episode_rewards'][-1000:]):.2f}")
    print(f"Average Service Level (last 1000 episodes): {np.mean(metrics['service_levels'][-1000:]):.2f}%")
    print(f"Average Stockouts (last 1000 episodes): {np.mean(metrics['stockouts'][-1000:]):.2f}")
