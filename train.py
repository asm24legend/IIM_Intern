import numpy as np
import matplotlib.pyplot as plt
from inventory_env import InventoryEnvironment
from q_learning_agent import TDAgent
import json
from datetime import datetime
import os

def plot_metrics(rewards, td_errors, stockouts, service_levels, filename_prefix):
    """Plot and save training metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot episode rewards
    ax1.plot(rewards)
    ax1.set_title('Episode Rewards')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    
    # Plot TD errors
    ax2.plot(td_errors)
    ax2.set_title('TD Errors')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average TD Error')
    
    # Plot stockouts
    ax3.plot(stockouts)
    ax3.set_title('Stockouts per Episode')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Number of Stockouts')
    
    # Plot service levels
    ax4.plot(service_levels)
    ax4.set_title('Service Level')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Service Level (%)')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_training_metrics.png')
    plt.close()

def save_metrics(metrics, filename_prefix):
    """Save training metrics to JSON file"""
    with open(f'{filename_prefix}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def train(num_episodes=1000, eval_interval=50):
    # Initialize environment and agent
    env = InventoryEnvironment()
    agent = TDAgent(
        action_space=env.action_space,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        lambda_trace=0.9
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
            
            # Get next action for SARSA update (only in early phase)
            if episode < agent.policy_switch_episode and not done:
                next_action = agent.get_action(next_state, env)
            else:
                next_action = None
            
            # Update agent
            td_error = agent.learn(state, action, reward, next_state, next_action)
            
            # Update metrics
            episode_reward += reward
            episode_stockouts += sum(info['stockouts'].values())
            
            # Track demand fulfillment for service level
            for sku_id in env.skus:
                demand = info['stockouts'][sku_id]
                fulfilled = max(0, demand - info['stockouts'][sku_id])
                total_demand += demand
                fulfilled_demand += fulfilled
            
            state = next_state
        
        # Reset eligibility traces at end of episode
        agent.reset_traces()
        
        # Calculate service level
        service_level = (fulfilled_demand / total_demand * 100) if total_demand > 0 else 100
        
        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['td_errors'].append(agent.get_average_td_error())
        metrics['stockouts'].append(episode_stockouts)
        metrics['service_levels'].append(service_level)
        
        # Track inventory levels and supplier metrics
        metrics['warehouse_stock_levels'].append(
            {sku_id: level for sku_id, level in info['warehouse_stock'].items()}
        )
        metrics['retail_stock_levels'].append(
            {sku_id: level for sku_id, level in info['retail_stock'].items()}
        )
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
            
            # Save intermediate results
            plot_metrics(
                metrics['episode_rewards'],
                metrics['td_errors'],
                metrics['stockouts'],
                metrics['service_levels'],
                f'{results_dir}/intermediate_{episode + 1}'
            )
            save_metrics(metrics, f'{results_dir}/intermediate_{episode + 1}')
    
    print("Training completed!")
    
    # Save final results
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
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Train agent
    agent, metrics = train(num_episodes=1000)
    
    print("\nFinal Performance Metrics:")
    print(f"Average Reward (last 100 episodes): {np.mean(metrics['episode_rewards'][-100:]):.2f}")
    print(f"Average Service Level (last 100 episodes): {np.mean(metrics['service_levels'][-100:]):.2f}%")
    print(f"Average Stockouts (last 100 episodes): {np.mean(metrics['stockouts'][-100:]):.2f}") 