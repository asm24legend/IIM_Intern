
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
import pandas as pd

#Ensures that Numpy object types are converted to JSON format without throwing errors
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
#Plots four main plots to evaluate key RL performance metrics 
#rewards: list of total rewards per episode.
#td_errors: list of average temporal-difference (TD) errors per episode.
#metrics: list of dictionaries per episode with additional metrics like service level and stockouts.
#save_dir: directory path where the plot image will be saved.
def plot_training_progress(rewards, td_errors, metrics, save_dir):
    """Plot comprehensive training metrics for enhanced multi-SKU DQN system"""
    # Create subplots with better spacing - expanded to 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # Plot 1: Training Rewards
    ax1 = axes[0, 0]
    ax1.plot(rewards, color='#2E86AB', linewidth=1.5, alpha=0.8)
    ax1.set_title('Training Rewards', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Episode', fontsize=12)
    ax1.set_ylabel('Total Reward', fontsize=12)
    ax1.grid(True, alpha=0.3)
    if len(rewards) > 0:
        ax1.set_ylim(bottom=min(rewards) * 0.9, top=max(rewards) * 1.1)
    
    # Plot 2: TD Errors
    ax2 = axes[0, 1]
    ax2.plot(td_errors, color='#A23B72', linewidth=1.5, alpha=0.8)
    ax2.set_title('TD Errors (Learning Stability)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Episode', fontsize=12)
    ax2.set_ylabel('Average TD Error', fontsize=12)
    ax2.grid(True, alpha=0.3)
    if len(td_errors) > 0 and max(td_errors) > 10 * min(td_errors):
        ax2.set_yscale('log')
    
    # Plot 3: Service Levels
    ax3 = axes[0, 2]
    service_levels = [m.get('service_level', 0) for m in metrics]
    ax3.plot(service_levels, color='#F18F01', linewidth=1.5, alpha=0.8)
    ax3.set_title('Overall Service Level', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Episode', fontsize=12)
    ax3.set_ylabel('Service Level (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Plot 4: Stockouts by Location
    ax4 = axes[1, 0]
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    colors = ['#C73E1D', '#3B1F2B', '#6A994E', '#A7C957']
    for i, location in enumerate(locations):
        stockouts = [m.get('location_stockouts', {}).get(location, 0) for m in metrics]
        ax4.plot(stockouts, label=location, color=colors[i], linewidth=1.5, alpha=0.8)
    ax4.set_title('Stockouts per Location', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Episode', fontsize=12)
    ax4.set_ylabel('Number of Stockouts', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Per-SKU Service Levels
    ax5 = axes[1, 1]
    sku_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, sku_id in enumerate(['Type_A', 'Type_B', 'Type_C']):
        sku_service_levels = [m.get('sku_service_levels', {}).get(sku_id, 0) for m in metrics]
        ax5.plot(sku_service_levels, label=f'{sku_id}', color=sku_colors[i], linewidth=1.5, alpha=0.8)
    ax5.set_title('Per-SKU Service Levels', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Episode', fontsize=12)
    ax5.set_ylabel('Service Level (%)', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 100)
    
    # Plot 6: Inventory Turnover
    ax6 = axes[1, 2]
    turnover_rates = [m.get('inventory_turnover', 0) for m in metrics]
    ax6.plot(turnover_rates, color='#96CEB4', linewidth=1.5, alpha=0.8)
    ax6.set_title('Inventory Turnover Rate', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Episode', fontsize=12)
    ax6.set_ylabel('Turnover Rate', fontsize=12)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Average Order Quantities
    ax7 = axes[2, 0]
    avg_orders = [m.get('avg_order_quantity', 0) for m in metrics]
    ax7.plot(avg_orders, color='#FFEAA7', linewidth=1.5, alpha=0.8)
    ax7.set_title('Average Order Quantity', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Episode', fontsize=12)
    ax7.set_ylabel('Quantity', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Supplier Reliability Impact
    ax8 = axes[2, 1]
    reliability_impact = [m.get('supplier_reliability_impact', 0) for m in metrics]
    ax8.plot(reliability_impact, color='#DDA0DD', linewidth=1.5, alpha=0.8)
    ax8.set_title('Supplier Reliability Impact', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Episode', fontsize=12)
    ax8.set_ylabel('Impact Score', fontsize=12)
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: State Space Utilization
    ax9 = axes[2, 2]
    state_utilization = [m.get('state_utilization', 0) for m in metrics]
    ax9.plot(state_utilization, color='#98D8C8', linewidth=1.5, alpha=0.8)
    ax9.set_title('State Space Utilization', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Episode', fontsize=12)
    ax9.set_ylabel('Utilization %', fontsize=12)
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(0, 100)
    
    plt.tight_layout(pad=3.0)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'comprehensive_training_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()

#Helps to study trends over time
def plot_moving_average(data, window, title, ylabel, save_path):
    """Plot moving average for a given data series with improved styling."""
    moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(12, 8))
    plt.plot(moving_avg, color='#2E86AB', linewidth=2, alpha=0.8)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    x = np.arange(len(moving_avg))
    z = np.polyfit(x, moving_avg, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "--", color='#A23B72', linewidth=1.5, alpha=0.7, label=f'Trend (slope: {z[0]:.4f})')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#Core training loop
def train(env, agent, num_episodes=100, max_steps=5000):
    """Train the agent with comprehensive metrics collection"""
    rewards_history = []
    td_errors = []
    metrics_history = []
    episode_lengths = []
    pbar = tqdm(range(num_episodes), desc="Training Progress")
    
    for episode in pbar:
        observation = env.reset()  #Resets the environment at the beginning of each episode.
        state = env._get_state()  # Get rich state for agent
        done = False
        episode_reward = 0 #Tracks the cumulative reward collected during this episode.Updates after every step
        episode_steps = 0 #Tracks how many steps the agent has taken in this episode.
        
        # Enhanced metrics collection
        episode_metrics = {
            'stockouts': {}, #Tracks stockouts at each location
            'location_stockouts': {
                'Location_1': 0,
                'Location_2': 0,
                'Location_3': 0,
                'Retail': 0
            },
            'sku_service_levels': {},  # Per-SKU service levels
            'sku_stockouts': {},       # Per-SKU stockouts
            'sku_demands': {},         # Per-SKU demands
            'sku_orders': {},          # Per-SKU order quantities
            'service_level': 0,
            'warehouse_levels': {
                'Location_1': {},
                'Location_2': {},
                'Location_3': {}
            },
            'retail_levels': {},
            'supplier_reliability': {},
            'total_demand': 0,
            'total_fulfilled': 0,
            'total_orders': 0,
            'avg_order_quantity': 0,
            'inventory_turnover': 0,
            'supplier_reliability_impact': 0,
            'state_utilization': 0,
            'lead_time_violations': 0,
            'safety_stock_violations': 0
        }
        
        # Initialize per-SKU metrics
        for sku_id in env.skus:
            episode_metrics['sku_service_levels'][sku_id] = 0
            episode_metrics['sku_stockouts'][sku_id] = 0
            episode_metrics['sku_demands'][sku_id] = 0
            episode_metrics['sku_orders'][sku_id] = 0
        
        while not done and episode_steps < max_steps:
            # Get action from agent (order quantities)
            action = agent.get_action(state, env)
            # Take action in environment
            observation, reward, done, info = env.step(action)
            next_state = env._get_state()  # Get rich state for agent
            # Update agent
            td_error = agent.learn(state, action, reward, next_state, env)
            
            # Update basic metrics
            episode_reward += reward
            episode_steps += 1
            
            # Track stockouts by location
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_metrics['location_stockouts'][location] += stockout
                episode_metrics['sku_stockouts'][sku_id] += stockout
                if sku.retail_stock <= 0:
                    episode_metrics['location_stockouts']['Retail'] += stockout
            
            # Track per-SKU metrics
            for sku_id in env.skus:
                sku = env.skus[sku_id]
                location = sku.inventory_location
                location_stock_key = location + '_stock'
                
                # Warehouse levels
                if sku_id not in episode_metrics['warehouse_levels'][location]:
                    episode_metrics['warehouse_levels'][location][sku_id] = []
                episode_metrics['warehouse_levels'][location][sku_id].append(
                    int(info[location_stock_key][sku_id])
                )
                
                # Retail levels
                if sku_id not in episode_metrics['retail_levels']:
                    episode_metrics['retail_levels'][sku_id] = []
                episode_metrics['retail_levels'][sku_id].append(int(info['retail_stock'][sku_id]))
                
                # Track demands and orders
                episode_metrics['sku_demands'][sku_id] += sku.previous_demand
                episode_metrics['sku_orders'][sku_id] += action[list(env.skus.keys()).index(sku_id)]
                
                # Track lead time and safety stock violations
                if sku.current_stock < sku.safety_stock:
                    episode_metrics['safety_stock_violations'] += 1
                if sku.lead_time_days > sku.shelf_life_days:
                    episode_metrics['lead_time_violations'] += 1
            
            # Track supplier reliability impact
            episode_metrics['supplier_reliability'] = {
                k: float(v) for k, v in info['supplier_reliability'].items()
            }
            
            # Calculate supplier reliability impact (weighted by order quantities)
            total_reliability_impact = 0
            total_orders = 0
            for sku_id in env.skus:
                sku = env.skus[sku_id]
                supplier = sku.supplier
                reliability = info['supplier_reliability'].get(supplier, 1.0)
                order_qty = action[list(env.skus.keys()).index(sku_id)]
                total_reliability_impact += (1 - reliability) * order_qty
                total_orders += order_qty
            
            if total_orders > 0:
                episode_metrics['supplier_reliability_impact'] = total_reliability_impact / total_orders
            
            state = next_state
        
        # Calculate final episode metrics
        episode_metrics['service_level'] = float(np.mean([
            service_level * 100 for service_level in info['service_levels'].values()
        ]))
        
        # Calculate per-SKU service levels
        for sku_id in env.skus:
            sku = env.skus[sku_id]
            if sku.total_demand > 0:
                episode_metrics['sku_service_levels'][sku_id] = (sku.fulfilled_demand / sku.total_demand) * 100
            else:
                episode_metrics['sku_service_levels'][sku_id] = 100.0
        
        # Calculate inventory turnover
        total_inventory = sum(sku.current_stock + sku.retail_stock for sku in env.skus.values())
        total_demand = sum(sku.total_demand for sku in env.skus.values())
        if total_inventory > 0:
            episode_metrics['inventory_turnover'] = total_demand / total_inventory
        
        # Calculate average order quantity
        total_orders = sum(episode_metrics['sku_orders'].values())
        if len(episode_metrics['sku_orders']) > 0:
            episode_metrics['avg_order_quantity'] = total_orders / len(episode_metrics['sku_orders'])
        
        # Calculate state space utilization (how much of the rich state is being used)
        state_features_used = 0
        total_features = len(state)
        for sku_id in env.skus:
            sku = env.skus[sku_id]
            # Count non-zero features for this SKU
            if sku.current_stock > 0 or sku.retail_stock > 0 or sku.open_pos_supplier_to_warehouse > 0:
                state_features_used += 1
        episode_metrics['state_utilization'] = (state_features_used / total_features) * 100
        
        # Store episode results
        rewards_history.append(float(episode_reward))
        td_errors.append(float(td_error))  # Use the actual batch TD error
        metrics_history.append(episode_metrics)
        episode_lengths.append(int(episode_steps))
        
        # Update progress bar with enhanced metrics
        if (episode + 1) % 10 == 0:
            avg_reward = float(np.mean(rewards_history[-10:]))
            avg_service_level = float(np.mean([
                m['service_level'] for m in metrics_history[-10:]
            ]))
            avg_stockouts = {
                location: np.mean([m['location_stockouts'][location] for m in metrics_history[-10:]])
                for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']
            }
            avg_turnover = float(np.mean([
                m['inventory_turnover'] for m in metrics_history[-10:]
            ]))
            
            pbar.set_postfix({
                'Reward': f'{avg_reward:.2f}',
                'Service': f'{avg_service_level:.1f}%',
                'Turnover': f'{avg_turnover:.2f}',
                'L1': f'{avg_stockouts["Location_1"]:.1f}',
                'L2': f'{avg_stockouts["Location_2"]:.1f}',
                'L3': f'{avg_stockouts["Location_3"]:.1f}',
                'Retail': f'{avg_stockouts["Retail"]:.1f}'
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
        'episode_lengths': []
    }
    for episode in range(num_episodes):
        observation = env.reset()
        state = env._get_state()  # Get rich state for agent
        done = False
        episode_reward = 0
        episode_steps = 0
        episode_stockouts = {
            'Location_1': 0,
            'Location_2': 0,
            'Location_3': 0,
            'Retail': 0
        }
        this_max_steps = max_steps
        if episode_lengths_override is not None and episode < len(episode_lengths_override):
            this_max_steps = episode_lengths_override[episode]
        while not done and episode_steps < this_max_steps:
            action = agent.get_action(state, env, greedy=True)
            observation, reward, done, info = env.step(action)
            next_state = env._get_state()  # Get rich state for agent
            episode_reward += reward
            episode_steps += 1
            for sku_id, stockout in info['stockouts'].items():
                sku = env.skus[sku_id]
                location = sku.inventory_location
                episode_stockouts[location] += stockout
                if sku.retail_stock <= 0:
                    episode_stockouts['Retail'] += stockout
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
    """Plot the cumulative reward as a function of the number of steps with improved styling."""
    rewards = eval_metrics['rewards']
    cumulative_rewards = np.cumsum(rewards)
    steps = np.arange(1, len(cumulative_rewards) + 1)
    
    plt.figure(figsize=(12, 8))
    plt.plot(steps, cumulative_rewards, color='#2E86AB', linewidth=2, alpha=0.8, label='Cumulative Reward')
    
    # Add trend line
    z = np.polyfit(steps, cumulative_rewards, 1)
    p = np.poly1d(z)
    plt.plot(steps, p(steps), "--", color='#A23B72', linewidth=1.5, alpha=0.7, 
             label=f'Trend (avg reward/episode: {z[0]:.2f})')
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Cumulative Reward', fontsize=14)
    plt.title('Cumulative Reward Over Episodes', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visibility
    plt.ylim(bottom=0, top=max(cumulative_rewards) * 1.05)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'cumulative_reward.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_location_stockouts(eval_metrics, save_dir):
    """Plot stockouts by location with improved styling"""
    plt.figure(figsize=(14, 8))
    
    locations = ['Location_1', 'Location_2', 'Location_3', 'Retail']
    colors = ['#C73E1D', '#3B1F2B', '#6A994E', '#A7C957']
    
    for i, location in enumerate(locations):
        stockouts = eval_metrics['location_stockouts'][location]
        plt.plot(stockouts, label=location, color=colors[i], linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Number of Stockouts', fontsize=14)
    plt.title('Stockouts by Location', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visibility
    all_stockouts = []
    for location in locations:
        all_stockouts.extend(eval_metrics['location_stockouts'][location])
    plt.ylim(bottom=0, top=max(all_stockouts) * 1.1)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'location_stockouts.png'), dpi=300, bbox_inches='tight')
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
        num_skus = len(self.env.skus)
        num_locations = len(state) // num_skus
        order_quantities = np.zeros(num_skus, dtype=np.int32)
        lead_time_reductions = np.zeros(num_skus, dtype=np.int32)
        for i, sku_id in enumerate(self.env.skus):
            sku = self.env.skus[sku_id]
            sku_stocks = state[i*num_locations:(i+1)*num_locations]
            target_level = sku.reorder_point
            # If any location is below target, order up to target
            if np.any(np.array(sku_stocks) < target_level):
                order_quantities[i] = target_level - min(sku_stocks)
            else:
                order_quantities[i] = 0
        return np.concatenate([order_quantities, lead_time_reductions])
    def learn(self, *args, **kwargs):
        return 0.0
    def get_average_td_error(self):
        return 0.0
    def save(self, path):
        pass

def plot_comprehensive_comparison(td_eval_metrics, dqn_eval_metrics, random_metrics, save_dir):
    """Create a comprehensive comparison plot showing all key metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Colors for consistency
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    agents = ['Q-learning', 'Double DQN', 'Random']
    
    # 1. Average Rewards Comparison
    avg_rewards = [
        np.mean(td_eval_metrics['rewards']),
        np.mean(dqn_eval_metrics['rewards']),
        np.mean(random_metrics['rewards'])
    ]
    
    bars1 = ax1.bar(agents, avg_rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Average Episode Rewards', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Average Reward', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars1, avg_rewards):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(avg_rewards) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 2. Service Levels Comparison
    avg_service_levels = [
        np.mean(td_eval_metrics['service_levels']),
        np.mean(dqn_eval_metrics['service_levels']),
        np.mean(random_metrics['service_levels'])
    ]
    
    bars2 = ax2.bar(agents, avg_service_levels, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Average Service Levels', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Service Level (%)', fontsize=14)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, avg_service_levels):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 3. Total Stockouts Comparison
    total_stockouts_td = sum([np.mean(td_eval_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    total_stockouts_dqn = sum([np.mean(dqn_eval_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    total_stockouts_random = sum([np.mean(random_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    
    total_stockouts = [total_stockouts_td, total_stockouts_dqn, total_stockouts_random]
    
    bars3 = ax3.bar(agents, total_stockouts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title('Total Average Stockouts (All Locations)', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Total Stockouts', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars3, total_stockouts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(total_stockouts) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Episode Lengths Comparison
    avg_episode_lengths = [
        np.mean(td_eval_metrics['episode_lengths']),
        np.mean(dqn_eval_metrics['episode_lengths']),
        np.mean(random_metrics['episode_lengths'])
    ]
    
    bars4 = ax4.bar(agents, avg_episode_lengths, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('Average Episode Lengths', fontsize=16, fontweight='bold')
    ax4.set_ylabel('Episode Length (Steps)', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars4, avg_episode_lengths):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + max(avg_episode_lengths) * 0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'comprehensive_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_performance_summary(td_eval_metrics, dqn_eval_metrics, random_metrics, save_dir):
    """Save a comprehensive performance summary as CSV"""
    
    # Calculate all metrics
    summary_data = {
        'Metric': [],
        'Q-learning': [],
        'Double DQN': [],
        'Random': []
    }
    
    # Average rewards
    summary_data['Metric'].append('Average Reward')
    summary_data['Q-learning'].append(np.mean(td_eval_metrics['rewards']))
    summary_data['Double DQN'].append(np.mean(dqn_eval_metrics['rewards']))
    summary_data['Random'].append(np.mean(random_metrics['rewards']))
    
    # Service levels
    summary_data['Metric'].append('Average Service Level (%)')
    summary_data['Q-learning'].append(np.mean(td_eval_metrics['service_levels']))
    summary_data['Double DQN'].append(np.mean(dqn_eval_metrics['service_levels']))
    summary_data['Random'].append(np.mean(random_metrics['service_levels']))
    
    # Episode lengths
    summary_data['Metric'].append('Average Episode Length')
    summary_data['Q-learning'].append(np.mean(td_eval_metrics['episode_lengths']))
    summary_data['Double DQN'].append(np.mean(dqn_eval_metrics['episode_lengths']))
    summary_data['Random'].append(np.mean(random_metrics['episode_lengths']))
    
    # Stockouts by location
    for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']:
        summary_data['Metric'].append(f'Average Stockouts - {location}')
        summary_data['Q-learning'].append(np.mean(td_eval_metrics['location_stockouts'][location]))
        summary_data['Double DQN'].append(np.mean(dqn_eval_metrics['location_stockouts'][location]))
        summary_data['Random'].append(np.mean(random_metrics['location_stockouts'][location]))
    
    # Total stockouts
    total_stockouts_td = sum([np.mean(td_eval_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    total_stockouts_dqn = sum([np.mean(dqn_eval_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    total_stockouts_random = sum([np.mean(random_metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']])
    
    summary_data['Metric'].append('Total Average Stockouts')
    summary_data['Q-learning'].append(total_stockouts_td)
    summary_data['Double DQN'].append(total_stockouts_dqn)
    summary_data['Random'].append(total_stockouts_random)
    
    # Reward per step
    avg_reward_per_step_td = np.mean(td_eval_metrics['rewards']) / np.mean(td_eval_metrics['episode_lengths'])
    avg_reward_per_step_dqn = np.mean(dqn_eval_metrics['rewards']) / np.mean(dqn_eval_metrics['episode_lengths'])
    avg_reward_per_step_random = np.mean(random_metrics['rewards']) / np.mean(random_metrics['episode_lengths'])
    
    summary_data['Metric'].append('Average Reward per Step')
    summary_data['Q-learning'].append(avg_reward_per_step_td)
    summary_data['Double DQN'].append(avg_reward_per_step_dqn)
    summary_data['Random'].append(avg_reward_per_step_random)
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, 'performance_summary.csv'), index=False)
    
    return df

def generate_comprehensive_report(metrics_history, rewards_history, td_errors, save_dir):
    """Generate a comprehensive performance report for the enhanced multi-SKU DQN system"""
    import pandas as pd
    from datetime import datetime
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert metrics to DataFrame for analysis
    df = pd.DataFrame(metrics_history)
    
    # Calculate summary statistics
    report = {
        'training_summary': {
            'total_episodes': len(rewards_history),
            'avg_reward': np.mean(rewards_history),
            'std_reward': np.std(rewards_history),
            'min_reward': np.min(rewards_history),
            'max_reward': np.max(rewards_history),
            'avg_td_error': np.mean(td_errors),
            'final_td_error': td_errors[-1] if td_errors else 0,
        },
        'service_level_analysis': {
            'avg_service_level': np.mean(df['service_level']),
            'service_level_trend': 'improving' if df['service_level'].iloc[-10:].mean() > df['service_level'].iloc[:10].mean() else 'declining',
            'service_level_stability': np.std(df['service_level']),
        },
        'inventory_performance': {
            'avg_turnover': np.mean(df['inventory_turnover']),
            'avg_order_quantity': np.mean(df['avg_order_quantity']),
            'state_utilization': np.mean(df['state_utilization']),
        },
        'stockout_analysis': {
            'total_stockouts': sum(df['location_stockouts'].apply(lambda x: sum(x.values()))),
            'stockout_distribution': {
                'Location_1': np.mean([m['location_stockouts']['Location_1'] for m in metrics_history]),
                'Location_2': np.mean([m['location_stockouts']['Location_2'] for m in metrics_history]),
                'Location_3': np.mean([m['location_stockouts']['Location_3'] for m in metrics_history]),
                'Retail': np.mean([m['location_stockouts']['Retail'] for m in metrics_history]),
            }
        },
        'per_sku_performance': {},
        'learning_analysis': {
            'td_error_trend': 'decreasing' if td_errors[-10:].mean() < td_errors[:10].mean() else 'increasing',
            'convergence_indicator': 'converged' if np.std(td_errors[-20:]) < np.std(td_errors[:20]) * 0.5 else 'not_converged',
        }
    }
    
    # Per-SKU analysis
    for sku_id in ['Type_A', 'Type_B', 'Type_C']:
        sku_service_levels = [m.get('sku_service_levels', {}).get(sku_id, 0) for m in metrics_history]
        sku_stockouts = [m.get('sku_stockouts', {}).get(sku_id, 0) for m in metrics_history]
        sku_orders = [m.get('sku_orders', {}).get(sku_id, 0) for m in metrics_history]
        
        report['per_sku_performance'][sku_id] = {
            'avg_service_level': np.mean(sku_service_levels),
            'total_stockouts': sum(sku_stockouts),
            'avg_order_quantity': np.mean(sku_orders),
            'service_level_stability': np.std(sku_service_levels),
        }
    
    # Generate detailed plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Reward progression
    axes[0, 0].plot(rewards_history, alpha=0.7)
    axes[0, 0].set_title('Training Rewards Progression')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Service level by SKU
    for sku_id in ['Type_A', 'Type_B', 'Type_C']:
        sku_service_levels = [m.get('sku_service_levels', {}).get(sku_id, 0) for m in metrics_history]
        axes[0, 1].plot(sku_service_levels, label=sku_id, alpha=0.7)
    axes[0, 1].set_title('Service Level by SKU')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Service Level (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Inventory turnover
    axes[0, 2].plot(df['inventory_turnover'], alpha=0.7)
    axes[0, 2].set_title('Inventory Turnover Rate')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Turnover Rate')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: TD Error progression
    axes[1, 0].plot(td_errors, alpha=0.7)
    axes[1, 0].set_title('TD Error Progression')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('TD Error')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: State utilization
    axes[1, 1].plot(df['state_utilization'], alpha=0.7)
    axes[1, 1].set_title('State Space Utilization')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Utilization (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Supplier reliability impact
    axes[1, 2].plot(df['supplier_reliability_impact'], alpha=0.7)
    axes[1, 2].set_title('Supplier Reliability Impact')
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Impact Score')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed report as JSON
    with open(os.path.join(save_dir, 'comprehensive_report.json'), 'w') as f:
        json.dump(convert_to_serializable(report), f, indent=2)
    
    # Save metrics as CSV
    df.to_csv(os.path.join(save_dir, 'detailed_metrics.csv'), index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPREHENSIVE PERFORMANCE REPORT")
    print("="*60)
    print(f"Training Episodes: {report['training_summary']['total_episodes']}")
    print(f"Average Reward: {report['training_summary']['avg_reward']:.2f} ± {report['training_summary']['std_reward']:.2f}")
    print(f"Average Service Level: {report['service_level_analysis']['avg_service_level']:.1f}%")
    print(f"Average Inventory Turnover: {report['inventory_performance']['avg_turnover']:.2f}")
    print(f"State Space Utilization: {report['inventory_performance']['state_utilization']:.1f}%")
    print(f"Learning Status: {report['learning_analysis']['convergence_indicator']}")
    print("\nPer-SKU Performance:")
    for sku_id, perf in report['per_sku_performance'].items():
        print(f"  {sku_id}: Service Level {perf['avg_service_level']:.1f}%, Stockouts {perf['total_stockouts']}")
    print("="*60)
    
    return report

def main():
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment and agents
    env = InventoryEnvironment()
    print("\n[INFO] Demand variability has been reduced for all SKUs (FMCG scenario).\n")
    for sku_id, sku in env.skus.items():
        print(f"SKU {sku_id}: alpha={sku.alpha}, beta={sku.beta}")
    
    # Q-learning agent
    td_agent = TDAgent(
        action_space=env.action_space,
        discount_factor=0.995,  # More focus on long-term rewards
        learning_rate=0.05,     # Smaller, more stable updates
        epsilon=1.0,
        epsilon_min=0.01,       # Allow more exploration
        epsilon_decay=0.9999    # Slower decay for more stable learning
    )
    
    # Double DQN agent
    dqn_agent = DoubleDQNAgent(
        action_space=env.action_space,
        discount_factor=0.98,   # Slightly less focus on long-term rewards for faster learning
        learning_rate=0.001,    # Higher learning rate for faster updates
        epsilon=1.0             # Initial exploration
    )
    # Manually override DQN agent's batch size and target update frequency for faster learning
    dqn_agent.batch_size = 32  # Reduced from 64 for more frequent updates
    dqn_agent.epsilon_decay = 0.9995  # Faster decay for quicker exploitation
    dqn_agent.update_target_every = 3  # Update target network more frequently
    dqn_agent.learning_starts = 100  # Start learning earlier
    
    # Training parameters
    num_episodes = 10000  # Reduced from 1500 for faster training
    eval_interval = 100
    
    print("Starting training for Q-learning agent...")
    print(f"Training will run for {num_episodes} episodes")
    
    # Train Double Q-learning agent
    td_rewards_history, td_metrics_history, td_episode_lengths, td_errors = train(
        env, 
        td_agent,
        num_episodes=num_episodes,
        max_steps=500
    )
    
    # Plot training progress for Q-learning
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
    
    # Plot training progress for Double DQN with enhanced metrics
    os.makedirs(results_dir, exist_ok=True)
    plot_training_progress(dqn_rewards_history, dqn_errors, dqn_metrics_history, results_dir)
    
    # Generate comprehensive performance report
    print("\nGenerating comprehensive performance report...")
    comprehensive_report = generate_comprehensive_report(dqn_metrics_history, dqn_rewards_history, dqn_errors, results_dir)
    
    # Plot moving average of TD error
    plot_moving_average(
        dqn_errors,
        window=100,
        title='DQN TD Error (Moving Average, window=100)',
        ylabel='TD Error',
        save_path=os.path.join(results_dir, 'dqn_td_error_moving_avg.png')
    )
    # Plot moving average of reward
    plot_moving_average(
        dqn_rewards_history,
        window=100,
        title='DQN Reward (Moving Average, window=100)',
        ylabel='Reward',
        save_path=os.path.join(results_dir, 'dqn_reward_moving_avg.png')
    )
    
    # Evaluate all agents
    print("\nEvaluating Q-learning agent...")
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
    plt.figure(figsize=(14, 8))
    
    # Calculate statistics for better scaling
    all_rewards = td_eval_metrics['rewards'] + dqn_eval_metrics['rewards'] + random_metrics['rewards']
    min_reward = min(all_rewards)
    max_reward = max(all_rewards)
    
    # Create histogram with better styling
    bins = np.linspace(min_reward, max_reward, 30)
    
    plt.hist(td_eval_metrics['rewards'], bins=bins, alpha=0.7, label='Double Q-learning Agent', 
             color='#2E86AB', edgecolor='black', linewidth=0.5)
    plt.hist(dqn_eval_metrics['rewards'], bins=bins, alpha=0.7, label='Double DQN Agent', 
             color='#A23B72', edgecolor='black', linewidth=0.5)
    plt.hist(random_metrics['rewards'], bins=bins, alpha=0.7, label='Random Agent', 
             color='#F18F01', edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Episode Reward', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Reward Distribution: Q-learning vs Double DQN vs Random Agent', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for means
    mean_td = np.mean(td_eval_metrics['rewards'])
    mean_dqn = np.mean(dqn_eval_metrics['rewards'])
    mean_random = np.mean(random_metrics['rewards'])
    
    plt.axvline(mean_td, color='#2E86AB', linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Q-learning Mean: {mean_td:.1f}')
    plt.axvline(mean_dqn, color='#A23B72', linestyle='--', linewidth=2, alpha=0.8, 
                label=f'DQN Mean: {mean_dqn:.1f}')
    plt.axvline(mean_random, color='#F18F01', linestyle='--', linewidth=2, alpha=0.8, 
                label=f'Random Mean: {mean_random:.1f}')
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'reward_histogram.png'), dpi=300, bbox_inches='tight')
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
    print(f"Average Reward (Q-learning): {np.mean(td_eval_metrics['rewards']):.2f}")
    print(f"Average Reward (Double DQN): {np.mean(dqn_eval_metrics['rewards']):.2f}")
    print(f"Average Reward (Random): {np.mean(random_metrics['rewards']):.2f}")
    print(f"Average Service Level (Q-learning): {np.mean(td_eval_metrics['service_levels']):.1f}%")
    print(f"Average Service Level (Double DQN): {np.mean(dqn_eval_metrics['service_levels']):.1f}%")
    print(f"Average Service Level (Random): {np.mean(random_metrics['service_levels']):.1f}%")
    print(f"Average Episode Length (Q-learning): {np.mean(td_eval_metrics['episode_lengths']):.2f}")
    print(f"Average Episode Length (Double DQN): {np.mean(dqn_eval_metrics['episode_lengths']):.2f}")
    print(f"Average Episode Length (Random): {np.mean(random_metrics['episode_lengths']):.2f}")
    
    print("\nAverage Stockouts by Location:")
    for location in ['Location_1', 'Location_2', 'Location_3', 'Retail']:
        avg_stockouts_td = np.mean(td_eval_metrics['location_stockouts'][location])
        avg_stockouts_dqn = np.mean(dqn_eval_metrics['location_stockouts'][location])
        avg_stockouts_random = np.mean(random_metrics['location_stockouts'][location])
        print(f"{location} (Q-learning): {avg_stockouts_td:.2f}")
        print(f"{location} (Double DQN): {avg_stockouts_dqn:.2f}")
        print(f"{location} (Random): {avg_stockouts_random:.2f}")

    # Plot bar chart comparing metrics for all three agents
    labels = ['Service Level', 'Stockouts L1', 'Stockouts L2', 'Stockouts L3', 'Stockouts Retail', 'Ep. Length']
    
    td_values = [
        np.mean(td_eval_metrics['service_levels']),
        np.mean(td_eval_metrics['location_stockouts']['Location_1']),
        np.mean(td_eval_metrics['location_stockouts']['Location_2']),
        np.mean(td_eval_metrics['location_stockouts']['Location_3']),
        np.mean(td_eval_metrics['location_stockouts']['Retail']),
        np.mean(td_eval_metrics['episode_lengths'])
    ]
    
    dqn_values = [
        np.mean(dqn_eval_metrics['service_levels']),
        np.mean(dqn_eval_metrics['location_stockouts']['Location_1']),
        np.mean(dqn_eval_metrics['location_stockouts']['Location_2']),
        np.mean(dqn_eval_metrics['location_stockouts']['Location_3']),
        np.mean(dqn_eval_metrics['location_stockouts']['Retail']),
        np.mean(dqn_eval_metrics['episode_lengths'])
    ]
    
    random_values = [
        np.mean(random_metrics['service_levels']),
        np.mean(random_metrics['location_stockouts']['Location_1']),
        np.mean(random_metrics['location_stockouts']['Location_2']),
        np.mean(random_metrics['location_stockouts']['Location_3']),
        np.mean(random_metrics['location_stockouts']['Retail']),
        np.mean(random_metrics['episode_lengths'])
    ]
    
    x = np.arange(len(labels))
    width = 0.25
    
    plt.figure(figsize=(18, 10))
    
    # Create bars with better styling
    bars1 = plt.bar(x - width, td_values, width, label='Q-learning', color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = plt.bar(x, dqn_values, width, label='Double DQN', color='#A23B72', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars3 = plt.bar(x + width, random_values, width, label='Random', color='#F18F01', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    add_value_labels(bars1, td_values)
    add_value_labels(bars2, dqn_values)
    add_value_labels(bars3, random_values)
    
    plt.xlabel('Metrics', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.title('Comparison of Key Metrics: Double Q-learning vs Double DQN vs Random Agent', fontsize=16, fontweight='bold')
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits for better visibility
    all_values = td_values + dqn_values + random_values
    plt.ylim(0, max(all_values) * 1.15)
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'metrics_comparison_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create comprehensive comparison plot
    plot_comprehensive_comparison(td_eval_metrics, dqn_eval_metrics, random_metrics, results_dir)

    # Save performance summary
    performance_df = save_performance_summary(td_eval_metrics, dqn_eval_metrics, random_metrics, results_dir)
    print("\nPerformance Summary:")
    print(performance_df.to_string(index=False))

    # Save trained agents
    td_agent.save(os.path.join(results_dir, 'trained_td_agent'))
    dqn_agent.save(os.path.join(results_dir, 'trained_dqn_agent'))

    # Export Q-table to CSV (use Q-table A for export)
    q_table_path = os.path.join(results_dir, 'trained_td_agent_A.pkl')
    csv_path = os.path.join(results_dir, 'td_q_table.csv')
   
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
