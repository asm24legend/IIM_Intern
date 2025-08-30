
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
        # Improved calculation that counts actual meaningful feature values
        state_features_used = 0
        total_features = len(state)
        feature_idx = 0
        
        for sku_id in env.skus:
            sku = env.skus[sku_id]
            
            # Count features that have meaningful/non-default values
            # Features per SKU: 24 total features (enhanced state in inventory_env.py)
            
            # Basic inventory levels (2 features)
            if sku.current_stock > 0: state_features_used += 1
            if sku.retail_stock > 0: state_features_used += 1
            
            # Open purchase orders (2 features)
            if sku.open_pos_supplier_to_warehouse > 0: state_features_used += 1
            if sku.open_pos_warehouse_to_retail > 0: state_features_used += 1
            
            # Demand information (1 feature) - always meaningful if > 0
            period_demand = state[feature_idx + 4] if feature_idx + 4 < len(state) else 0
            if period_demand > 0: state_features_used += 1
            
            # Lead time information (2 features) - always meaningful
            state_features_used += 2  # lead_time_days and days_to_delivery always used
            
            # Supplier information (2 features) - always meaningful
            state_features_used += 2  # supplier load and reliability always used
            
            # Time information (1 feature) - always meaningful
            state_features_used += 1  # time_delta always used
            
            # Service level metrics (1 feature) - meaningful if demand exists
            if sku.total_demand > 0: state_features_used += 1
            
            # Inventory thresholds (3 features) - always meaningful
            state_features_used += 3  # safety_stock, reorder_point, max_stock always used
            
            # ABC classification (1 feature) - always meaningful
            state_features_used += 1  # ABC class always used
            
            # Demand history statistics (2 features) - meaningful if history exists
            if len(sku.demand_history) > 0: 
                state_features_used += 2  # avg_demand and demand_std
            else:
                state_features_used += 1  # base_demand used as fallback
            
            # Stockout and replenishment metrics (2 features) - always meaningful
            state_features_used += 2  # stockout_occasions and replenishment_cycles always tracked
            
            # Enhanced features (7 features) - mostly always meaningful
            state_features_used += 1  # demand_volatility (if history exists)
            state_features_used += 1  # seasonal_factor (always meaningful)
            state_features_used += 1  # trend_factor (always meaningful)
            state_features_used += 1  # forecast_accuracy (always meaningful)
            state_features_used += 1  # days_since_stockout (always meaningful)
            state_features_used += 1  # consecutive_stockouts (always meaningful)
            state_features_used += 1  # demand_forecast (always meaningful)
            
            feature_idx += 24  # Move to next SKU's features (now 24 features per SKU)
        
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
    """
    Random agent that uses gamma distribution demand forecasting for inventory decisions.
    This agent generates demand forecasts using the same gamma distribution parameters
    as the environment (alpha, beta) and makes ordering decisions based on these forecasts.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        
    def get_gamma_demand_forecast(self, sku, forecast_days=7):
        """Generate gamma distribution-based demand forecast for the SKU"""
        # Use the same gamma parameters as the environment
        alpha = sku.alpha
        beta = sku.beta
        
        # Generate demand forecast using gamma distribution
        forecast_demand = np.random.gamma(shape=alpha * forecast_days, scale=beta)
        return max(0, forecast_demand)
    
    def get_action(self, state, env=None, greedy=False):
        num_skus = len(self.env.skus)
        num_locations = len(state) // num_skus
        order_quantities = np.zeros(num_skus, dtype=np.int32)
        
        for i, sku_id in enumerate(self.env.skus):
            sku = self.env.skus[sku_id]
            sku_stocks = state[i*num_locations:(i+1)*num_locations]
            
            # Generate gamma distribution-based demand forecast
            forecast_demand = self.get_gamma_demand_forecast(sku, forecast_days=sku.lead_time_days)
            
            # Calculate target inventory level based on gamma demand forecast
            # Target = forecast demand during lead time + safety stock
            target_level = int(forecast_demand + sku.safety_stock)
            
            # Current stock across all locations for this SKU
            current_total_stock = np.sum(sku_stocks)
            
            # Calculate order quantity based on gamma demand forecast
            if current_total_stock < target_level:
                order_quantities[i] = min(
                    target_level - current_total_stock,
                    sku.max_stock - current_total_stock  # Respect max stock limit
                )
            else:
                order_quantities[i] = 0
                
        return order_quantities
        
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
    import numpy as np
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
            'td_error_trend': 'decreasing' if np.mean(td_errors[-10:]) < np.mean(td_errors[:10]) else 'increasing',
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

def plot_demand_comparison(gamma_results, fixed_results, save_dir):
    """Plot comprehensive comparison between gamma and fixed demand scenarios"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Colors for agents
    colors = {'q_learning': '#2E86AB', 'dqn': '#A23B72', 'random': '#F18F01'}
    
    # Metrics to compare
    metrics = ['rewards', 'service_levels', 'episode_lengths']
    metric_names = ['Average Reward', 'Service Level (%)', 'Episode Length']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        # Gamma demand results
        ax1 = axes[0, i]
        agents = ['Q-Learning', 'Double DQN', 'Random']
        gamma_values = [
            np.mean(gamma_results['q_learning'][metric]),
            np.mean(gamma_results['dqn'][metric]),
            np.mean(gamma_results['random'][metric])
        ]
        
        bars1 = ax1.bar(agents, gamma_values, color=list(colors.values()), alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_title(f'{name} - Gamma Demand', fontsize=14, fontweight='bold')
        ax1.set_ylabel(name, fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, gamma_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(gamma_values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Fixed demand results
        ax2 = axes[1, i]
        fixed_values = [
            np.mean(fixed_results['q_learning'][metric]),
            np.mean(fixed_results['dqn'][metric]),
            np.mean(fixed_results['random'][metric])
        ]
        
        bars2 = ax2.bar(agents, fixed_values, color=list(colors.values()), alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.set_title(f'{name} - Fixed Demand', fontsize=14, fontweight='bold')
        ax2.set_ylabel(name, fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, fixed_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(fixed_values) * 0.01,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout(pad=3.0)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'gamma_vs_fixed_demand_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_histograms(gamma_results, fixed_results, save_dir):
    """Plot combined histograms showing reward distributions for both demand types"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Colors for agents
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    agents = ['Q-Learning', 'Double DQN', 'Random']
    
    # Gamma demand histogram
    ax1 = axes[0]
    all_gamma_rewards = (gamma_results['q_learning']['rewards'] + 
                        gamma_results['dqn']['rewards'] + 
                        gamma_results['random']['rewards'])
    min_reward = min(all_gamma_rewards)
    max_reward = max(all_gamma_rewards)
    bins = np.linspace(min_reward, max_reward, 30)
    
    ax1.hist(gamma_results['q_learning']['rewards'], bins=bins, alpha=0.7, 
             label='Q-Learning', color=colors[0], edgecolor='black', linewidth=0.5)
    ax1.hist(gamma_results['dqn']['rewards'], bins=bins, alpha=0.7, 
             label='Double DQN', color=colors[1], edgecolor='black', linewidth=0.5)
    ax1.hist(gamma_results['random']['rewards'], bins=bins, alpha=0.7, 
             label='Random', color=colors[2], edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('Episode Reward', fontsize=14)
    ax1.set_ylabel('Frequency', fontsize=14)
    ax1.set_title('Reward Distribution - Gamma Demand', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Fixed demand histogram
    ax2 = axes[1]
    all_fixed_rewards = (fixed_results['q_learning']['rewards'] + 
                        fixed_results['dqn']['rewards'] + 
                        fixed_results['random']['rewards'])
    min_reward = min(all_fixed_rewards)
    max_reward = max(all_fixed_rewards)
    bins = np.linspace(min_reward, max_reward, 30)
    
    ax2.hist(fixed_results['q_learning']['rewards'], bins=bins, alpha=0.7, 
             label='Q-Learning', color=colors[0], edgecolor='black', linewidth=0.5)
    ax2.hist(fixed_results['dqn']['rewards'], bins=bins, alpha=0.7, 
             label='Double DQN', color=colors[1], edgecolor='black', linewidth=0.5)
    ax2.hist(fixed_results['random']['rewards'], bins=bins, alpha=0.7, 
             label='Random', color=colors[2], edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Episode Reward', fontsize=14)
    ax2.set_ylabel('Frequency', fontsize=14)
    ax2.set_title('Reward Distribution - Fixed Demand', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'combined_reward_histograms.png'), dpi=300, bbox_inches='tight')
    plt.close()

def run_agent_experiments(env, results_dir, num_episodes=5000, eval_episodes=100):
    """Run experiments for all three agents and return evaluation metrics"""
    results = {}
    
    # Q-learning agent
    print("Training Q-learning agent...")
    td_agent = TDAgent(
        action_space=env.action_space,
        discount_factor=0.995,
        learning_rate=0.05,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.9999
    )
    
    td_rewards_history, td_metrics_history, td_episode_lengths, td_errors = train(
        env, td_agent, num_episodes=num_episodes, max_steps=500
    )
    
    print("Evaluating Q-learning agent...")
    td_eval_metrics = evaluate(env, td_agent, num_episodes=eval_episodes, max_steps=500)
    results['q_learning'] = td_eval_metrics
    
    # Double DQN agent
    print("Training Double DQN agent...")
    dqn_agent = DoubleDQNAgent(
        action_space=env.action_space,
        discount_factor=0.98,
        learning_rate=0.001,
        epsilon=1.0
    )
    dqn_agent.batch_size = 32
    dqn_agent.epsilon_decay = 0.9995
    dqn_agent.update_target_every = 3
    dqn_agent.learning_starts = 100
    
    dqn_rewards_history, dqn_metrics_history, dqn_episode_lengths, dqn_errors = train(
        env, dqn_agent, num_episodes=num_episodes, max_steps=500
    )
    
    print("Evaluating Double DQN agent...")
    dqn_eval_metrics = evaluate(env, dqn_agent, num_episodes=eval_episodes, max_steps=500)
    results['dqn'] = dqn_eval_metrics
    
    # Random agent
    print("Evaluating Random agent...")
    random_agent = RandomAgent(env)
    random_metrics = evaluate(env, random_agent, num_episodes=eval_episodes, max_steps=500,
                             episode_lengths_override=td_eval_metrics['episode_lengths'])
    results['random'] = random_metrics
    
    return results

def main():
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print("COMPREHENSIVE AGENT COMPARISON: GAMMA vs FIXED DEMAND")
    print("="*80)
    
    # Experiment 1: Gamma Distribution Demand
    print("\n" + "="*50)
    print("EXPERIMENT 1: GAMMA DISTRIBUTION DEMAND")
    print("="*50)
    
    env_gamma = InventoryEnvironment()
    print("\n[INFO] Testing with gamma distribution demand (default)...")
    for sku_id, sku in env_gamma.skus.items():
        print(f"SKU {sku_id}: alpha={sku.alpha}, beta={sku.beta}")
    
    gamma_results = run_agent_experiments(env_gamma, results_dir, num_episodes=5000, eval_episodes=100)
    
    # Experiment 2: Fixed Demand
    print("\n" + "="*50)
    print("EXPERIMENT 2: FIXED DEMAND")
    print("="*50)
    
    fixed_config = env_gamma.config.copy()
    fixed_config['use_fixed_demand'] = True
    fixed_config['fixed_daily_demand'] = 5
    env_fixed = InventoryEnvironment(config=fixed_config)
    print(f"\n[INFO] Testing with fixed demand (daily demand = {fixed_config['fixed_daily_demand']})...")
    
    fixed_results = run_agent_experiments(env_fixed, results_dir, num_episodes=5000, eval_episodes=100)
    
    # Create comparison plots
    print("\n" + "="*50)
    print("GENERATING COMPARISON PLOTS")
    print("="*50)
    
    plot_demand_comparison(gamma_results, fixed_results, results_dir)
    plot_combined_histograms(gamma_results, fixed_results, results_dir)
    
    # Save all results
    all_results = {
        'gamma_demand': {
            'q_learning': convert_to_serializable(gamma_results['q_learning']),
            'dqn': convert_to_serializable(gamma_results['dqn']),
            'random': convert_to_serializable(gamma_results['random'])
        },
        'fixed_demand': {
            'q_learning': convert_to_serializable(fixed_results['q_learning']),
            'dqn': convert_to_serializable(fixed_results['dqn']),
            'random': convert_to_serializable(fixed_results['random'])
        }
    }
    
    with open(os.path.join(results_dir, 'comprehensive_comparison_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*80)
    
    print("\nGAMMA DEMAND RESULTS:")
    print("-" * 40)
    for agent_name, metrics in gamma_results.items():
        agent_display = agent_name.replace('_', ' ').title()
        print(f"{agent_display}:")
        print(f"  Average Reward: {np.mean(metrics['rewards']):.2f}")
        print(f"  Service Level: {np.mean(metrics['service_levels']):.1f}%")
        print(f"  Episode Length: {np.mean(metrics['episode_lengths']):.2f}")
        print(f"  Total Stockouts: {sum([np.mean(metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']]):.2f}")
        print()
    
    print("FIXED DEMAND RESULTS:")
    print("-" * 40)
    for agent_name, metrics in fixed_results.items():
        agent_display = agent_name.replace('_', ' ').title()
        print(f"{agent_display}:")
        print(f"  Average Reward: {np.mean(metrics['rewards']):.2f}")
        print(f"  Service Level: {np.mean(metrics['service_levels']):.1f}%")
        print(f"  Episode Length: {np.mean(metrics['episode_lengths']):.2f}")
        print(f"  Total Stockouts: {sum([np.mean(metrics['location_stockouts'][loc]) for loc in ['Location_1', 'Location_2', 'Location_3', 'Retail']]):.2f}")
        print()
    
    print("PERFORMANCE COMPARISON (Gamma vs Fixed):")
    print("-" * 40)
    for agent_name in ['q_learning', 'dqn', 'random']:
        agent_display = agent_name.replace('_', ' ').title()
        gamma_reward = np.mean(gamma_results[agent_name]['rewards'])
        fixed_reward = np.mean(fixed_results[agent_name]['rewards'])
        improvement = ((gamma_reward - fixed_reward) / fixed_reward) * 100 if fixed_reward != 0 else 0
        
        print(f"{agent_display}:")
        print(f"  Reward improvement with gamma: {improvement:+.1f}%")
        
        gamma_service = np.mean(gamma_results[agent_name]['service_levels'])
        fixed_service = np.mean(fixed_results[agent_name]['service_levels'])
        service_diff = gamma_service - fixed_service
        
        print(f"  Service level difference: {service_diff:+.1f}%")
        print()
    
    print(f"\nAll results saved in: {results_dir}")
    print("Key files generated:")
    print("- gamma_vs_fixed_demand_comparison.png")
    print("- combined_reward_histograms.png")
    print("- comprehensive_comparison_results.json")
    print("="*80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main() 
