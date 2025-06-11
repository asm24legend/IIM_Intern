import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime, timedelta
import seaborn as sns
from collections import defaultdict

def load_results(results_dir):
    """Load training and evaluation results from the specified directory"""
    with open(os.path.join(results_dir, 'training_results.json'), 'r') as f:
        training_results = json.load(f)
    with open(os.path.join(results_dir, 'evaluation_results.json'), 'r') as f:
        eval_results = json.load(f)
    return training_results, eval_results

def calculate_six_month_metrics(metrics_history):
    """Calculate metrics over a 6-month period (180 days)"""
    # Assuming each step is one day
    days = 180
    
    # Initialize metric containers
    monthly_stockouts = defaultdict(list)
    monthly_inventory = defaultdict(lambda: defaultdict(list))
    monthly_costs = defaultdict(list)
    
    # Process last 180 days of data
    for day in range(min(days, len(metrics_history))):
        metrics = metrics_history[-(day+1)]  # Start from most recent
        month = day // 30  # Group by month (30-day periods)
        
        # Track stockouts
        monthly_stockouts[month].append(metrics.get('stockouts', 0))
        
        # Track inventory levels
        for sku_type in ['Type_A', 'Type_B', 'Type_C']:
            warehouse_level = metrics.get('warehouse_inventory', {}).get(sku_type, 0)
            retail_level = metrics.get('retail_inventory', {}).get(sku_type, 0)
            monthly_inventory[month][sku_type].append({
                'warehouse': warehouse_level,
                'retail': retail_level,
                'total': warehouse_level + retail_level
            })
        
        # Calculate costs
        holding_cost = sum(
            metrics.get('warehouse_inventory', {}).get(sku_type, 0) * 2 +
            metrics.get('retail_inventory', {}).get(sku_type, 0) * 1.5
            for sku_type in ['Type_A', 'Type_B', 'Type_C']
        )
        stockout_cost = metrics.get('stockouts', 0) * 50
        order_cost = metrics.get('order_cost', 0)
        total_cost = holding_cost + stockout_cost + order_cost
        monthly_costs[month].append(total_cost)
    
    return {
        'stockouts': monthly_stockouts,
        'inventory': monthly_inventory,
        'costs': monthly_costs
    }

def plot_six_month_analysis(metrics, save_dir):
    """Create visualizations for 6-month performance analysis"""
    # Set up the style
    plt.style.use('default')  
    colors = ['#FF9999', '#66B2FF', '#99FF99']  # Custom color palette
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = plt.GridSpec(3, 2, figure=fig)
    
    # 1. Average Stockouts by Month
    ax1 = fig.add_subplot(gs[0, 0])
    months = range(1, 7)
    avg_stockouts = [np.mean(metrics['stockouts'][m]) for m in range(6)]
    ax1.bar(months, avg_stockouts)
    ax1.set_title('Average Daily Stockouts by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Stockouts')
    
    # 2. Total Costs by Month
    ax2 = fig.add_subplot(gs[0, 1])
    monthly_costs = [np.sum(metrics['costs'][m]) for m in range(6)]
    ax2.bar(months, monthly_costs)
    ax2.set_title('Total Monthly Costs')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Cost ($)')
    for i, cost in enumerate(monthly_costs):
        ax2.text(i+1, cost, f'${cost:,.0f}', ha='center', va='bottom')
    
    # 3. Inventory Levels Over Time (Warehouse)
    ax3 = fig.add_subplot(gs[1, :])
    for sku_type in ['Type_A', 'Type_B', 'Type_C']:
        warehouse_levels = []
        for month in range(6):
            monthly_avg = np.mean([
                day['warehouse'] 
                for day in metrics['inventory'][month][sku_type]
            ])
            warehouse_levels.append(monthly_avg)
        ax3.plot(months, warehouse_levels, marker='o', label=f'{sku_type} Warehouse')
    ax3.set_title('Average Monthly Warehouse Inventory Levels')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Units')
    ax3.legend()
    
    # 4. Inventory Levels Over Time (Retail)
    ax4 = fig.add_subplot(gs[2, :])
    for sku_type in ['Type_A', 'Type_B', 'Type_C']:
        retail_levels = []
        for month in range(6):
            monthly_avg = np.mean([
                day['retail'] 
                for day in metrics['inventory'][month][sku_type]
            ])
            retail_levels.append(monthly_avg)
        ax4.plot(months, retail_levels, marker='o', label=f'{sku_type} Retail')
    ax4.set_title('Average Monthly Retail Inventory Levels')
    ax4.set_xlabel('Month')
    ax4.set_ylabel('Units')
    ax4.legend()
    
    # Calculate cost breakdown for table
    table_data = []
    for month in range(6):
        # Calculate average daily costs for the month
        daily_costs = defaultdict(float)
        for day in range(len(metrics['costs'][month])):
            # Holding costs
            holding_cost = sum(
                np.mean([
                    day_data['warehouse'] * 2 + day_data['retail'] * 1.5
                    for day_data in metrics['inventory'][month][sku_type]
                ])
                for sku_type in ['Type_A', 'Type_B', 'Type_C']
            )
            
            # Stockout costs
            stockout_cost = np.mean(metrics['stockouts'][month]) * 50
            
            # Order costs (remaining cost)
            total_cost = metrics['costs'][month][day]
            order_cost = total_cost - (holding_cost + stockout_cost)
            
            daily_costs['holding'] += holding_cost
            daily_costs['stockout'] += stockout_cost
            daily_costs['order'] += order_cost
            daily_costs['total'] += total_cost
        
        # Average the costs
        n_days = len(metrics['costs'][month])
        table_data.append([
            f'Month {month+1}',
            f'${daily_costs["holding"]/n_days:,.0f}',
            f'${daily_costs["stockout"]/n_days:,.0f}',
            f'${daily_costs["order"]/n_days:,.0f}',
            f'${daily_costs["total"]/n_days:,.0f}'
        ])
    
    # Add table below the plots
    ax5 = fig.add_subplot(gs[2:, :])
    ax5.axis('off')
    table = ax5.table(
        cellText=table_data,
        colLabels=['Month', 'Avg Daily\nHolding Cost', 'Avg Daily\nStockout Cost', 
                  'Avg Daily\nOrder Cost', 'Avg Daily\nTotal Cost'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'six_month_analysis.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # Find most recent results directory
    results_dirs = [d for d in os.listdir('.') if d.startswith('results_')]
    if not results_dirs:
        print("No results directories found!")
        return
    
    latest_results = max(results_dirs)
    print(f"Analyzing results from: {latest_results}")
    
    # Load results
    training_results, eval_results = load_results(latest_results)
    
    # Calculate 6-month metrics
    metrics = calculate_six_month_metrics(training_results['metrics'])
    
    # Create analysis directory
    analysis_dir = os.path.join(latest_results, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Generate visualizations
    plot_six_month_analysis(metrics, analysis_dir)
    
    # Print summary statistics
    print("\nSix Month Performance Summary:")
    print("-" * 30)
    
    # Average stockouts
    total_stockouts = sum(
        sum(stockouts) 
        for stockouts in metrics['stockouts'].values()
    )
    avg_daily_stockouts = total_stockouts / 180
    print(f"Average Daily Stockouts: {avg_daily_stockouts:.2f}")
    
    # Total costs
    total_costs = sum(
        sum(costs) 
        for costs in metrics['costs'].values()
    )
    print(f"Total 6-Month Costs: ${total_costs:,.2f}")
    print(f"Average Monthly Cost: ${total_costs/6:,.2f}")
    
    # Average inventory levels
    for sku_type in ['Type_A', 'Type_B', 'Type_C']:
        avg_warehouse = np.mean([
            np.mean([day['warehouse'] for day in month[sku_type]])
            for month in metrics['inventory'].values()
        ])
        avg_retail = np.mean([
            np.mean([day['retail'] for day in month[sku_type]])
            for month in metrics['inventory'].values()
        ])
        print(f"\n{sku_type} Average Inventory Levels:")
        print(f"  Warehouse: {avg_warehouse:.1f} units")
        print(f"  Retail: {avg_retail:.1f} units")

        

if __name__ == "__main__":
    main() 