import numpy as np
from inventory_env import InventoryEnvironment
from dqn_agent import DoubleDQNAgent
from main import train, generate_comprehensive_report
import os
from datetime import datetime

def test_enhanced_metrics():
    """Test the enhanced metrics collection and reporting system"""
    print("Testing Enhanced Metrics System...")
    
    # Create environment and agent
    env = InventoryEnvironment()
    agent = DoubleDQNAgent(env.action_space)
    
    # Run a short training session to test metrics collection
    print("\nRunning short training session (10 episodes)...")
    rewards_history, metrics_history, episode_lengths, td_errors = train(
        env, agent, num_episodes=10, max_steps=100
    )
    
    # Test comprehensive report generation
    print("\nGenerating comprehensive report...")
    test_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = generate_comprehensive_report(metrics_history, rewards_history, td_errors, test_dir)
    
    # Verify metrics structure
    print("\nVerifying metrics structure...")
    if len(metrics_history) > 0:
        sample_metrics = metrics_history[0]
        required_keys = [
            'sku_service_levels', 'sku_stockouts', 'sku_demands', 'sku_orders',
            'inventory_turnover', 'avg_order_quantity', 'supplier_reliability_impact',
            'state_utilization', 'lead_time_violations', 'safety_stock_violations'
        ]
        
        missing_keys = [key for key in required_keys if key not in sample_metrics]
        if missing_keys:
            print(f"❌ Missing metrics: {missing_keys}")
        else:
            print("✅ All required metrics present")
        
        # Check per-SKU metrics
        for sku_id in ['Type_A', 'Type_B', 'Type_C']:
            if sku_id in sample_metrics['sku_service_levels']:
                print(f"✅ {sku_id} metrics collected")
            else:
                print(f"❌ {sku_id} metrics missing")
    
    # Verify report structure
    print("\nVerifying report structure...")
    required_report_sections = [
        'training_summary', 'service_level_analysis', 'inventory_performance',
        'stockout_analysis', 'per_sku_performance', 'learning_analysis'
    ]
    
    missing_sections = [section for section in required_report_sections if section not in report]
    if missing_sections:
        print(f"❌ Missing report sections: {missing_sections}")
    else:
        print("✅ All report sections present")
    
    # Check file generation
    print("\nChecking generated files...")
    expected_files = [
        'comprehensive_training_progress.png',
        'comprehensive_analysis.png',
        'comprehensive_report.json',
        'detailed_metrics.csv'
    ]
    
    for file_name in expected_files:
        file_path = os.path.join(test_dir, file_name)
        if os.path.exists(file_path):
            print(f"✅ {file_name} generated")
        else:
            print(f"❌ {file_name} missing")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("ENHANCED METRICS TEST SUMMARY")
    print("="*50)
    print(f"Episodes trained: {len(rewards_history)}")
    print(f"Average reward: {np.mean(rewards_history):.2f}")
    print(f"Average service level: {report['service_level_analysis']['avg_service_level']:.1f}%")
    print(f"Average inventory turnover: {report['inventory_performance']['avg_turnover']:.2f}")
    print(f"State space utilization: {report['inventory_performance']['state_utilization']:.1f}%")
    print(f"Learning status: {report['learning_analysis']['convergence_indicator']}")
    
    print("\nPer-SKU Performance:")
    for sku_id, perf in report['per_sku_performance'].items():
        print(f"  {sku_id}: Service Level {perf['avg_service_level']:.1f}%, Stockouts {perf['total_stockouts']}")
    
    print("\n✅ Enhanced metrics system test completed successfully!")
    print(f"Results saved in: {test_dir}")

if __name__ == "__main__":
    test_enhanced_metrics() 