#!/usr/bin/env python3
"""
Test script to verify Double DQN agent functionality
"""

import numpy as np
from inventory_env import InventoryEnvironment
from dqn_agent import DoubleDQNAgent
import time

def test_dqn_optimizations():
    """Test the optimized DQN agent to ensure it works correctly"""
    print("Testing optimized DQN agent...")
    
    # Initialize environment
    env = InventoryEnvironment()
    print(f"Environment initialized with {len(env.skus)} SKUs")
    
    # Initialize DQN agent with optimized parameters
    dqn_agent = DoubleDQNAgent(
        action_space=env.action_space,
        learning_rate=0.001,
        discount_factor=0.98,
        epsilon=1.0
    )
    
    # Test a few episodes
    num_test_episodes = 5
    total_reward = 0
    
    start_time = time.time()
    
    for episode in range(num_test_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 50:  # Short episodes for testing
            # Get action
            action = dqn_agent.get_action(state, env)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Learn
            td_error = dqn_agent.learn(state, action, reward, next_state, env)
            
            episode_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        total_reward += episode_reward
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"\nTest completed in {training_time:.2f} seconds")
    print(f"Average reward per episode: {total_reward / num_test_episodes:.2f}")
    print(f"Average TD error: {dqn_agent.get_average_td_error():.4f}")
    print(f"Final epsilon: {dqn_agent.epsilon:.4f}")
    
    # Test demand variability
    print("\nTesting demand variability...")
    for sku_id, sku in env.skus.items():
        demands = []
        for _ in range(100):
            demand = env.calculate_demand_for_period(sku, 0, 1)
            demands.append(demand)
        
        mean_demand = np.mean(demands)
        std_demand = np.std(demands)
        cv_demand = std_demand / mean_demand if mean_demand > 0 else 0
        
        print(f"{sku_id}: Mean={mean_demand:.2f}, Std={std_demand:.2f}, CV={cv_demand:.3f}")
    
    print("\nOptimization test completed successfully!")

if __name__ == "__main__":
    test_dqn_optimizations() 