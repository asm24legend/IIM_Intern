#!/usr/bin/env python3
"""
Test script to verify Double DQN agent functionality
"""

import numpy as np
from inventory_env import InventoryEnvironment
from dqn_agent import DoubleDQNAgent

def test_dqn_agent():
    """Test basic functionality of the Double DQN agent"""
    print("Testing Double DQN agent...")
    
    # Initialize environment
    env = InventoryEnvironment()
    
    # Initialize Double DQN agent
    agent = DoubleDQNAgent(
        action_space=env.action_space,
        discount_factor=0.99
    )
    
    # Test agent initialization
    print(f"Agent initialized successfully")
    print(f"Device: {agent.device}")
    print(f"Action space: {env.action_space}")
    
    # Test action generation
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    
    # Get action from agent
    action = agent.get_action(state, env)
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")
    
    # Test learning
    next_state, reward, done, info = env.step(action)
    td_error = agent.learn(state, action, reward, next_state)
    print(f"TD error: {td_error}")
    
    print("Double DQN agent test completed successfully!")

if __name__ == "__main__":
    test_dqn_agent() 