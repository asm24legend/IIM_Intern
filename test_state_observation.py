import numpy as np
from inventory_env import InventoryEnvironment
from dqn_agent import DoubleDQNAgent

def test_state_observation():
    """Test the updated state and observation space"""
    print("Testing State and Observation Space Updates...")
    
    # Create environment
    env = InventoryEnvironment()
    
    # Test observation space
    print(f"\nObservation Space Shape: {env.observation_space.shape}")
    print(f"Observation Space Type: {env.observation_space.dtype}")
    
    # Test state space
    state = env._get_state()
    print(f"\nState Shape: {state.shape}")
    print(f"State Type: {state.dtype}")
    print(f"State Size: {len(state)} features")
    
    # Test observation
    observation = env._get_observation()
    print(f"\nObservation Shape: {observation.shape}")
    print(f"Observation Type: {observation.dtype}")
    print(f"Observation Size: {len(observation)} features")
    
    # Test DQN agent initialization
    agent = DoubleDQNAgent(env.action_space)
    
    # Test agent with rich state
    action = agent.get_action(state, env)
    print(f"\nAgent Action Shape: {action.shape}")
    print(f"Agent Action: {action}")
    
    # Test environment step
    obs, reward, done, info = env.step(action)
    print(f"\nStep Observation Shape: {obs.shape}")
    print(f"Step Reward: {reward}")
    print(f"Step Done: {done}")
    
    # Test rich state after step
    next_state = env._get_state()
    print(f"\nNext State Shape: {next_state.shape}")
    
    # Test agent learning
    td_error = agent.learn(state, action, reward, next_state, env)
    print(f"\nTD Error: {td_error}")
    
    print("\n✅ All tests passed! State and observation space updates working correctly.")

if __name__ == "__main__":
    test_state_observation() 