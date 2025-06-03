import numpy as np
from collections import defaultdict

class TDAgent:
    def __init__(self, action_space, learning_rate=0.05, discount_factor=0.99, epsilon=1.0):
        """
        Initialize Q-Learning agent
        
        Args:
            action_space: Gym space defining the action space
            learning_rate: Learning rate for Q-value updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(action_space.shape[0]))
        
        # TD error history for monitoring learning
        self.td_errors = []
    
    def discretize_state(self, state):
        """
        Convert continuous state values to discrete for Q-table lookup.
        State is now just an array of inventory levels.
        """
        # Discretize inventory levels into bins of 50 units
        return tuple(state.astype(int) // 50)
    
    def get_action(self, state, env):
        """
        Get action using epsilon-greedy policy
        """
        discrete_state = self.discretize_state(state)
        
        if np.random.random() < self.epsilon:
            # Exploration: random action
            action = np.zeros(self.action_space.shape[0])
            for i, (sku_id, sku) in enumerate(env.skus.items()):
                if state[i] < env.calculate_rop(sku_id):
                    # Calculate order quantity
                    base_order = env.calculate_eoq(sku_id)
                    action[i] = base_order * np.random.uniform(0.8, 1.2)  # Add some noise
                    action[i] = np.clip(action[i], sku.min_order_qty, 
                                      sku.max_stock - state[i])
            return action.astype(np.int32)
        else:
            # Exploitation: choose best action from Q-table
            return self.q_table[discrete_state].astype(np.int32)
    
    def learn(self, state, action, reward, next_state, next_action=None):
        """
        Update Q-values using Q-learning (off-policy TD learning)
        """
        current_state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)
        
        # Get current and next Q-values
        current_q = self.q_table[current_state]
        next_q = self.q_table[next_state]
        
        # Initialize total TD error
        total_td_error = 0
        
        # Calculate TD error for each SKU
        for i in range(len(action)):
            # Q-learning update
            td_target = reward + self.discount_factor * np.max(next_q)
            td_error = td_target - current_q[i]
            
            # Update Q-value
            current_q[i] += self.learning_rate * td_error
            total_td_error += td_error
        
        # Average TD error across all SKUs
        avg_td_error = total_td_error / len(action)
        self.td_errors.append(avg_td_error)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return avg_td_error
    
    def get_average_td_error(self, window=100):
        """Get average TD error over last n steps"""
        if len(self.td_errors) == 0:
            return 0
        return np.mean(self.td_errors[-window:])
    
    def save(self, filename):
        """Save the Q-table to a file"""
        np.save(filename, dict(self.q_table))
    
    def load(self, filename):
        """Load the Q-table from a file"""
        loaded_dict = np.load(filename, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        self.q_table.update(loaded_dict) 