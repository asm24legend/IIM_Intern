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
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.997
        
        # Initialize Q-table
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        
        # TD error history for monitoring learning
        self.td_errors = []
        
        # Number of discrete levels for order quantities and lead time reductions
        self.num_order_levels = 10
        self.num_lead_time_levels = 4  # 0 to max_lead_time_reduction
        
        # Calculate the size of each discrete level
        self.order_level_size = self.action_space.high[0] / self.num_order_levels
    
    def discretize_state(self, state):
        """
        Convert continuous state values to discrete for Q-table lookup.
        State is now just an array of inventory levels.
        """
        # Discretize inventory levels into bins of 50 units
        discrete_state = tuple(int(s / 50) for s in state)
        return discrete_state
    
    def get_action(self, state, env):
        """
        Get action using epsilon-greedy policy
        Returns both order quantities and lead time reduction decisions
        """
        if np.random.random() < self.epsilon:
            # Random action
            num_skus = len(env.skus)
            order_quantities = np.random.randint(0, self.action_space.high[0], num_skus)
            lead_time_reductions = np.random.randint(0, self.action_space.high[-1] + 1, num_skus)
            return np.concatenate([order_quantities, lead_time_reductions])
        
        discrete_state = self.discretize_state(state)
        q_values = self.q_table[discrete_state]
        
        # Split Q-values into order quantities and lead time reductions
        num_skus = len(env.skus)
        order_q_values = q_values[:num_skus]
        lead_time_q_values = q_values[num_skus:]
        
        # Get best actions
        best_orders = np.zeros(num_skus)
        best_reductions = np.zeros(num_skus)
        
        for i in range(num_skus):
            # Discretize order quantity
            order_level = int(np.argmax(order_q_values[i::num_skus]))
            best_orders[i] = order_level * self.order_level_size
            
            # Get lead time reduction
            best_reductions[i] = np.argmax(lead_time_q_values[i::num_skus])
        
        return np.concatenate([best_orders, best_reductions])
    
    def learn(self, state, action, reward, next_state):
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
        
        # Calculate TD error for each action component (orders and lead time reductions)
        num_actions = len(action)
        for i in range(num_actions):
            # Q-learning update
            td_target = reward + self.discount_factor * np.max(next_q)
            td_error = td_target - current_q[i]
            
            # Update Q-value
            current_q[i] += self.learning_rate * td_error
            total_td_error += td_error
        
        # Average TD error across all action components
        avg_td_error = total_td_error / num_actions
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