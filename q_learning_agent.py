import numpy as np
from collections import defaultdict
import random
import pickle
from scipy.stats import gamma

class TDAgent:
    def __init__(self, action_space, discount_factor=0.99, learning_rate=0.1, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9997):
        """
        Initialize Q-Learning agent
        
        Args:
            action_space: Gym space defining the action space
            discount_factor: Discount factor for future rewards
            learning_rate: Learning rate for Q-value updates
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration rate decay
        """
        self.action_space = action_space
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Number of discrete levels for order quantities and lead time reductions
        self.num_order_levels = 20
        
        # Define actual discrete values for order quantities and lead time reductions
        # These depend on the environment's action space (max_inventory, max_lead_time_reduction)
        # Will be initialized more precisely when environment is available.
        self.order_level_values = None
        self.q_table = {}
        
        # TD error history for monitoring learning
        self.td_errors = []
    
    def get_demand_stats(self, env):
        """Return a vector of 90th percentile lead time demand for each SKU."""
        demand_stats = []
        for sku_id, sku in env.skus.items():
            quantile = 0.9
            shape = sku.alpha * sku.lead_time_days
            scale = sku.beta
            demand90 = gamma.ppf(quantile, a=shape, scale=scale)
            demand_stats.append(demand90)
        return np.array(demand_stats)

    def discretize_state(self, state, env=None):
        """
        Convert continuous state values to discrete for Q-table lookup.
        State is now an array of inventory levels plus demand stats if env is provided.
        """
        max_inventory = 1000
        state_arr = np.array(state)
        if env is not None:
            demand_stats = self.get_demand_stats(env)
            state_arr = np.concatenate([state_arr, demand_stats])
        normalized_state = tuple((state_arr / max_inventory).round(3))
        return normalized_state
    
    def get_action(self, state, env, greedy=False):
        """
        Get action using epsilon-greedy policy (or greedy if specified)
        Returns both order quantities and lead time reduction decisions
        """
        num_skus = len(env.skus)
        if self.order_level_values is None:
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        current_state = self.discretize_state(state, env)
        # Per-SKU action selection: each SKU's action is chosen independently
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(num_skus * self.num_order_levels)
        if not greedy and np.random.random() < self.epsilon:
            random_order_levels = np.random.randint(0, self.num_order_levels, num_skus)
            order_quantities = self.order_level_values[random_order_levels]
            return order_quantities
        q_values = self.q_table[current_state]
        best_orders = np.zeros(num_skus)
        for i in range(num_skus):
            order_q_start_idx = i * self.num_order_levels
            order_q_end_idx = order_q_start_idx + self.num_order_levels
            sku_order_q_values = q_values[order_q_start_idx:order_q_end_idx]
            best_order_level_idx = np.argmax(sku_order_q_values)
            best_orders[i] = self.order_level_values[best_order_level_idx]
        return best_orders
    
    def learn(self, state, action, reward, next_state, env=None):
        """
        Update Q-values using standard Q-learning (off-policy TD learning)
        """
        current_state = self.discretize_state(state, env)
        next_state_discrete = self.discretize_state(next_state, env)
        num_skus = len(action)
        if current_state not in self.q_table:
            self.q_table[current_state] = np.zeros(num_skus * self.num_order_levels)
        if next_state_discrete not in self.q_table:
            self.q_table[next_state_discrete] = np.zeros(num_skus * self.num_order_levels)
        q_values = self.q_table[current_state]
        next_q_values = self.q_table[next_state_discrete]
        total_td_error = 0
        # Per-SKU Q-learning update: each SKU's Q-value is updated independently
        for i in range(num_skus):
            order_level_idx = np.argmin(np.abs(self.order_level_values - action[i]))
            q_idx_in_flat_array = i * self.num_order_levels + order_level_idx
            next_q_order = next_q_values[i*self.num_order_levels : (i+1)*self.num_order_levels]
            max_next_q_order = np.max(next_q_order)
            td_target_order = reward + self.discount_factor * max_next_q_order
            td_error_order = td_target_order - q_values[q_idx_in_flat_array]
            q_values[q_idx_in_flat_array] += self.learning_rate * td_error_order
            total_td_error += td_error_order
        avg_td_error = total_td_error / num_skus
        self.td_errors.append(avg_td_error)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return avg_td_error
    
    def get_average_td_error(self, window=100):
        """Get average TD error over last n steps"""
        if len(self.td_errors) == 0:
            return 0
        return np.mean(self.td_errors[-window:])
    
    def save(self, filename):
        """Save the Q-table to a .pkl file using pickle"""
        with open(filename + '.pkl', 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filename):
        """Load the Q-table from a .pkl file using pickle"""
        with open(filename + '.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        self.q_table.update(loaded_dict) 
