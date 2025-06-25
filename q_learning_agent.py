import numpy as np
from collections import defaultdict
import random
import pickle

class TDAgent:
    def __init__(self, action_space, learning_rate=0.05, discount_factor=0.98, epsilon=1.0):
        """
        Initialize Double Q-Learning agent
        
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
        self.epsilon_decay = 0.999
        
        # Initialize Q-tables 
        # This will be dynamically sized in get_action or learn methods once num_skus is known
        self.q_table_A = defaultdict(lambda: np.array([]))
        self.q_table_B = defaultdict(lambda: np.array([]))
        
        # TD error history for monitoring learning
        self.td_errors = []
        
        # Number of discrete levels for order quantities and lead time reductions
        self.num_order_levels = 20
        self.num_lead_time_levels = 4  # 0 to max_lead_time_reduction
        
        # Define actual discrete values for order quantities and lead time reductions
        # These depend on the environment's action space (max_inventory, max_lead_time_reduction)
        # Will be initialized more precisely when environment is available.
        self.order_level_values = None
        self.lead_time_reduction_values = None
    
    def discretize_state(self, state):
        """
        Convert continuous state values to discrete for Q-table lookup.
        State is now just an array of inventory levels.
        """
        # Discretize inventory levels into bins of 25 units
        discrete_state = tuple(int(s / 25) for s in state)
        return discrete_state
    
    def get_action(self, state, env, greedy=False):
        """
        Get action using epsilon-greedy policy (or greedy if specified)
        Returns both order quantities and lead time reduction decisions
        """
        num_skus = len(env.skus)
        if self.order_level_values is None:
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        if self.lead_time_reduction_values is None:
            self.lead_time_reduction_values = np.arange(0, self.num_lead_time_levels, dtype=np.int32)
        expected_q_table_size = num_skus * self.num_order_levels + num_skus * self.num_lead_time_levels
        discrete_state = self.discretize_state(state)
        for q_table in [self.q_table_A, self.q_table_B]:
            if len(q_table[discrete_state]) != expected_q_table_size:
                q_table[discrete_state] = np.zeros(expected_q_table_size)
        # Only use epsilon-greedy if not greedy mode
        if not greedy and np.random.random() < self.epsilon:
            random_order_levels = np.random.randint(0, self.num_order_levels, num_skus)
            random_lead_time_levels = np.random.randint(0, self.num_lead_time_levels, num_skus)
            order_quantities = self.order_level_values[random_order_levels]
            lead_time_reductions = self.lead_time_reduction_values[random_lead_time_levels]
            return np.concatenate([order_quantities, lead_time_reductions])
        # Use sum of Q-tables for action selection
        q_values = self.q_table_A[discrete_state] + self.q_table_B[discrete_state]
        best_orders = np.zeros(num_skus)
        best_reductions = np.zeros(num_skus)
        for i in range(num_skus):
            order_q_start_idx = i * self.num_order_levels
            order_q_end_idx = order_q_start_idx + self.num_order_levels
            sku_order_q_values = q_values[order_q_start_idx:order_q_end_idx]
            lead_time_q_start_idx = num_skus * self.num_order_levels + i * self.num_lead_time_levels
            lead_time_q_end_idx = lead_time_q_start_idx + self.num_lead_time_levels
            sku_lead_time_q_values = q_values[lead_time_q_start_idx:lead_time_q_end_idx]
            best_order_level_idx = np.argmax(sku_order_q_values)
            best_lead_time_level_idx = np.argmax(sku_lead_time_q_values)
            best_orders[i] = self.order_level_values[best_order_level_idx]
            best_reductions[i] = self.lead_time_reduction_values[best_lead_time_level_idx]
        return np.concatenate([best_orders, best_reductions])
    
    def learn(self, state, action, reward, next_state):
        """
        Update Q-values using Q-learning (off-policy TD learning)
        """
        current_state = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)
        
        # Ensure next_q_values are initialized if next_state is new
        num_skus = len(state) // 2 # Assuming state is [warehouse_sku1, retail_sku1, ...]
        expected_q_table_size = num_skus * self.num_order_levels + num_skus * self.num_lead_time_levels
        
        for q_table in [self.q_table_A, self.q_table_B]:
            if len(q_table[next_state_discrete]) != expected_q_table_size:
                q_table[next_state_discrete] = np.zeros(expected_q_table_size)
        for q_table in [self.q_table_A, self.q_table_B]:
            if len(q_table[current_state]) != expected_q_table_size:
                q_table[current_state] = np.zeros(expected_q_table_size)
        next_q_values = self.q_table_A[next_state_discrete] + self.q_table_B[next_state_discrete]

        current_q_values = self.q_table_A[current_state] + self.q_table_B[current_state]
        
        # Identify the discrete levels that correspond to the continuous actions taken
        # This requires finding the closest discrete value for each continuous action component.
        num_skus_action = len(action) // 2 # Action is [order_qty_sku1,..., lead_time_red_sku1,...]

        chosen_order_levels_indices = np.zeros(num_skus_action, dtype=np.int32)
        chosen_lead_time_levels_indices = np.zeros(num_skus_action, dtype=np.int32)

        for i in range(num_skus_action):
            chosen_order_levels_indices[i] = np.argmin(np.abs(self.order_level_values - action[i]))
            chosen_lead_time_levels_indices[i] = np.argmin(np.abs(self.lead_time_reduction_values - action[num_skus_action + i]))

        total_td_error = 0
        
        # Randomly choose which Q-table to update
        if random.random() < 0.5:
            Q_main, Q_target = self.q_table_A, self.q_table_B
        else:
            Q_main, Q_target = self.q_table_B, self.q_table_A
        
        for i in range(num_skus_action):
            # Update Q-value for order quantity decision for this SKU
            order_level_idx = chosen_order_levels_indices[i]
            q_idx_in_flat_array = i * self.num_order_levels + order_level_idx
            
            # Double Q-learning: action selection from Q_main, evaluation from Q_target
            next_q_main = Q_main[next_state_discrete][i*self.num_order_levels : (i+1)*self.num_order_levels]
            best_next_action = np.argmax(next_q_main)
            next_q_target = Q_target[next_state_discrete][i*self.num_order_levels + best_next_action]
            td_target_order = reward + self.discount_factor * next_q_target
            td_error_order = td_target_order - current_q_values[q_idx_in_flat_array]
            Q_main[current_state][q_idx_in_flat_array] += self.learning_rate * td_error_order
            total_td_error += td_error_order

            # Update Q-value for lead time reduction decision for this SKU
            lead_time_level_idx = chosen_lead_time_levels_indices[i]
            q_idx_in_flat_array_lt = num_skus_action * self.num_order_levels + i * self.num_lead_time_levels + lead_time_level_idx

            # Double Q-learning: action selection from Q_main, evaluation from Q_target
            next_q_main_lt = Q_main[next_state_discrete][num_skus_action * self.num_order_levels + i*self.num_lead_time_levels : num_skus_action * self.num_order_levels + (i+1)*self.num_lead_time_levels]
            best_next_action_lt = np.argmax(next_q_main_lt)
            next_q_target_lt = Q_target[next_state_discrete][num_skus_action * self.num_order_levels + i*self.num_lead_time_levels + best_next_action_lt]
            td_target_lead_time = reward + self.discount_factor * next_q_target_lt
            td_error_lead_time = td_target_lead_time - current_q_values[q_idx_in_flat_array_lt]
            Q_main[current_state][q_idx_in_flat_array_lt] += self.learning_rate * td_error_lead_time
            total_td_error += td_error_lead_time
        
        # Average TD error across all action components (2 actions per SKU)
        avg_td_error = total_td_error / (2 * num_skus_action)
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
        """Save the Q-tables to separate .pkl files using pickle"""
        with open(filename + '_A.pkl', 'wb') as fA:
            pickle.dump(dict(self.q_table_A), fA)
        with open(filename + '_B.pkl', 'wb') as fB:
            pickle.dump(dict(self.q_table_B), fB)

    def load(self, filename):
        """Load the Q-tables from .pkl files using pickle"""
        with open(filename + '_A.pkl', 'rb') as fA:
            loaded_dict_A = pickle.load(fA)
        with open(filename + '_B.pkl', 'rb') as fB:
            loaded_dict_B = pickle.load(fB)
        self.q_table_A = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        self.q_table_B = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        self.q_table_A.update(loaded_dict_A)
        self.q_table_B.update(loaded_dict_B) 
