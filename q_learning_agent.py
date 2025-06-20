import numpy as np
from collections import defaultdict

class TDAgent:
    def __init__(self, action_space, learning_rate=0.01, discount_factor=0.99, epsilon=1.0):
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
        # This will be dynamically sized in get_action or learn methods once num_skus is known
        self.q_table = defaultdict(lambda: np.array([]))
        
        # TD error history for monitoring learning
        self.td_errors = []
        
        # Number of discrete levels for order quantities and lead time reductions
        self.num_order_levels = 10
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
        # Discretize inventory levels into bins of 50 units
        discrete_state = tuple(int(s / 50) for s in state)
        return discrete_state
    
    def get_action(self, state, env):
        """
        Get action using epsilon-greedy policy
        Returns both order quantities and lead time reduction decisions
        """
        num_skus = len(env.skus)

        # Initialize order and lead time reduction values if not already done
        if self.order_level_values is None:
            # Assuming action_space.high[0] gives max_inventory for order quantities
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        if self.lead_time_reduction_values is None:
            # Assuming action_space.high[-1] gives max_lead_time_reduction
            self.lead_time_reduction_values = np.arange(0, self.num_lead_time_levels, dtype=np.int32)

        # Dynamically size the Q-table for this state if not already sized
        expected_q_table_size = num_skus * self.num_order_levels + num_skus * self.num_lead_time_levels
        discrete_state = self.discretize_state(state)
        if len(self.q_table[discrete_state]) != expected_q_table_size:
            self.q_table[discrete_state] = np.zeros(expected_q_table_size)


        if np.random.random() < self.epsilon:
            # Random action
            random_order_levels = np.random.randint(0, self.num_order_levels, num_skus)
            random_lead_time_levels = np.random.randint(0, self.num_lead_time_levels, num_skus)

            order_quantities = self.order_level_values[random_order_levels]
            lead_time_reductions = self.lead_time_reduction_values[random_lead_time_levels]

            return np.concatenate([order_quantities, lead_time_reductions])
        
        q_values = self.q_table[discrete_state]
        
        best_orders = np.zeros(num_skus)
        best_reductions = np.zeros(num_skus)
        
        for i in range(num_skus):
            # Extract Q-values for order quantity levels for current SKU
            order_q_start_idx = i * self.num_order_levels
            order_q_end_idx = order_q_start_idx + self.num_order_levels
            sku_order_q_values = q_values[order_q_start_idx:order_q_end_idx]

            # Extract Q-values for lead time reduction levels for current SKU
            lead_time_q_start_idx = num_skus * self.num_order_levels + i * self.num_lead_time_levels
            lead_time_q_end_idx = lead_time_q_start_idx + self.num_lead_time_levels
            sku_lead_time_q_values = q_values[lead_time_q_start_idx:lead_time_q_end_idx]

            # Get best discrete level index for each SKU
            best_order_level_idx = np.argmax(sku_order_q_values)
            best_lead_time_level_idx = np.argmax(sku_lead_time_q_values)

            # Convert discrete level to actual value
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
        
        if len(self.q_table[next_state_discrete]) != expected_q_table_size:
            self.q_table[next_state_discrete] = np.zeros(expected_q_table_size)
        next_q_values = self.q_table[next_state_discrete]

        current_q_values = self.q_table[current_state]
        
        # Identify the discrete levels that correspond to the continuous actions taken
        # This requires finding the closest discrete value for each continuous action component.
        num_skus_action = len(action) // 2 # Action is [order_qty_sku1,..., lead_time_red_sku1,...]

        chosen_order_levels_indices = np.zeros(num_skus_action, dtype=np.int32)
        chosen_lead_time_levels_indices = np.zeros(num_skus_action, dtype=np.int32)

        for i in range(num_skus_action):
            chosen_order_levels_indices[i] = np.argmin(np.abs(self.order_level_values - action[i]))
            chosen_lead_time_levels_indices[i] = np.argmin(np.abs(self.lead_time_reduction_values - action[num_skus_action + i]))

        total_td_error = 0
        
        for i in range(num_skus_action):
            # Update Q-value for order quantity decision for this SKU
            order_level_idx = chosen_order_levels_indices[i]
            q_idx_in_flat_array = i * self.num_order_levels + order_level_idx
            
            # Max Q-value for the next state and this SKU's order levels
            max_next_q_order = np.max(next_q_values[i*self.num_order_levels : (i+1)*self.num_order_levels])
            td_target_order = reward + self.discount_factor * max_next_q_order
            td_error_order = td_target_order - current_q_values[q_idx_in_flat_array]
            current_q_values[q_idx_in_flat_array] += self.learning_rate * td_error_order
            total_td_error += td_error_order

            # Update Q-value for lead time reduction decision for this SKU
            lead_time_level_idx = chosen_lead_time_levels_indices[i]
            q_idx_in_flat_array_lt = num_skus_action * self.num_order_levels + i * self.num_lead_time_levels + lead_time_level_idx

            # Max Q-value for the next state and this SKU's lead time reduction levels
            max_next_q_lt = np.max(next_q_values[num_skus_action * self.num_order_levels + i*self.num_lead_time_levels : num_skus_action * self.num_order_levels + (i+1)*self.num_lead_time_levels])
            td_target_lead_time = reward + self.discount_factor * max_next_q_lt
            td_error_lead_time = td_target_lead_time - current_q_values[q_idx_in_flat_array_lt]
            current_q_values[q_idx_in_flat_array_lt] += self.learning_rate * td_error_lead_time
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
        """Save the Q-table to a file"""
        np.save(filename, dict(self.q_table))
    
    def load(self, filename):
        """Load the Q-table from a file"""
        loaded_dict = np.load(filename, allow_pickle=True).item()
        self.q_table = defaultdict(lambda: np.zeros(self.action_space.shape[0]))
        self.q_table.update(loaded_dict) 
