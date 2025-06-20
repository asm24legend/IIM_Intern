import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

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
        self.num_action_levels = 10
        
        # Define actual discrete values for order quantities and lead time reductions
        # These depend on the environment's action space (max_inventory, max_lead_time_reduction)
        # Will be initialized more precisely when environment is available.
        self.action_level_values = None
    
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
        if self.action_level_values is None:
            # Assuming action_space.high[0] gives max_inventory for order quantities
            self.action_level_values = np.linspace(0, env.action_space.high[0], self.num_action_levels, dtype=np.int32)

        # Dynamically size the Q-table for this state if not already sized
        expected_q_table_size = num_skus * self.num_action_levels * 2  # [order_qty, on_hand] for each SKU
        discrete_state = self.discretize_state(state)
        if len(self.q_table[discrete_state]) != expected_q_table_size:
            self.q_table[discrete_state] = np.zeros(expected_q_table_size)


        if np.random.random() < self.epsilon:
            # Random action
            random_levels = np.random.randint(0, self.num_action_levels, num_skus * 2)
            action = self.action_level_values[random_levels]
            return action
        
        q_values = self.q_table[discrete_state]
        
        best_action = np.zeros(num_skus * 2)
        
        for i in range(num_skus * 2):
            # Extract Q-values for order quantity levels for current SKU
            idx_start = i * self.num_action_levels
            idx_end = idx_start + self.num_action_levels
            action_qs = q_values[idx_start:idx_end]

            # Get best discrete level index for each SKU
            best_idx = np.argmax(action_qs)

            # Convert discrete level to actual value
            best_action[i] = self.action_level_values[best_idx]
        
        return best_action
    
    def learn(self, state, action, reward, next_state):
        """
        Update Q-values using Q-learning (off-policy TD learning)
        """
        current_state = self.discretize_state(state)
        next_state_discrete = self.discretize_state(next_state)
        
        # Ensure next_q_values are initialized if next_state is new
        num_skus = len(state) // 2 # Assuming state is [warehouse_sku1, retail_sku1, ...]
        expected_q_table_size = num_skus * self.num_action_levels * 2
        
        if len(self.q_table[next_state_discrete]) != expected_q_table_size:
            self.q_table[next_state_discrete] = np.zeros(expected_q_table_size)
        next_q_values = self.q_table[next_state_discrete]

        current_q_values = self.q_table[current_state]
        
        # Identify the discrete levels that correspond to the continuous actions taken
        # This requires finding the closest discrete value for each continuous action component.
        num_skus_action = len(action) // 2 # Action is [order_qty_sku1,..., on_hand_sku1,...]

        chosen_action_indices = np.zeros(num_skus_action, dtype=np.int32)

        for i in range(num_skus_action):
            chosen_action_indices[i] = np.argmin(np.abs(self.action_level_values - action[i]))

        total_td_error = 0
        
        for i in range(num_skus_action):
            # Update Q-value for order quantity decision for this SKU
            action_idx = chosen_action_indices[i]
            q_idx = i * self.num_action_levels + action_idx
            
            # Max Q-value for the next state and this SKU's order levels
            max_next_q = np.max(next_q_values[i*self.num_action_levels : (i+1)*self.num_action_levels])
            td_target = reward + self.discount_factor * max_next_q
            td_error = td_target - current_q_values[q_idx]
            current_q_values[q_idx] += self.learning_rate * td_error
            total_td_error += td_error

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

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, action_space, learning_rate=1e-3, discount_factor=0.99, epsilon=1.0, batch_size=64, buffer_size=10000, target_update=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space = action_space
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.target_update = target_update
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=buffer_size)
        self.td_errors = []
        self.learn_step = 0

    def get_action(self, state, env=None):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        if np.random.rand() < self.epsilon:
            # Sample random action from action_space
            action = self.action_space.sample()
            return action
        with torch.no_grad():
            q_values = self.policy_net(state).cpu().numpy().flatten()
        # For each action dimension, pick the closest valid action in the space
        # (Assume action_space is Box, so clip to bounds)
        action = np.clip(q_values, self.action_space.low, self.action_space.high)
        return action

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, state, action, reward, next_state, done=False):
        self.store(state, action, reward, next_state, done)
        if len(self.memory) < self.batch_size:
            return 0.0
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # Q(s,a) for taken actions
        q_values = self.policy_net(states)
        # For continuous actions, treat Q(s,a) as the value for the action vector
        # Use MSE between predicted Q and target Q
        # For each sample, get Q-value for the action taken (closest index)
        # Here, we use the sum Q-value as a proxy (since action is continuous)
        q_pred = q_values.sum(dim=1, keepdim=True)

        # Q(s',a') for next state (max over actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q = next_q_values.sum(dim=1, keepdim=True)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_pred, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_error = (q_pred - target_q).abs().mean().item()
        self.td_errors.append(td_error)

        # Update target network
        self.learn_step += 1
        if self.learn_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return td_error

    def get_average_td_error(self, window=100):
        if len(self.td_errors) == 0:
            return 0
        return np.mean(self.td_errors[-window:])

    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename + '_dqn.pt')

    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename + '_dqn.pt', map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
