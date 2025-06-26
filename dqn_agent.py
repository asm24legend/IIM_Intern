import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import pickle

# Define the experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*batch))
        return batch
    
    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DoubleDQNAgent:
    def __init__(self, action_space, learning_rate=0.001, discount_factor=0.98, epsilon=1.0):
        """
        Initialize Double DQN agent
        
        Args:
            action_space: Gym space defining the action space
            learning_rate: Learning rate for neural network updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
        """
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        
        # Neural network parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # State and action space setup
        self.state_size = None  # Will be set when environment is available
        self.action_size = None  # Will be set when environment is available
        
        # Neural networks
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Experience replay
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        
        # Training parameters
        self.update_target_every = 100
        self.steps = 0
        
        # TD error history for monitoring learning
        self.td_errors = []
        
        # Number of discrete levels for order quantities and lead time reductions
        self.num_order_levels = 20
        self.num_lead_time_levels = 4
        
        # Define actual discrete values for order quantities and lead time reductions
        self.order_level_values = None
        self.lead_time_reduction_values = None
    
    def initialize_networks(self, state_size, num_skus):
        """Initialize neural networks once state size and num_skus are known"""
        self.state_size = state_size
        self.action_size = num_skus * self.num_order_levels + num_skus * self.num_lead_time_levels
        # Create Q-networks
        self.q_network = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
    def discretize_state(self, state):
        """Convert continuous state values to discrete for neural network input"""
        # Normalize state values to [0, 1] range
        # Assuming state contains inventory levels that can be normalized
        max_inventory = 1000  # Adjust based on your environment
        normalized_state = np.array(state) / max_inventory
        return normalized_state.astype(np.float32)
    
    def get_action(self, state, env, greedy=False):
        """
        Get action using epsilon-greedy policy (or greedy if specified)
        Returns both order quantities and lead time reduction decisions
        """
        num_skus = len(env.skus)
        # Initialize discrete action values if not done
        if self.order_level_values is None:
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        if self.lead_time_reduction_values is None:
            self.lead_time_reduction_values = np.arange(0, self.num_lead_time_levels, dtype=np.int32)
        # Initialize networks if not done
        if self.q_network is None:
            state_size = len(self.discretize_state(state))
            self.initialize_networks(state_size, num_skus)
        # Only use epsilon-greedy if not greedy mode
        if not greedy and np.random.random() < self.epsilon:
            random_order_levels = np.random.randint(0, self.num_order_levels, num_skus)
            random_lead_time_levels = np.random.randint(0, self.num_lead_time_levels, num_skus)
            order_quantities = self.order_level_values[random_order_levels]
            lead_time_reductions = self.lead_time_reduction_values[random_lead_time_levels]
            return np.concatenate([order_quantities, lead_time_reductions])
        # Use neural network to get Q-values
        discrete_state = self.discretize_state(state)
        state_tensor = torch.FloatTensor(discrete_state).unsqueeze(0).to(self.device)
        if self.q_network is None:
            raise ValueError("Q-network is not initialized.")
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            if q_values is None:
                raise ValueError("Q-network returned None.")
            q_values = q_values.cpu().numpy()[0]
        # Select best actions for each SKU
        best_orders = np.zeros(num_skus)
        best_reductions = np.zeros(num_skus)
        for i in range(num_skus):
            # For order quantities
            order_q_start_idx = i * self.num_order_levels
            order_q_end_idx = order_q_start_idx + self.num_order_levels
            sku_order_q_values = q_values[order_q_start_idx:order_q_end_idx]
            best_order_level_idx = np.argmax(sku_order_q_values)
            best_orders[i] = self.order_level_values[best_order_level_idx]
            # For lead time reductions
            lead_time_q_start_idx = num_skus * self.num_order_levels + i * self.num_lead_time_levels
            lead_time_q_end_idx = lead_time_q_start_idx + self.num_lead_time_levels
            sku_lead_time_q_values = q_values[lead_time_q_start_idx:lead_time_q_end_idx]
            best_lead_time_level_idx = np.argmax(sku_lead_time_q_values)
            best_reductions[i] = self.lead_time_reduction_values[best_lead_time_level_idx]
        return np.concatenate([best_orders, best_reductions])
    
    def learn(self, state, action, reward, next_state):
        """
        Update Q-values using Double DQN for multi-discrete actions
        """
        # Store experience in replay buffer
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Convert action to discrete indices
        num_skus = len(state) // 2
        action_indices = self._action_to_indices(action, num_skus)  # shape: [2*num_skus]
        
        self.memory.push(discrete_state, action_indices, reward, discrete_next_state, False)
        
        # Only learn if we have enough samples
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        
        # Convert batch to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)  # (batch, state_dim)
        action_batch = torch.LongTensor(np.array(batch.action)).to(self.device)  # (batch, n_action_components)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)  # (batch,)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)  # (batch, state_dim)
        done_batch = torch.BoolTensor(batch.done).to(self.device)  # (batch,)
        
        # Compute current Q values and next Q values
        if self.q_network is None:
            raise ValueError("Q-network is not initialized.")
        if self.target_network is None:
            raise ValueError("Target network is not initialized.")
        current_q_values = self.q_network(state_batch)  # (batch, action_dim)
        next_q_values = self.q_network(next_state_batch)  # (batch, action_dim)
        next_q_values_target = self.target_network(next_state_batch)  # (batch, action_dim)
        
        # For each action component, gather Q-value for the action taken
        n_action_components = action_batch.shape[1]
        loss = torch.tensor(0.0, device=self.device)
        td_errors = []
        for i in range(n_action_components):
            # For each component, select the Q-value for the action taken
            current_q = current_q_values[:, i]  # (batch,)
            action_idx = action_batch[:, i]  # (batch,)
            current_q_selected = current_q.gather(0, action_idx.unsqueeze(0)).squeeze(0) if current_q.dim() > 1 else current_q[action_idx]
            # Double DQN: action selection from q_network, evaluation from target_network
            next_actions = next_q_values[:, i].argmax(dim=0)  # (batch,)
            next_q_selected = next_q_values_target[:, i][next_actions]
            target_q = reward_batch + (self.discount_factor * next_q_selected * (~done_batch).float())
            # Compute loss for this component
            component_loss = F.mse_loss(current_q_selected, target_q)
            loss = loss + component_loss
            td_errors.append((current_q_selected - target_q).detach().cpu().numpy())
        loss = loss / n_action_components
        
        # Optimize the model
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store mean TD error for monitoring
        mean_td_error = np.mean([np.abs(e).mean() for e in td_errors])
        self.td_errors.append(mean_td_error)
        
        return mean_td_error
    
    def _action_to_indices(self, action, num_skus):
        """Convert continuous action to discrete indices for neural network"""
        indices = []
        for i in range(num_skus):
            # Order quantity index
            order_idx = np.argmin(np.abs(self.order_level_values - action[i]))
            indices.append(order_idx)
            
            # Lead time reduction index
            lead_time_idx = np.argmin(np.abs(self.lead_time_reduction_values - action[num_skus + i]))
            indices.append(lead_time_idx)
        
        return indices
    
    def get_average_td_error(self, window=100):
        """Get average TD error over last n steps"""
        if len(self.td_errors) == 0:
            return 0
        return np.mean(self.td_errors[-window:])
    
    def save(self, filename):
        """Save the neural networks"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict() if self.q_network is not None else None,
            'target_network_state_dict': self.target_network.state_dict() if self.target_network is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
            'epsilon': self.epsilon,
            'td_errors': self.td_errors,
            'order_level_values': self.order_level_values,
            'lead_time_reduction_values': self.lead_time_reduction_values
        }, filename + '_dqn.pkl')
    
    def load(self, filename):
        """Load the neural networks"""
        checkpoint = torch.load(filename + '_dqn.pkl', map_location=self.device)
        
        if self.q_network is None:
            # Initialize networks first
            dummy_state = np.zeros(10)  # Temporary state size
            self.initialize_networks(len(dummy_state), 1)

        if self.q_network is not None and checkpoint['q_network_state_dict'] is not None:
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        if self.target_network is not None and checkpoint['target_network_state_dict'] is not None:
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        if self.optimizer is not None and checkpoint['optimizer_state_dict'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.td_errors = checkpoint.get('td_errors', [])
        self.order_level_values = checkpoint.get('order_level_values', self.order_level_values)
        self.lead_time_reduction_values = checkpoint['lead_time_reduction_values'] 