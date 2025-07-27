import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import pickle
from scipy.stats import gamma
import bisect

# Define the experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done', 'episode_stockouts'))

# --- Episode-Stockout-Based Replay Buffer ---
class StockoutReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.episode_stockouts = []
        self.pos = 0

    def push(self, *args, episode_stockouts=0):
        # args should be (state, action, reward, next_state, done)
        if len(args) == 5:
            exp = Experience(args[0], args[1], args[2], args[3], args[4], episode_stockouts)
        else:
            raise ValueError("Expected 5 arguments for Experience (state, action, reward, next_state, done)")
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.episode_stockouts.append(episode_stockouts)
        else:
            self.buffer[self.pos] = exp
            self.episode_stockouts[self.pos] = episode_stockouts
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        # Prioritize experiences with <1 retail stockout
        indices_priority = [i for i, exp in enumerate(self.buffer) if exp.episode_stockouts < 1]
        indices_other = [i for i, exp in enumerate(self.buffer) if exp.episode_stockouts >= 1]
        n_priority = int(batch_size * 0.7)  # 70% from priority
        n_other = batch_size - n_priority
        if len(indices_priority) >= n_priority:
            chosen_priority = np.random.choice(indices_priority, n_priority, replace=False)
        else:
            chosen_priority = np.random.choice(indices_priority, len(indices_priority), replace=False) if indices_priority else []
            n_other = batch_size - len(chosen_priority)
        if len(indices_other) >= n_other:
            chosen_other = np.random.choice(indices_other, n_other, replace=False)
        else:
            chosen_other = np.random.choice(indices_other, len(indices_other), replace=False) if indices_other else []
        chosen_indices = np.concatenate([chosen_priority, chosen_other]) if len(chosen_priority) or len(chosen_other) else []
        if len(chosen_indices) == 0:
            chosen_indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[int(idx)] for idx in chosen_indices]
        return samples

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # Reduced from 256
        self.fc2 = nn.Linear(128, 128)         # Reduced from 256
        self.fc3 = nn.Linear(128, output_size) # Removed one layer
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        self.epsilon_min = 0.01  # Reduced from 0.05 for more exploration
        self.epsilon_decay = 0.9995  # Faster decay from 0.9997
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = None
        self.action_size = None
        
        # Optimized hyperparameters for faster training
        self.batch_size = 32  # Reduced from 64 for more frequent updates
        self.memory_size = 10000  # Reduced from default for faster learning
        self.update_target_every = 3  # More frequent target updates
        self.learning_starts = 100  # Start learning earlier
        
        # Initialize replay buffer with stockout-based prioritization
        self.memory = StockoutReplayBuffer(self.memory_size)
        
        # Networks will be initialized when we know the state size
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Tracking
        self.td_errors = []
        self.update_count = 0
    
    def initialize_networks(self, state_size, num_skus):
        """Initialize Q-network and target network"""
        # Simplified action space: discrete actions for each SKU
        # Each SKU can have actions from 0 to max_inventory in steps
        max_inventory = 1000  # This should match env.config['max_inventory']
        action_size = 21  # 0, 50, 100, ..., 1000 (21 discrete levels)
        
        self.q_network = DQNNetwork(state_size, action_size * num_skus).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size * num_skus).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Use Adam optimizer with the specified learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        self.state_size = state_size
        self.action_size = action_size * num_skus
    
    def get_action(self, state, env, greedy=False):
        """Get action using epsilon-greedy policy"""
        if self.q_network is None:
            # Initialize networks if not done yet
            state_size = len(state)
            num_skus = len(env.skus)
            self.initialize_networks(state_size, num_skus)
        
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get Q values
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        if greedy or np.random.random() > self.epsilon:
            # Exploit: choose best action
            action_indices = q_values.argmax(1).cpu().numpy()
        else:
            # Explore: choose random action
            action_indices = np.random.randint(0, q_values.shape[1], size=q_values.shape[0])
        
        # Convert action indices to actual order quantities
        actions = []
        for i, action_idx in enumerate(action_indices):
            # Map action index to order quantity (0 to max_inventory)
            order_qty = int(action_idx * env.action_space.high[0] / (q_values.shape[1] - 1))
            actions.append(order_qty)
        
        return np.array(actions, dtype=np.int32)
    
    def learn(self, state, action, reward, next_state, env=None, episode_stockouts=0):
        """Learn from experience using Double DQN"""
        # Store experience in replay buffer
        self.memory.push(state, action, reward, next_state, False, episode_stockouts)
        
        # Only learn if we have enough samples and have started learning
        if len(self.memory) < self.learning_starts or len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        if not batch:
            return 0.0
        
        # Prepare batch data
        states = torch.FloatTensor([exp.state for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in batch]).to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.discount_factor * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss and update
        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network more frequently
        self.update_count += 1
        if self.update_count % self.update_target_every == 0:
            self.soft_update()
        
        # Track TD error for monitoring
        td_error = loss.item()
        self.td_errors.append(td_error)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return td_error
    
    def soft_update(self):
        """Soft update target network using Polyak averaging"""
        if self.q_network is not None and self.target_network is not None:
            tau = 0.005  # Small update rate for stability
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
    
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