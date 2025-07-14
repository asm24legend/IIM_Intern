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
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_size)
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DoubleDQNAgent:
    def __init__(self, action_space, learning_rate=0.0005, discount_factor=0.99, epsilon=1.0):
        """
        Initialize Double DQN agent
        
        Args:
            action_space: Gym space defining the action space
            learning_rate: Learning rate for neural network updates
            discount_factor: Discount factor for future rewards
            epsilon: Initial exploration rate
        """
        self.action_space = action_space
        self.learning_rate = 0.0005
        self.discount_factor = 0.99
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9997
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = None
        self.action_size = None
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.memory = StockoutReplayBuffer(500000)
        self.batch_size = 128
        self.update_target_every = 1
        self.tau = 0.005
        self.steps = 0
        self.td_errors = []
        self.num_order_levels = 20
        self.order_level_values = None
    
    def initialize_networks(self, state_size, num_skus):
        """Initialize neural networks once state size and num_skus are known"""
        self.state_size = state_size
        self.action_size = num_skus * self.num_order_levels
        # Create Q-networks
        self.q_network = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Create optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
    
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
        """Convert continuous state values to discrete for neural network input, including demand stats if env is provided."""
        max_inventory = 1000
        state_arr = np.array(state)
        if env is not None:
            demand_stats = self.get_demand_stats(env)
            state_arr = np.concatenate([state_arr, demand_stats])
        normalized_state = state_arr / max_inventory
        return normalized_state.astype(np.float32)
    
    def get_action(self, state, env, greedy=False):
        if env is None:
            raise ValueError('env must not be None when calling get_action')
        num_skus = len(env.skus)
        if self.order_level_values is None:
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        if self.q_network is None or self.target_network is None:
            state_size = len(self.discretize_state(state, env))
            self.initialize_networks(state_size, num_skus)
        if not greedy and np.random.random() < self.epsilon:
            random_order_levels = np.random.randint(0, self.num_order_levels, num_skus)
            order_quantities = self.order_level_values[random_order_levels]
            return order_quantities
        discrete_state = self.discretize_state(state, env)
        state_tensor = torch.FloatTensor(discrete_state).unsqueeze(0).to(self.device)
        if self.q_network is None:
            raise ValueError('Q-network is not initialized.')
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values = q_values.cpu().numpy()[0]
        best_orders = np.zeros(num_skus)
        for i in range(num_skus):
            order_q_start_idx = i * self.num_order_levels
            order_q_end_idx = order_q_start_idx + self.num_order_levels
            sku_order_q_values = q_values[order_q_start_idx:order_q_end_idx]
            best_order_level_idx = np.argmax(sku_order_q_values)
            best_orders[i] = self.order_level_values[best_order_level_idx]
        return best_orders
    
    def learn(self, state, action, reward, next_state, env=None, episode_stockouts=0):
        if env is None:
            raise ValueError('env must not be None when calling learn')
        discrete_state = self.discretize_state(state, env)
        discrete_next_state = self.discretize_state(next_state, env)
        num_skus = len(action)
        if self.order_level_values is None:
            self.order_level_values = np.linspace(0, env.action_space.high[0], self.num_order_levels, dtype=np.int32)
        if self.q_network is None or self.target_network is None:
            state_size = len(discrete_state)
            self.initialize_networks(state_size, num_skus)
        action_indices = []
        for i in range(num_skus):
            order_level_idx = np.argmin(np.abs(self.order_level_values - action[i]))
            action_indices.append(order_level_idx)
        self.memory.push(discrete_state, action_indices, reward, discrete_next_state, False, episode_stockouts=episode_stockouts)
        if len(self.memory) < self.batch_size:
            return 0.0
        samples = self.memory.sample(self.batch_size)
        if not samples:
            return 0.0
        batch = Experience(*zip(*samples))
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        reward_batch = torch.clamp(reward_batch, -1000, 1000)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        if self.q_network is None or self.target_network is None:
            raise ValueError('Q-network and target network must be initialized.')
        self.q_network.train()
        self.target_network.eval()
        current_q_values = self.q_network(state_batch)
        with torch.no_grad():
            next_q_values = self.q_network(next_state_batch)
            next_q_values_target = self.target_network(next_state_batch)
        n_action_components = action_batch.shape[1]
        loss = torch.tensor(0.0, device=self.device)
        for i in range(n_action_components):
            current_q = current_q_values[:, i]
            action_idx = action_batch[:, i]
            current_q_selected = current_q.gather(0, action_idx.unsqueeze(0)).squeeze(0) if current_q.dim() > 1 else current_q[action_idx]
            next_actions = next_q_values[:, i].argmax(dim=0)
            next_q_selected = next_q_values_target[:, i][next_actions]
            target_q = reward_batch + (self.discount_factor * next_q_selected * (~done_batch).float())
            component_loss = F.smooth_l1_loss(current_q_selected, target_q)
            loss = loss + component_loss
        loss = loss / n_action_components
        
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized.")
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.soft_update()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss.item()
    
    def soft_update(self):
        """Soft update target network using Polyak averaging"""
        if self.q_network is not None and self.target_network is not None:
            for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
    
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