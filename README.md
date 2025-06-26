# Multi-Agent Inventory Management System

This project implements a sophisticated inventory management system using multiple reinforcement learning agents for multi-echelon supply chain optimization. The system compares Double Q-learning, Double DQN, and Random agents to manage inventory levels across multiple locations while considering seasonal demand patterns.

## Features

- **Multi-Echelon Supply Chain**
  - Multiple suppliers (X, Y)
  - Multiple inventory locations
  - Retail store management
  - SKU tracking (Products A, B, C)

- **Three Learning Agents**
  - **Double Q-learning Agent**: Traditional tabular Q-learning with dual Q-tables
  - **Double DQN Agent**: Deep Q-Network with neural networks and experience replay
  - **Random Agent**: Baseline EOQ-based policy for comparison

- **Seasonal Demand Modeling**
  - Cosine-based seasonal patterns: Demand = Acos(wt+phi) + noise
  - Product-specific cycles:
    * Type A: Annual cycle (365 days)
    * Type B: Semi-annual cycle (182.5 days)
    * Type C: Quarterly cycle (91.25 days)
  - Gaussian noise with σ = 10

- **Advanced Learning Strategies**
  - **Double Q-learning**: Dual-phase learning with TD(λ) and eligibility traces
  - **Double DQN**: Neural network-based learning with experience replay buffer
  - Enhanced numerical stability
  - Extended training (10,000 episodes per agent)

- **Performance Metrics**
  - Stockout tracking by location
  - Lead time optimization
  - Service level monitoring
  - Cost analysis
  - Seasonal adaptation performance
  - Comparative analysis between agents

## Requirements

```python
numpy>=1.24.0
pandas>=1.3.0
matplotlib>=3.7.0
gym>=0.21.0
seaborn>=0.12.0
torch>=1.13.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/asm24legend/IIMintern.git
cd IIMintern
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main training and comparison script:
```bash
python main.py
```

This will:
- Train the Double Q-learning agent for 10,000 episodes
- Train the Double DQN agent for 10,000 episodes
- Evaluate all three agents (including Random baseline)
- Generate comprehensive comparison plots and metrics

2. Test the Double DQN agent individually:
```bash
python test_dqn.py
```

3. Analyze performance metrics:
```bash
python analyze_performance.py
```

## Agent Comparison

The system provides detailed comparison between:

### Double Q-learning Agent
- Uses tabular Q-learning with dual Q-tables
- Discretized state and action spaces
- Traditional reinforcement learning approach
- Good for smaller state spaces

### Double DQN Agent
- Uses neural networks for function approximation
- Experience replay buffer for stable learning
- Target network for reduced overestimation bias
- Better for continuous or large state spaces
- Requires PyTorch

### Random Agent (Baseline)
- EOQ-based policy
- No learning component
- Serves as performance baseline

## Performance Analysis

The system generates comprehensive performance metrics including:
- Daily stockout rates by location
- Inventory holding costs
- Order fulfillment rates
- Total operational costs
- Service levels
- Transportation costs
- Comparative reward distributions

Results are saved in the `results_[timestamp]/` directory with detailed visualizations:
- Training progress plots for each agent
- Reward distribution histograms
- Metrics comparison bar charts
- Location-specific stockout analysis
- Cumulative reward plots

## File Structure

```
IIM_Intern/
├── main.py                 # Main training and comparison script
├── inventory_env.py        # Inventory environment implementation
├── q_learning_agent.py     # Double Q-learning agent
├── dqn_agent.py           # Double DQN agent implementation
├── test_dqn.py            # Test script for DQN agent
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── results_[timestamp]/   # Training results and visualizations
```

## License

MIT License

## Author

asm24legend 