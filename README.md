# Double Q-Learning Based Inventory Management System

This project implements a sophisticated Double Q-learning based inventory management system with multi-echelon supply chain optimization. The system uses temporal difference learning with eligibility traces to manage inventory levels across multiple locations while considering seasonal demand patterns.

## Features

- **Multi-Echelon Supply Chain**
  - Multiple suppliers (X, Y)
  - Multiple inventory locations
  - Retail store management
  - SKU tracking (Products A, B, C)

- **Seasonal Demand Modeling**
  - Cosine-based seasonal patterns: Demand = Acos(wt+phi) + noise
  - Product-specific cycles:
    * Type A: Annual cycle (365 days)
    * Type B: Semi-annual cycle (182.5 days)
    * Type C: Quarterly cycle (91.25 days)
  - Gaussian noise with σ = 10

- **Advanced Learning Strategy**
  - Dual-phase learning:
    * Phase 1 (Episodes 1-500): SARSA with EOQ-based exploration
    * Phase 2 (Episodes 501+): Double Q-learning with Boltzmann exploration
  - TD(λ) with eligibility traces
  - Enhanced numerical stability
  - Extended training (15,000 episodes)

- **Performance Metrics**
  - Stockout tracking
  - Lead time optimization
  - Service level monitoring
  - Cost analysis
  - Seasonal adaptation performance

## Requirements

```python
numpy
matplotlib
seaborn
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/asm24legend/IIMintern.git
cd IIMintern
```

2. Install dependencies:
```bash
pip install numpy matplotlib seaborn
```

## Usage

1. Run the Double Q-learning agent:
```bash
python q_learning_agent.py
```

2. Analyze performance metrics:
```bash
python analyze_performance.py
```

The analysis script will generate visualizations showing:
- Average stockouts over 6 months
- Number of on-time orders
- Inventory levels (warehouse and retail)
- Total costs breakdown

## Performance Analysis

The system generates comprehensive performance metrics including:
- Daily stockout rates
- Inventory holding costs
- Order fulfillment rates
- Total operational costs

Results are saved in the `results_[timestamp]/analysis` directory with detailed visualizations.

## License

MIT License

## Author

asm24legend 