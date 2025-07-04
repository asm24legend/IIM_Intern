import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta

@dataclass
class SKUData:
    #Define the datatypes for the sku data
    sku: str
    description: str
    current_stock: int
    reorder_point: int
    safety_stock: int
    lead_time_days: int
    eoq: int
    max_stock: int
    min_order_qty: int
    inventory_location: str
    supplier: str
    open_pos: int
    last_order_date: datetime  
    next_delivery_date: datetime  
    previous_demand: float = 0
    retail_stock: int = 0  # Stock at retail store
    stockout_cost_multiplier: float = 1.0 # New field for SKU-specific stockout cost
    # Add ABC classification and value
    abc_class: str = 'C'  # A, B, or C classification
    unit_value: float = 0.0  # Per-unit value for revenue/cost analysis
    # Add seasonal demand parameters
    base_demand: float = 0  # Base demand level
    
    last_decision_time: float = 0  # New field to track when last decision was made
    # Service level tracking
    total_demand: float = 0.0  # Total historical demand
    fulfilled_demand: float = 0.0  # Total fulfilled demand
    stockout_occasions: int = 0  # Number of stockout events
    replenishment_cycles: int = 0  # Number of replenishment cycles
    alpha: float = 2.0  # Shape parameter for gamma demand
    beta: float = 2.0   # Scale parameter for gamma demand
    retail_replenishment_lead_time: int = 1  # Lead time (days) for warehouse to retail
    retail_replenishment_queue: list = field(default_factory=list)  # List of (arrival_time, amount) tuples
    shelf_life_days: int = 30  # Default shelf life in days
    demand_history: list = field(default_factory=list)  # Rolling window of recent daily demand
    demand_window: int = 30  # Number of days for rolling window

@dataclass
class InventoryLocation:
    name: str
    sku_types: List[str]  # Types of SKUs stored here
    capacity: int
    current_stock: Dict[str, int]  # SKU -> quantity mapping

class InventoryEnvironment(gym.Env):
    def __init__(self, config=None):
        super(InventoryEnvironment, self).__init__()
        
        self.config = {
            'max_inventory': 1000,
            'max_demand': 100,
            'holding_cost': 2,
            'stockout_cost': 50,
            'order_cost': 100,
            'transportation_cost_per_unit': 5,  # Added transportation cost per unit
            'min_lead_time': 1,
            'max_lead_time': 14,
            'retail_replenishment_time': 1,  # Days to replenish retail from warehouse
            'noise_std': 10,
            'min_decision_interval': 0.1,  # Minimum time between decisions (in days)
            'current_time': 0.0,  # Current simulation time in days (float)
            'lead_time_reduction_cost': 50,  # Cost per day of lead time reduction
            'max_lead_time_reduction': 3,  # Maximum days by which lead time can be reduced
            'use_fixed_demand': False,      # Toggle for benchmarking
            'fixed_daily_demand': 5         # Value for fixed demand
        } if config is None else config

        # Initialize inventory locations
        self.inventory_locations = {
            'Location_1': InventoryLocation(
                name='Location_1',
                sku_types=['Type_A'],
                capacity=1000,
                current_stock={'Type_A': 450}
            ),
            'Location_2': InventoryLocation(
                name='Location_2',
                sku_types=['Type_B'],
                capacity=800,
                current_stock={'Type_B': 350}
            ),
            'Location_3': InventoryLocation(
                name='Location_3',
                sku_types=['Type_C'],
                capacity=600,
                current_stock={'Type_C': 250}
            )
        }

        # Initialize retail store
        self.retail_store = {
            'Type_A': 100,  # Initial retail stock
            'Type_B': 80,
            'Type_C': 60
        }

        # Set quantile for forecasted demand
        demand_quantile = 0.9  # 90th percentile
        # Import gamma only when needed for quantile calculation
        from scipy.stats import gamma
        # Initialize SKUs with seasonal demand parameters
        base_date = datetime.now()
        # For each SKU, set forecasted_demand to the 90th percentile of the gamma distribution
        alpha_A, beta_A = 15.0, 2.5   # Increased alpha, decreased beta
        alpha_B, beta_B = 30.0, 5.0   # Increased alpha, decreased beta
        alpha_C, beta_C = 60.0, 12.5  # Increased alpha, decreased beta
        self.skus = {
            'Type_A': SKUData(
                sku='Type_A',
                description='Product Type A',
                current_stock=120,
                reorder_point=40,
                safety_stock=20,
                lead_time_days=3,
                alpha=alpha_A,
                beta=beta_A,
                eoq=20,
                max_stock=170,
                min_order_qty=10,
                inventory_location='Location_1',
                supplier='Supplier_X',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=3),
                retail_stock=40,
                base_demand=50,
                stockout_cost_multiplier=2.0,
                abc_class='A',
                unit_value=1000.0,
                last_decision_time=0.0
            ),
            'Type_B': SKUData(
                sku='Type_B',
                description='Product Type B',
                current_stock=350,
                reorder_point=120,
                safety_stock=60,
                lead_time_days=7,
                alpha=alpha_B,
                beta=beta_B,
                eoq=60,
                max_stock=450,
                min_order_qty=40,
                inventory_location='Location_2',
                supplier='Supplier_Y',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=7),
                retail_stock=75,
                base_demand=200,
                stockout_cost_multiplier=1.0,
                abc_class='B',
                unit_value=300.0,
                last_decision_time=0.0
            ),
            'Type_C': SKUData(
                sku='Type_C',
                description='Product Type C',
                current_stock=900,
                reorder_point=350,
                safety_stock=50,
                lead_time_days=1,
                alpha=alpha_C,
                beta=beta_C,
                eoq=600,
                max_stock=3000,
                min_order_qty=100,
                inventory_location='Location_3',
                supplier='Supplier_Z',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=1),
                retail_stock=180,
                base_demand=1000,
                stockout_cost_multiplier=0.5,
                abc_class='C',
                unit_value=50.0,
                shelf_life_days=6,
                last_decision_time=0.0
            )
        }
        # NOTE: forecasted_demand is now set to the 90th percentile (quantile=0.9) of the gamma distribution for each SKU.
        
        # Initialize suppliers with their capabilities and products
        self.suppliers = {
            'Supplier_X': {
                'lead_time_range': (5, 10),
                'current_load': 0,
                'reliability': 0.95,
                'products': ['Type_A']
            },
            'Supplier_Y': {
                'lead_time_range': (7, 12),
                'current_load': 0,
                'reliability': 0.90,
                'products': ['Type_B']  # Only Type_B now
            },
            'Supplier_Z': {
                'lead_time_range': (6, 11),
                'current_load': 0,
                'reliability': 0.92,
                'products': ['Type_C']  # New supplier for Type_C
            }
        }
        
        num_skus = len(self.skus)
        
        # New action space: order for warehouse for each SKU (no lead time reductions)
        self.action_space = spaces.Box(
            low=np.zeros(num_skus, dtype=np.int32),
            high=np.array([self.config['max_inventory']] * num_skus, dtype=np.int32),
            dtype=np.int32
        )
        
        # Initialize observation space
        self.observation_space = spaces.Dict({
            'warehouse_stock': spaces.Box(low=0, high=self.config['max_inventory'], shape=(num_skus,), dtype=np.int32),
            'retail_stock': spaces.Box(low=0, high=self.config['max_inventory'], shape=(num_skus,), dtype=np.int32),
            'open_pos': spaces.Box(low=0, high=self.config['max_inventory'], shape=(num_skus,), dtype=np.int32),
            'demand': spaces.Box(low=0, high=np.inf, shape=(num_skus,), dtype=np.float32),
            'lead_time': spaces.Box(low=0, high=np.inf, shape=(num_skus,), dtype=np.float32),
            'supplier_load': spaces.Box(low=0, high=1, shape=(num_skus,), dtype=np.float32),
            'supplier_reliability': spaces.Box(low=0, high=1, shape=(num_skus,), dtype=np.float32),
            'time_delta': spaces.Box(low=0, high=np.inf, shape=(num_skus,), dtype=np.float32)
        })
        
        self.reset()

    #def generate_seasonal_demand(self, sku: SKUData, time_period: float) -> float:
       # """Generate seasonal demand using cosine function with noise for a specific time period"""
       # seasonal_component = sku.amplitude * np.cos(
            #sku.frequency * time_period + sku.phase
       # )
       # noise = np.random.normal(0, self.config['noise_std'])
        #demand = max(0, sku.base_demand + seasonal_component + noise)
       # return demand

    def calculate_demand_for_period(self, sku: SKUData, start_time: float, end_time: float) -> float:
        """Calculate total demand over a time period using fixed or gamma demand (for benchmarking)."""
        days = max(1, end_time - start_time)
        if self.config.get('use_fixed_demand', False):
            demand = self.config.get('fixed_daily_demand', 5) * days
        else:
            demand = np.random.gamma(shape=sku.alpha * days, scale=sku.beta)
        return max(0, demand)

    def calculate_lead_time_demand(self, sku_id: str) -> float:
        """Calculate demand during lead time"""
        sku = self.skus[sku_id]
        total_demand = 0
        current_time = self.config['current_time']
        
        
        # Reset time step
        self.config['current_time'] = current_time
        return total_demand * 1.20  # Add 20% safety factor

    def calculate_eoq(self, sku_id: str) -> int:
        """Calculate Economic Order Quantity for a specific SKU using dynamic gamma params."""
        sku = self.skus[sku_id]
        alpha, beta = self.estimate_gamma_params(sku)
        D = alpha * beta  # Use mean of dynamic gamma
        K = self.config['order_cost']  # Order cost
        H = self.config['holding_cost']  # Holding cost
        return int(np.sqrt((2 * D * K) / H))

    def calculate_rop(self, sku_id: str) -> int:
        """Calculate Reorder Point as a high percentile of the gamma distribution for demand during lead time, using dynamic gamma params."""
        from scipy.stats import gamma
        sku = self.skus[sku_id]
        alpha, beta = self.estimate_gamma_params(sku)
        quantile = 0.999 if sku_id == 'Type_C' else (0.98 if sku_id == 'Type_C' else 0.97)
        shape = alpha * sku.lead_time_days
        scale = beta
        rop = gamma.ppf(quantile, a=shape, scale=scale)
        # Add safety stock (already calculated for high service level)
        return int(np.ceil(rop + sku.safety_stock))

    def calculate_reward(self, sku_id: str, stockout: int, current_stock: int, 
                           daily_demand: float, current_lead_time: int, previous_lead_time: int) -> float:
        """
        Calculate reward with location-specific rewards and penalties:
        1. Location-specific stockout penalties
        2. Location-specific demand fulfillment rewards
        3. Inventory level penalties to maintain efficiency
        4. Lead time reduction rewards
        """
        reward = 0.0
        sku = self.skus[sku_id]
        location = self.inventory_locations[sku.inventory_location]
        # Location-specific stockout penalties (further reduced)
        if stockout > 0:
            if sku.inventory_location == 'Location_1':
                reward -= 25 * stockout  # Was 50
            elif sku.inventory_location == 'Location_2':
                reward -= 20 * stockout   # Was 40
            elif sku.inventory_location == 'Location_3':
                reward -= 15 * stockout   # Was 30
            # Retail penalty further reduced
            if sku.retail_stock <= 0:
                reward -= 375 * stockout  # Was 750
        else:
            # Higher rewards for fulfillment (further doubled)
            if sku.inventory_location == 'Location_1':
                reward += 400  # Was 200
            elif sku.inventory_location == 'Location_2':
                reward += 320   # Was 160
            elif sku.inventory_location == 'Location_3':
                reward += 240   # Was 120
            if sku.retail_stock > 0:
                reward += 400  # Was 200
        # Inventory level management (penalties further reduced, rewards further increased)
        eoq = self.calculate_eoq(sku_id)
        # Penalty for excess inventory (halved again)
        if current_stock > eoq:
            excess = current_stock - eoq
            reward -= excess * 0.0125  # Was 0.025
        # Penalty for being below safety stock (halved again)
        elif current_stock < sku.safety_stock:
            deficit = sku.safety_stock - current_stock
            reward -= deficit * 0.5  # Was 1.0
        # Higher reward for optimal inventory level (further doubled)
        if sku.safety_stock <= current_stock <= eoq:
            reward += 400  # Was 200
        # Additional reward for maintaining good service level (further doubled)
        if sku.total_demand > 0:
            service_level = sku.fulfilled_demand / sku.total_demand
            if service_level >= 0.95:
                reward += 800  # Was 400
            elif service_level >= 0.90:
                reward += 400  # Was 200
        # Add a minimum reward floor to prevent extreme negatives
        reward = max(reward, -50)
        return reward

    def _get_state(self):
        """Return the current state (inventory levels for all SKUs, warehouse and retail)"""
        state = []
        for sku_id in sorted(self.skus.keys()): # Ensure consistent order
            sku = self.skus[sku_id]
            state.append(sku.current_stock) # Warehouse stock
            state.append(sku.retail_stock)   # Retail stock
        return np.array(state, dtype=np.int32)

    def _replenish_retail(self, sku_id: str, demand: int):
        """Replenish retail stock from the primary warehouse location only."""
        sku = self.skus[sku_id]
        total_replenished = 0
        
        # Only replenish from the primary location
        primary_location = self.inventory_locations[sku.inventory_location]
        if sku.current_stock > 0:
            replenishment = min(demand, sku.current_stock)
            sku.retail_stock += replenishment
            sku.current_stock -= replenishment
            primary_location.current_stock[sku_id] = sku.current_stock
            total_replenished += replenishment
            demand -= replenishment
        
        # No fallback to other locations
        return total_replenished

    def step(self, action):
        rewards = np.zeros(len(self.skus))
        info = {}
        supplier_loads = defaultdict(float)
        stockouts = {}
        service_levels = {}
        num_skus = len(self.skus)
        order_qty_warehouse = action[:num_skus]
        current_time = self.config['current_time']
        time_deltas = {
            sku_id: max(self.config['min_decision_interval'], current_time - sku.last_decision_time)
            for sku_id, sku in self.skus.items()
        }
        # --- Process retail replenishment arrivals for each SKU independently ---
        for sku_id, sku in self.skus.items():
            if sku.retail_replenishment_queue is not None:
                arrivals = [item for item in sku.retail_replenishment_queue if item[0] <= current_time]
                sku.retail_replenishment_queue = [item for item in sku.retail_replenishment_queue if item[0] > current_time]
                for arrival_time, amount in arrivals:
                    # Only replenish from Location (warehouse) to Retail for this SKU
                    actual_amount = min(amount, sku.current_stock)
                    sku.retail_stock += actual_amount
                    sku.current_stock -= actual_amount
                    self.inventory_locations[sku.inventory_location].current_stock[sku_id] = sku.current_stock
        # --- Handle each SKU independently: Supplier -> Location -> Retail ---
        for i, (sku_id, sku) in enumerate(self.skus.items()):
            # Shelf life check: if shelf_life_days < lead_time_days, treat as stockout
            shelf_life_stockout = 0
            if sku.shelf_life_days < sku.lead_time_days:
                shelf_life_stockout = 1
            # --- Supplier to Location (warehouse) ---
            if order_qty_warehouse[i] > 0:
                current_rop = self.calculate_rop(sku_id)
                if sku.current_stock < current_rop and sku.current_stock < sku.max_stock:
                    base_order = max(order_qty_warehouse[i], sku.min_order_qty)
                    available_capacity = sku.max_stock - (sku.current_stock + sku.open_pos)
                    order_qty = min(base_order, available_capacity)
                    if order_qty > 0:
                        sku.open_pos += order_qty
                        self.inventory_locations[sku.inventory_location].current_stock[sku_id] += 0  # No immediate stock
            # --- Demand realization and fulfillment at Retail for this SKU ---
            period_demand = self.calculate_demand_for_period(
                sku,
                sku.last_decision_time,
                current_time
            )
            # --- Update demand history for dynamic gamma forecasting ---
            sku.demand_history.append(period_demand)
            if len(sku.demand_history) > sku.demand_window:
                sku.demand_history = sku.demand_history[-sku.demand_window:]
            sku.total_demand += period_demand
            fulfilled_current_demand = min(period_demand, sku.retail_stock)
            sku.retail_stock -= int(fulfilled_current_demand)
            stockout = int(period_demand - fulfilled_current_demand)
            stockout += shelf_life_stockout
            if stockout > 0:
                sku.stockout_occasions += 1
            sku.fulfilled_demand += fulfilled_current_demand
            stockouts[sku_id] = stockout
            backorder_penalty = 0
            # --- Service level calculation for this SKU ---
            if sku.total_demand > 0:
                service_levels[sku_id] = sku.fulfilled_demand / sku.total_demand
            else:
                service_levels[sku_id] = 1.0
            # --- Supplier delivery to Location (warehouse) for this SKU ---
            if sku.open_pos > 0 and self._is_delivery_due(sku, current_time):
                sku.replenishment_cycles += 1
                sku.current_stock += sku.open_pos
                self.inventory_locations[sku.inventory_location].current_stock[sku_id] += sku.open_pos
                supplier_loads[sku.supplier] += sku.open_pos
                sku.open_pos = 0
            # --- Location to Retail replenishment trigger for this SKU ---
            if sku_id == 'Type_C':
                retail_threshold = max(1, int(1.0 * sku.base_demand))
                target_retail = max(1, int(5.0 * sku.base_demand))
            else:
                retail_threshold = max(1, int(1.2 * sku.base_demand))
                target_retail = max(1, int(2.2 * sku.base_demand))
            if sku.retail_stock < retail_threshold:
                replenish_amount = min(target_retail - sku.retail_stock, sku.current_stock)
                if replenish_amount > 0:
                    arrival_time = current_time + sku.retail_replenishment_lead_time
                    sku.retail_replenishment_queue.append((arrival_time, replenish_amount))
            sku.previous_demand = period_demand
            sku.last_decision_time = current_time
            rewards[i] = self.calculate_reward(
                sku_id,
                int(stockout),
                sku.current_stock,
                period_demand / time_deltas[sku_id],
                self._get_current_lead_time(sku_id),
                sku.lead_time_days
            ) - backorder_penalty
        # --- All other logic remains per-SKU, no cross-SKU dependencies ---
        # Update supplier metrics (still per supplier, but only based on their own SKUs)
        for supplier_id, load in supplier_loads.items():
            self.suppliers[supplier_id]['current_load'] = min(1.0, load / 1000)
            self.suppliers[supplier_id]['reliability'] = max(0.7, 0.95 - 0.1 * self.suppliers[supplier_id]['current_load'])
        # Update info dictionary (all per-SKU)
        info = {
            'warehouse_stock': {sku_id: sku.current_stock for sku_id, sku in self.skus.items()},
            'retail_stock': {sku_id: sku.retail_stock for sku_id, sku in self.skus.items()},
            'open_pos': {sku_id: sku.open_pos for sku_id, sku in self.skus.items()},
            'stockouts': stockouts,
            'service_levels': service_levels,
            'supplier_loads': {supplier: info['current_load'] for supplier, info in self.suppliers.items()},
            'supplier_reliability': {supplier: info['reliability'] for supplier, info in self.suppliers.items()},
            'time_deltas': time_deltas,
            'current_time': current_time,
            'lead_times': {sku_id: sku.lead_time_days for sku_id, sku in self.skus.items()},
        }
        self.config['current_time'] += self.config['min_decision_interval']
        done = False
        if any(sku.retail_stock <= 0 for sku in self.skus.values()):
            done = True
            rewards -= 20
        # After reward calculation, adjust service level penalty (halved again)
        avg_service_level = np.mean(list(service_levels.values()))
        if avg_service_level < 0.95:
            penalty = -250 * (0.98 - avg_service_level)  # Was -500
            rewards += penalty
        # Ensure rewards are not NaN or inf, and clip total reward
        total_reward = np.sum(rewards)
        if not np.isfinite(total_reward):
            total_reward = 0.0
        total_reward = np.clip(total_reward, -1000, 2000)
        return self._get_state(), total_reward, done, info
    
    def calculate_dynamic_safety_stock(self, sku, z=None):
        """Calculate safety stock accurately based on demand variability and lead time, using dynamic gamma params."""
        if z is None:
            z = 4.0 if sku.sku == 'Type_C' else 2.5
        # Use dynamic gamma parameters
        alpha, beta = self.estimate_gamma_params(sku)
        L = sku.lead_time_days
        demand_variance_per_day = alpha * (beta ** 2)
        std_lead_time_demand = np.sqrt(L * demand_variance_per_day)
        return int(np.ceil(z * std_lead_time_demand))

    def reset(self):
        # Reset simulation time
        self.config['current_time'] = 0.0
        # Reset SKUs to initial state
        base_date = datetime.now()
        for sku_id, sku in self.skus.items():
            # For Type_C, set lead_time_days to 1 and shelf_life_days to lead_time_days + 5
            if sku_id == 'Type_C':
                self.skus[sku_id].lead_time_days = 1
                self.skus[sku_id].shelf_life_days = self.skus[sku_id].lead_time_days + 5
            # Dynamically recalculate safety stock
            self.skus[sku_id].safety_stock = self.calculate_dynamic_safety_stock(sku)
            self.skus[sku_id].current_stock = sku.reorder_point
            self.skus[sku_id].open_pos = 0
            # For Type_C, set retail_stock to 5x safety stock; others unchanged
            if sku_id == 'Type_C':
                self.skus[sku_id].retail_stock = int(self.skus[sku_id].safety_stock * 5.0)
            else:
                self.skus[sku_id].retail_stock = int(self.skus[sku_id].safety_stock * 2.2)
            self.skus[sku_id].last_decision_time = 0.0
            self.skus[sku_id].last_order_date = base_date
            self.skus[sku_id].next_delivery_date = base_date + timedelta(days=self.skus[sku_id].lead_time_days)
            # Reset service level metrics
            self.skus[sku_id].total_demand = 0.0
            self.skus[sku_id].fulfilled_demand = 0.0
            self.skus[sku_id].stockout_occasions = 0
            self.skus[sku_id].replenishment_cycles = 0
            self.skus[sku_id].previous_demand = 0.0
            self.skus[sku_id].retail_replenishment_queue = []
        # Reset supplier loads
        for supplier in self.suppliers.values():
            supplier['current_load'] = 0
        return self._get_state()
    
    def _is_delivery_due(self, sku: SKUData, current_time: float) -> bool:
        """Check if delivery is due for a SKU based on simulation time"""
        if not sku.next_delivery_date:
            return False
        
        delivery_time = (sku.next_delivery_date - sku.last_order_date).total_seconds() / 86400  # Convert to days
        return current_time >= delivery_time
    
    def _update_delivery_date(self, sku_id: str):
        """Update next delivery date based on constant lead time (from SKUData)"""
        sku = self.skus[sku_id]
        lead_time = sku.lead_time_days  # Use constant lead time
        self.skus[sku_id].last_order_date = datetime.now()
        self.skus[sku_id].next_delivery_date = self.skus[sku_id].last_order_date + timedelta(days=lead_time)

    def _get_current_lead_time(self, sku_id: str) -> int:
        """Get current lead time for a SKU (constant)"""
        return self.skus[sku_id].lead_time_days
    
    def _get_days_to_delivery(self, sku_id: str) -> int:
        """Calculate days until next delivery (constant lead time)"""
        sku = self.skus[sku_id]
        if sku.next_delivery_date and sku.last_order_date:
            delta = sku.next_delivery_date - datetime.now()
            return max(0, delta.days)
        return sku.lead_time_days

    def estimate_gamma_params(self, sku):
        """Estimate gamma distribution parameters from recent demand history."""
        history = sku.demand_history[-sku.demand_window:]
        if len(history) < 2:
            # Not enough data, fall back to static params
            return sku.alpha, sku.beta
        mean = np.mean(history)
        var = np.var(history)
        if var == 0 or mean == 0:
            # Avoid division by zero
            return 1000.0, mean / 1000.0 if mean > 0 else 1.0
        alpha = mean ** 2 / var
        beta = var / mean
        return alpha, beta
