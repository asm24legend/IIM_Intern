import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta

#Importing all the required libraries
@dataclass
class SKUData:
    #Define the datatypes for the sku data
    sku: str  #The item code
    description: str #The item category
    current_stock: int  #Inventory level present
    reorder_point: int  #Point at which inventory is reordered
    safety_stock: int  #Amount of minimum inventory that should be there
    lead_time_days: int #Number of days required for order fulfillment  
    max_stock: int #Maximum amount of stock that can be there in an inventory
    min_order_qty: int #Minimum quantity that can be ordered at a given time
    inventory_location: str #A, B, C are warehouses
    supplier: str #Supplier X, Y and Z initialized
    open_pos: int #Number of open purchase orders, stock ordered but not yet arrived, in transit stuff
    last_order_date: datetime  
    next_delivery_date: datetime  
    previous_demand: float = 0 #Demand in the previous time range
    retail_stock: int = 0  # Stock at retail store
    # Add ABC classification and value
    abc_class: str = 'C'  # A, B, or C classification
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
    retail_lead_time_days: int = 1  # Lead time (days) for SKU to reach retail from its warehouse location
    retail_replenishment_queue: list = field(default_factory=list)  # List of (arrival_time, amount) tuples
    shelf_life_days: int = 30  # Default shelf life in days
    demand_history: list = field(default_factory=list)  # Rolling window of recent daily demand
    demand_window: int = 30  # Number of days for rolling window
    open_pos_supplier_to_warehouse: int = 0  # Open POs from supplier to warehouse
    open_pos_warehouse_to_retail: int = 0    # Open POs from warehouse to retail
    retail_reorder_point: int = 0            # ROP for retail
    
    # Enhanced features for increased state space utilization
    demand_volatility: float = 0.0           # Volatility measure of recent demand
    seasonal_factor: float = 1.0             # Current seasonal multiplier
    trend_factor: float = 1.0                # Current trend multiplier
    forecast_accuracy: float = 1.0           # Rolling accuracy of demand forecasts
    days_since_stockout: int = 0             # Days since last stockout
    consecutive_stockouts: int = 0           # Number of consecutive stockout periods
    demand_forecast: float = 0.0             # Next period demand forecast

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
            
            'retail_replenishment_time': 1,  # Days to replenish retail from warehouse
            'noise_std': 5,  # Reduced from 10 to 5 for more stable environment
            'min_decision_interval': 1,  # Minimum time between decisions (in days)
            'current_time': 0.0,  # Current simulation time in days (float)
            
            'use_fixed_demand': False,      # Toggle for benchmarking
            'fixed_daily_demand': 5         # Value for fixed demand
        } if config is None else config

        # Initialize inventory locations
        self.inventory_locations = {
            'Location_1': InventoryLocation(
                name='Location_1',
                sku_types=['Type_A'], #High value, low demand items
                capacity=1000, 
                current_stock={'Type_A': 450} #Initial stock
            ),
            'Location_2': InventoryLocation(
                name='Location_2',
                sku_types=['Type_B'], #Medium value. Medium demand
                capacity=800,
                current_stock={'Type_B': 350} 
            ),
            'Location_3': InventoryLocation(
                name='Location_3', #Low value, low demand
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

        base_date = datetime.now()
        self.skus = {
            'Type_A': SKUData(
                sku='Type_A',
                description='Product Type A',
                current_stock=120,
                reorder_point=40,
                safety_stock=20,
                lead_time_days=2,  # Reduced from 3 to 2
                
                max_stock=170,
                min_order_qty=10,
                inventory_location='Location_1',
                supplier='Supplier_X',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=2),
                retail_stock=40,
                base_demand=50,
                abc_class='A',
                last_decision_time=0.0,
                alpha=5.0,  # Increased from 2.0 to 5.0 for more stable demand
                beta=1.0,   # Reduced from 2.0 to 1.0 for more stable demand
                retail_lead_time_days=1,  # Reduced from 2 to 1
                open_pos_supplier_to_warehouse=0,
                open_pos_warehouse_to_retail=0,
                retail_reorder_point=0
            ),
            'Type_B': SKUData(
                sku='Type_B',
                description='Product Type B',
                current_stock=350,
                reorder_point=120,
                safety_stock=60,
                lead_time_days=4,  # Reduced from 7 to 4
                
                max_stock=450,
                min_order_qty=40,
                inventory_location='Location_2',
                supplier='Supplier_Y',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=4),
                retail_stock=75,
                base_demand=200,
                abc_class='B',
                last_decision_time=0.0,
                alpha=8.0,  # Increased from 2.0 to 8.0 for more stable demand
                beta=1.0,   # Reduced from 2.0 to 1.0 for more stable demand
                retail_lead_time_days=2,  # Reduced from 3 to 2
                open_pos_supplier_to_warehouse=0,
                open_pos_warehouse_to_retail=0,
                retail_reorder_point=0
            ),
            'Type_C': SKUData(
                sku='Type_C',
                description='Product Type C',
                current_stock=900,
                reorder_point=350,
                safety_stock=50,
                lead_time_days=1,
               
                max_stock=3000,
                min_order_qty=100,
                inventory_location='Location_3',
                supplier='Supplier_Z',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=1),
                retail_stock=180,
                base_demand=1000,
                abc_class='C',
                last_decision_time=0.0,
                alpha=20.0,  # Increased from 2.0 to 20.0 for more stable demand
                beta=1.0,    # Reduced from 2.0 to 1.0 for more stable demand
                retail_lead_time_days=1,
                open_pos_supplier_to_warehouse=0,
                open_pos_warehouse_to_retail=0,
                retail_reorder_point=0
            )
        }
       
        
        # Initialize suppliers with their capabilities and products
        self.suppliers = {
            'Supplier_X': {
                'lead_time_range': (1, 4),  # Reduced from (2, 10) to (1, 4)
                'current_load': 0,
                'reliability': 0.95,
                'products': ['Type_A']
            },
            'Supplier_Y': {
                'lead_time_range': (2, 6),  # Reduced from (5, 10) to (2, 6)
                'current_load': 0,
                'reliability': 0.90,
                'products': ['Type_B']  # Only Type_B now
            },
            'Supplier_Z': {
                'lead_time_range': (1, 3),  # Reduced from (6, 10) to (1, 3)
                'current_load': 0,
                'reliability': 0.92,
                'products': ['Type_C']  # New supplier for Type_C
            }
        }
        
        num_skus = len(self.skus)
        
        # New action space: order for warehouse for each SKU (no lead time reductions) and also does not include the retail order quantities because they are decided by the warehouse level quantity.

        self.action_space = spaces.Box(
            low=np.zeros(num_skus, dtype=np.int32),
            high=np.array([self.config['max_inventory']] * num_skus, dtype=np.int32),
            dtype=np.int32
        )
        
        # Simplified observation space: only current inventory levels across locations
        self.observation_space = spaces.Box(
            low=0, 
            high=self.config['max_inventory'], 
            shape=(num_skus * 4,),  # 4 locations: Location_1, Location_2, Location_3, retail
            dtype=np.int32
        )
        
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

    def update_enhanced_features(self, sku_id: str, actual_demand: float):
        """Update enhanced features for better state space utilization"""
        sku = self.skus[sku_id]
        current_time = self.config['current_time']
        
        # Update demand volatility (rolling standard deviation)
        if len(sku.demand_history) >= 3:
            sku.demand_volatility = np.std(sku.demand_history[-10:]) / max(np.mean(sku.demand_history[-10:]), 1)
        
        # Update seasonal factor (sinusoidal pattern)
        # Different SKUs have different seasonal patterns
        if sku_id == 'Type_A':
            sku.seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * current_time / 365)  # Annual cycle
        elif sku_id == 'Type_B':
            sku.seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * current_time / 90)   # Quarterly cycle
        else:  # Type_C
            sku.seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * current_time / 30)   # Monthly cycle
        
        # Update trend factor (gradual growth/decline)
        trend_rate = 0.001 if sku_id == 'Type_A' else (-0.0005 if sku_id == 'Type_B' else 0.0002)
        sku.trend_factor = 1 + trend_rate * current_time
        
        # Update forecast accuracy (comparing previous forecast to actual)
        if sku.demand_forecast > 0:
            forecast_error = abs(sku.demand_forecast - actual_demand) / max(actual_demand, 1)
            # Exponential moving average of forecast accuracy
            sku.forecast_accuracy = 0.9 * sku.forecast_accuracy + 0.1 * (1 - min(forecast_error, 1))
        
        # Update days since stockout
        if sku.retail_stock <= 0:
            sku.days_since_stockout = 0
            sku.consecutive_stockouts += 1
        else:
            sku.days_since_stockout += 1
            sku.consecutive_stockouts = 0
        
        # Generate demand forecast for next period
        base_forecast = sku.base_demand * sku.seasonal_factor * sku.trend_factor
        if len(sku.demand_history) >= 3:
            # Use exponential smoothing for forecast
            alpha = 0.3
            recent_avg = np.mean(sku.demand_history[-3:])
            sku.demand_forecast = alpha * recent_avg + (1 - alpha) * base_forecast
        else:
            sku.demand_forecast = base_forecast

    def calculate_lead_time_demand(self, sku_id: str) -> float:
        """Calculate demand during lead time"""
        sku = self.skus[sku_id]
        total_demand = 0
        current_time = self.config['current_time']
        
        
        # Reset time step
        self.config['current_time'] = current_time
        return total_demand * 1.20  # Add 20% safety factor

    def calculate_reward(self, sku_id: str, stockout: int, current_stock: int, 
                           daily_demand: float, current_lead_time: int, previous_lead_time: int) -> float:
        """
        Calculate reward with location-specific rewards and penalties:
        1. Location-specific stockout penalties (reduced)
        2. Location-specific demand fulfillment rewards (reduced)
        3. Inventory level penalties to maintain efficiency (reduced)
        4. Lead time reduction rewards (reduced)
        """
        reward = 0.0
        sku = self.skus[sku_id]
        location = self.inventory_locations[sku.inventory_location]
        # Location-specific stockout penalties (further reduced)
        # Scale by SKU criticality: A=2.0, B=1.5, C=1.0
        criticality_multiplier = 2.0 if sku.abc_class == 'A' else (1.5 if sku.abc_class == 'B' else 1.0)
        if stockout > 0:
            if sku.inventory_location == 'Location_1':
                reward -= 5.0 * stockout * criticality_multiplier  # Reduced from 12.5
            elif sku.inventory_location == 'Location_2':
                reward -= 4.0 * stockout * criticality_multiplier  # Reduced from 10
            elif sku.inventory_location == 'Location_3':
                reward -= 3.0 * stockout * criticality_multiplier  # Reduced from 7.5
            # Retail penalty 
            if sku.retail_stock <= 0:
                reward -= 15.0 * stockout * criticality_multiplier  # Reduced from 187.5
        else:
            # Higher rewards for fulfillment (reduced)
            if sku.inventory_location == 'Location_1':
                reward += 10.0  # Reduced from 200
            elif sku.inventory_location == 'Location_2':
                reward += 8.0   # Reduced from 160
            elif sku.inventory_location == 'Location_3':
                reward += 6.0   # Reduced from 120
            if sku.retail_stock > 0:
                reward += 10.0  # Reduced from 200
        # Penalty for excess inventory (reduced)
        if current_stock > sku.max_stock:
            excess = current_stock - sku.max_stock
            reward -= excess * 0.01  # Reduced from 0.00625
        # Penalty for being below safety stock (reduced)
        elif current_stock < sku.safety_stock:
            deficit = sku.safety_stock - current_stock
            reward -= deficit * 0.25  
        # Higher reward for optimal inventory level (reduced)
        if sku.safety_stock <= current_stock <= sku.max_stock:
            reward += 200  
        # Additional reward for maintaining good service level (reduced)
        if sku.total_demand > 0:
            service_level = sku.fulfilled_demand / sku.total_demand
            if service_level >= 0.99:
                reward += 400  
            elif service_level >= 0.90:
                reward += 200  
        # Add a minimum reward floor to prevent extreme negatives as was the case observed in some runs of the simulation.
        reward = max(reward, -25)
        return reward

    def calculate_rop(self, sku: SKUData, location: str = 'warehouse') -> int:
        """Calculate reorder point for warehouse or retail."""
        if location == 'warehouse':
            # ROP = demand during lead time + safety stock
            lead_time = sku.lead_time_days
            avg_daily_demand = sku.base_demand
        elif location == 'retail':
            lead_time = sku.retail_lead_time_days
            avg_daily_demand = sku.base_demand
        else:
            raise ValueError('Unknown location for ROP calculation')
        demand_during_lead_time = avg_daily_demand * lead_time
        if location == 'warehouse':
            return int(np.ceil(demand_during_lead_time + sku.safety_stock))
        else:
            # For retail, use a smaller safety stock if needed
            return int(np.ceil(demand_during_lead_time + max(1, sku.safety_stock // 2)))

    def _get_state(self):
        """Return the current state with all rich features for better decision making"""
        state = []
        current_time = self.config['current_time']
        
        for sku_id in sorted(self.skus.keys()): # Ensure consistent order
            sku = self.skus[sku_id]
            
            # Basic inventory levels
            state.append(sku.current_stock)  # Warehouse stock
            state.append(sku.retail_stock)   # Retail stock
            
            # Open purchase orders
            state.append(sku.open_pos_supplier_to_warehouse)  # Supplier to warehouse POs
            state.append(sku.open_pos_warehouse_to_retail)    # Warehouse to retail POs
            
            # Demand information
            period_demand = self.calculate_demand_for_period(
                sku, sku.last_decision_time, current_time
            ) if current_time > sku.last_decision_time else 0.0
            state.append(period_demand)  # Current period demand
            
            # Lead time information
            state.append(sku.lead_time_days)  # Current lead time
            state.append(self._get_days_to_delivery(sku_id))  # Days until next delivery
            
            # Supplier information
            supplier_info = self.suppliers[sku.supplier]
            state.append(supplier_info['current_load'])  # Supplier load
            state.append(supplier_info['reliability'])   # Supplier reliability
            
            # Time information
            time_delta = max(self.config['min_decision_interval'], 
                           current_time - sku.last_decision_time)
            state.append(time_delta)  # Time since last decision
            
            # Service level metrics
            service_level = (sku.fulfilled_demand / sku.total_demand 
                           if sku.total_demand > 0 else 1.0)
            state.append(service_level)  # Current service level
            
            # Inventory thresholds
            state.append(sku.safety_stock)  # Safety stock level
            state.append(sku.reorder_point)  # Reorder point
            state.append(sku.max_stock)      # Maximum stock level
            
            # ABC classification (encoded as numeric)
            abc_value = 3.0 if sku.abc_class == 'A' else (2.0 if sku.abc_class == 'B' else 1.0)
            state.append(abc_value)  # ABC classification value
            
            # Demand history statistics
            if len(sku.demand_history) > 0:
                avg_demand = np.mean(sku.demand_history)
                demand_std = np.std(sku.demand_history) if len(sku.demand_history) > 1 else 0.0
            else:
                avg_demand = sku.base_demand
                demand_std = 0.0
            state.append(avg_demand)  # Average historical demand
            state.append(demand_std)   # Demand standard deviation
            
            # Stockout and replenishment metrics
            state.append(sku.stockout_occasions)  # Number of stockouts
            state.append(sku.replenishment_cycles)  # Number of replenishment cycles
            
            # Enhanced features for better state space utilization
            state.append(sku.demand_volatility)     # Recent demand volatility
            state.append(sku.seasonal_factor)      # Seasonal adjustment factor
            state.append(sku.trend_factor)         # Trend adjustment factor
            state.append(sku.forecast_accuracy)    # Forecast accuracy metric
            state.append(sku.days_since_stockout)  # Days since last stockout
            state.append(sku.consecutive_stockouts) # Consecutive stockout periods
            state.append(sku.demand_forecast)      # Forecasted demand
            
        return np.array(state, dtype=np.float32)
    
    def _get_observation(self):
        """Return simplified observation with only inventory levels across locations"""
        observation = []
        for sku_id in sorted(self.skus.keys()):
            sku = self.skus[sku_id]
            # Get stock levels for each location
            location_1_stock = sku.current_stock if sku.inventory_location == 'Location_1' else 0
            location_2_stock = sku.current_stock if sku.inventory_location == 'Location_2' else 0
            location_3_stock = sku.current_stock if sku.inventory_location == 'Location_3' else 0
            retail_stock = sku.retail_stock
            
            observation.extend([location_1_stock, location_2_stock, location_3_stock, retail_stock])
        
        return np.array(observation, dtype=np.int32)

    def _replenish_retail(self, sku_id: str, demand: int):
        """Replenish retail stock from the primary warehouse location only, respecting warehouse safety stock."""
        sku = self.skus[sku_id]
        total_replenished = 0
        primary_location = self.inventory_locations[sku.inventory_location]
        # Only allow transfer if warehouse will retain at least safety stock after transfer
        available_to_transfer = max(0, sku.current_stock - sku.safety_stock)
        replenishment = min(demand, available_to_transfer)
        if replenishment > 0:
            sku.retail_stock += replenishment
            sku.current_stock -= replenishment
            primary_location.current_stock[sku_id] = sku.current_stock
            total_replenished += replenishment
            demand -= replenishment
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
                    # Only replenish from Location (warehouse) to Retail for this SKU, respecting safety stock
                    available_to_transfer = max(0, sku.current_stock - sku.safety_stock)
                    actual_amount = min(amount, available_to_transfer)
                    sku.retail_stock += actual_amount
                    sku.current_stock -= actual_amount
                    self.inventory_locations[sku.inventory_location].current_stock[sku_id] = sku.current_stock
                    sku.open_pos_warehouse_to_retail += actual_amount
                    if sku.open_pos_warehouse_to_retail < 0:
                        sku.open_pos_warehouse_to_retail = 0
        
        for i, (sku_id, sku) in enumerate(self.skus.items()):
            # Shelf life check: if shelf_life_days < lead_time_days, treat as stockout
            shelf_life_stockout = 0
            if sku.shelf_life_days < sku.lead_time_days:
                shelf_life_stockout = 1
            # --- Supplier to Location (warehouse) ---
            if order_qty_warehouse[i] > 0:
                
                if sku.current_stock < sku.reorder_point:
                    base_order = max(order_qty_warehouse[i], sku.min_order_qty)
                    available_capacity = sku.max_stock - (sku.current_stock + sku.open_pos_supplier_to_warehouse)
                    order_qty = min(base_order, available_capacity)
                    if order_qty > 0:
                        sku.open_pos_supplier_to_warehouse += order_qty
                        self.inventory_locations[sku.inventory_location].current_stock[sku_id] += 0  # No immediate stock
            # --- Demand realization and fulfillment at Retail for this SKU ---
            period_demand = self.calculate_demand_for_period(
                sku,
                sku.last_decision_time,
                current_time
            )
            # --- Update demand history 
            sku.demand_history.append(period_demand)
            if len(sku.demand_history) > sku.demand_window:
                sku.demand_history = sku.demand_history[-sku.demand_window:]
            sku.total_demand += period_demand
            
            # Update enhanced features for better state utilization
            self.update_enhanced_features(sku_id, period_demand)
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
            if sku.open_pos_supplier_to_warehouse > 0 and self._is_delivery_due(sku, current_time):
                sku.replenishment_cycles += 1
                sku.current_stock += sku.open_pos_supplier_to_warehouse
                self.inventory_locations[sku.inventory_location].current_stock[sku_id] += sku.open_pos_supplier_to_warehouse
                supplier_loads[sku.supplier] += sku.open_pos_supplier_to_warehouse
                sku.open_pos_supplier_to_warehouse = 0
            # --- Location to Retail replenishment trigger for this SKU ---
            if sku_id == 'Type_C':
                retail_threshold = max(1, int(1.0 * sku.base_demand))
                target_retail = max(1, int(5.0 * sku.base_demand))
            else:
                retail_threshold = max(1, int(1.2 * sku.base_demand))
                target_retail = max(1, int(2.2 * sku.base_demand))
            if sku.retail_stock < retail_threshold:
                # Only allow transfer if warehouse will retain at least safety stock after transfer
                available_to_transfer = max(0, sku.current_stock - sku.safety_stock)
                replenish_amount = min(target_retail - sku.retail_stock, available_to_transfer)
                if replenish_amount > 0:
                    arrival_time = current_time + sku.retail_lead_time_days
                    sku.retail_replenishment_queue.append((arrival_time, replenish_amount))
                    sku.open_pos_warehouse_to_retail += replenish_amount
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
            'Location_1_stock': {sku_id: sku.current_stock for sku_id, sku in self.skus.items()},
            'Location_2_stock': {sku_id: sku.current_stock for sku_id, sku in self.skus.items()},
            'Location_3_stock': {sku_id: sku.current_stock for sku_id, sku in self.skus.items()},
            'retail_stock': {sku_id: sku.retail_stock for sku_id, sku in self.skus.items()},
            'open_pos': {sku_id: sku.open_pos for sku_id, sku in self.skus.items()},
            'stockouts': stockouts,
            'service_levels': service_levels,
            'supplier_loads': {supplier: info['current_load'] for supplier, info in self.suppliers.items()},
            'supplier_reliability': {supplier: info['reliability'] for supplier, info in self.suppliers.items()},
            'time_deltas': time_deltas,
            'current_time': current_time,
            'lead_times': {sku_id: sku.lead_time_days for sku_id, sku in self.skus.items()},
            'open_pos_supplier_to_warehouse': {sku_id: sku.open_pos_supplier_to_warehouse for sku_id, sku in self.skus.items()},
            'open_pos_warehouse_to_retail': {sku_id: sku.open_pos_warehouse_to_retail for sku_id, sku in self.skus.items()},
        }
        self.config['current_time'] += self.config['min_decision_interval']
        done = False
        if any(sku.retail_stock <= 0 for sku in self.skus.values()):
            done = True
            rewards -= 20
        # After reward calculation, adjust service level penalty 
        avg_service_level = np.mean(list(service_levels.values()))
        if avg_service_level < 0.99:
            penalty = -250 * (0.99 - avg_service_level)  # Was -500
            rewards += penalty
        # Ensure rewards are not NaN or inf, and clip total reward
        total_reward = np.sum(rewards)
        if not np.isfinite(total_reward):
            total_reward = 0.0
        total_reward = np.clip(total_reward, -1000, 2000)
        return self._get_observation(), total_reward, done, info
    #High safety stock for Type C is strongly indicated in the results where C experiences nearly zero stockouts.
    def calculate_dynamic_safety_stock(self, sku, z=None):
        """Calculate safety stock accurately based on demand variability and lead time, using fixed gamma params."""
        if z is None:
            z = 4.0 if sku.sku == 'Type_C' else 2.5
        # Use fixed gamma parameters from SKU
        alpha, beta = sku.alpha, sku.beta
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
            self.skus[sku_id].open_pos_supplier_to_warehouse = 0
            self.skus[sku_id].open_pos_warehouse_to_retail = 0
            # Calculate and store retail reorder point
            self.skus[sku_id].retail_reorder_point = self.calculate_rop(sku, location='retail')
        # Reset supplier loads
        for supplier in self.suppliers.values():
            supplier['current_load'] = 0
        return self._get_observation()
    # Update the lead times and the delivery dates 
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

    
