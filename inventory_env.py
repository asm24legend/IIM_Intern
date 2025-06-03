import numpy as np
import gym
from gym import spaces
from dataclasses import dataclass
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
    forecasted_demand: float
    eoq: int
    max_stock: int
    min_order_qty: int
    inventory_location: str
    supplier: str
    open_pos: int
    last_order_date: datetime  # Changed from str to datetime
    next_delivery_date: datetime  # Changed from str to datetime
    previous_demand: float = 0
    retail_stock: int = 0  # Stock at retail store
    # Add seasonal demand parameters
    base_demand: float = 0  # Base demand level
    amplitude: float = 0    # Amplitude of seasonal variation
    frequency: float = 0    # Frequency of seasonal cycle
    phase: float = 0        # Phase shift of seasonal pattern
    last_decision_time: float = 0  # New field to track when last decision was made
    # Service level tracking
    total_demand: float = 0.0  # Total historical demand
    fulfilled_demand: float = 0.0  # Total fulfilled demand
    stockout_occasions: int = 0  # Number of stockout events
    replenishment_cycles: int = 0  # Number of replenishment cycles

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
            'min_lead_time': 1,
            'max_lead_time': 14,
            'retail_replenishment_time': 1,  # Days to replenish retail from warehouse
            'noise_std': 10,
            'min_decision_interval': 0.1,  # Minimum time between decisions (in days)
            'current_time': 0.0,  # Current simulation time in days (float)
            'lead_time_reduction_cost': 50,  # Cost per day of lead time reduction
            'max_lead_time_reduction': 3  # Maximum days by which lead time can be reduced
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

        # Initialize SKUs with seasonal demand parameters
        base_date = datetime.now()
        self.skus = {
            'Type_A': SKUData(
                sku='Type_A',
                description='Product Type A',
                current_stock=450,
                reorder_point=300,
                safety_stock=200,
                lead_time_days=7,
                forecasted_demand=400,
                eoq=250,
                max_stock=1000,
                min_order_qty=100,
                inventory_location='Location_1',
                supplier='Supplier_X',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=7),
                retail_stock=100,
                base_demand=400,
                amplitude=150,
                frequency=2*np.pi/365,  # One year cycle
                phase=0,
                last_decision_time=0.0
            ),
            'Type_B': SKUData(
                sku='Type_B',
                description='Product Type B',
                current_stock=350,
                reorder_point=250,
                safety_stock=150,
                lead_time_days=10,
                forecasted_demand=300,
                eoq=200,
                max_stock=800,
                min_order_qty=80,
                inventory_location='Location_2',
                supplier='Supplier_Y',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=10),
                retail_stock=80,
                base_demand=300,
                amplitude=100,
                frequency=2*np.pi/182.5,  # Six month cycle
                phase=np.pi/2,
                last_decision_time=0.0
            ),
            'Type_C': SKUData(
                sku='Type_C',
                description='Product Type C',
                current_stock=250,
                reorder_point=200,
                safety_stock=100,
                lead_time_days=5,
                forecasted_demand=200,
                eoq=150,
                max_stock=600,
                min_order_qty=50,
                inventory_location='Location_3',
                supplier='Supplier_Y',
                open_pos=0,
                last_order_date=base_date,
                next_delivery_date=base_date + timedelta(days=5),
                retail_stock=60,
                base_demand=200,
                amplitude=80,
                frequency=2*np.pi/91.25,  # Three month cycle
                phase=np.pi/4,
                last_decision_time=0.0
            )
        }
        
        # Initialize suppliers with their capabilities and products
        self.suppliers = {
            'Supplier_X': {
                'lead_time_range': (5, 10),
                'current_load': 0,
                'reliability': 0.95,
                'products': ['Type_A', 'Type_B']
            },
            'Supplier_Y': {
                'lead_time_range': (7, 12),
                'current_load': 0,
                'reliability': 0.90,
                'products': ['Type_B', 'Type_C']
            }
        }
        
        num_skus = len(self.skus)
        
        # Action space: [order_quantities, lead_time_reductions]
        self.action_space = spaces.Box(
            low=np.array([0] * num_skus + [0] * num_skus),  # [order_qty, lead_time_reduction]
            high=np.array([self.config['max_inventory']] * num_skus + [self.config['max_lead_time_reduction']] * num_skus),
            dtype=np.int32
        )
        
        # Simplified state space: only inventory levels
        self.observation_space = spaces.Box(
            low=0,
            high=self.config['max_inventory'],
            shape=(num_skus,),
            dtype=np.int32
        )
        
        self.reset()

    def generate_seasonal_demand(self, sku: SKUData, time_period: float) -> float:
        """Generate seasonal demand using cosine function with noise for a specific time period"""
        seasonal_component = sku.amplitude * np.cos(
            sku.frequency * time_period + sku.phase
        )
        noise = np.random.normal(0, self.config['noise_std'])
        demand = max(0, sku.base_demand + seasonal_component + noise)
        return demand

    def calculate_demand_for_period(self, sku: SKUData, start_time: float, end_time: float) -> float:
        """Calculate total demand over a time period using numerical integration"""
        num_points = max(10, int((end_time - start_time) * 24))  # At least 10 points, or one per hour
        time_points = np.linspace(start_time, end_time, num_points)
        demands = [self.generate_seasonal_demand(sku, t) for t in time_points]
        return np.trapz(demands, time_points)  # Integrate demand over time

    def calculate_lead_time_demand(self, sku_id: str) -> float:
        """Calculate demand during lead time considering seasonality"""
        sku = self.skus[sku_id]
        total_demand = 0
        current_time = self.config['current_time']
        
        # Project demand over lead time period
        for t in range(sku.lead_time_days):
            self.config['current_time'] = current_time + t
            daily_demand = self.generate_seasonal_demand(sku, current_time + t)
            total_demand += daily_demand
        
        # Reset time step
        self.config['current_time'] = current_time
        return total_demand * 1.2  # Add 20% safety factor

    def calculate_eoq(self, sku_id: str) -> int:
        """Calculate Economic Order Quantity for a specific SKU"""
        sku = self.skus[sku_id]
        D = sku.forecasted_demand  # Annual demand
        K = self.config['order_cost']  # Order cost
        H = self.config['holding_cost']  # Holding cost
        return int(np.sqrt((2 * D * K) / H))

    def calculate_rop(self, sku_id: str) -> int:
        """Calculate Reorder Point considering lead time demand"""
        sku = self.skus[sku_id]
        lead_time_demand = self.calculate_lead_time_demand(sku_id)
        
        # Adjust ROP if lead time demand exceeds forecasted demand
        if lead_time_demand > sku.forecasted_demand:
            safety_stock_adjustment = int(0.2 * lead_time_demand)  # Increase safety stock by 20%
            return int(lead_time_demand + safety_stock_adjustment)
        
        return int(lead_time_demand + sku.safety_stock)

    def select_best_supplier(self, sku_id: str) -> str:
        """Select the best supplier based on load and reliability"""
        sku = self.skus[sku_id]
        best_supplier = sku.supplier
        min_load = float('inf')
        
        for supplier_id, supplier in self.suppliers.items():
            if sku_id in supplier['products']:
                # Consider both load and reliability
                effective_load = supplier['current_load'] / supplier['reliability']
                if effective_load < min_load:
                    min_load = effective_load
                    best_supplier = supplier_id
        
        return best_supplier

    def adjust_delivery_date(self, sku_id: str, current_demand: float):
        """Adjust delivery date based on demand changes"""
        sku = self.skus[sku_id]
        if sku.open_pos > 0:
            demand_change = current_demand - sku.previous_demand
            
            if demand_change > 100:  # Significant increase in demand
                # Advance delivery date
                current_date = sku.next_delivery_date
                new_date = current_date - timedelta(days=2)
                self.skus[sku_id].next_delivery_date = new_date
            elif demand_change < -50:  # Significant decrease in demand
                # Delay delivery date
                current_date = sku.next_delivery_date
                new_date = current_date + timedelta(days=2)
                self.skus[sku_id].next_delivery_date = new_date

    def calculate_reward(self, sku_id: str, stockout: int, current_stock: int, 
                           daily_demand: float, current_lead_time: int, previous_lead_time: int) -> float:
        """
        Calculate reward with stronger emphasis on service level and lead time reduction:
        1. Stockout penalty: -50 per unit stockout
        2. Service level reward: +20 for fulfilling demand
        3. Inventory level penalties to maintain efficiency
        4. Lead time reduction rewards
        """
        reward = 0.0
        sku = self.skus[sku_id]
        
        # Major penalty for stockouts
        if stockout > 0:
            reward -= 50 * stockout
        else:
            # Reward for fulfilling demand
            reward += 20
        
        # Inventory level penalties
        if current_stock > self.calculate_eoq(sku_id):
            # Penalty for excess inventory
            reward -= (current_stock - self.calculate_eoq(sku_id)) * 0.1
        elif current_stock < sku.safety_stock:
            # Penalty for being below safety stock
            reward -= (sku.safety_stock - current_stock) * 0.2
        
        # Small reward for optimal inventory level
        if sku.safety_stock <= current_stock <= self.calculate_eoq(sku_id):
            reward += 5
        
        # Reward for lead time reduction
        if current_lead_time < previous_lead_time:
            # Reward proportional to the reduction achieved
            lead_time_improvement = previous_lead_time - current_lead_time
            reward += 10 * lead_time_improvement  # +10 points per day reduced
            
            # Additional bonus for maintaining short lead times
            if current_lead_time <= self.config['min_lead_time'] + 2:  # Within 2 days of minimum
                reward += 15
        
        return reward

    def _get_state(self):
        """Convert current system state to observation space format - only inventory levels"""
        return np.array([sku.current_stock for sku in self.skus.values()], dtype=np.int32)

    def _replenish_retail(self, sku_id: str, demand: int):
        """Transfer inventory from warehouse to retail store"""
        sku = self.skus[sku_id]
        location = self.inventory_locations[sku.inventory_location]
        
        # Calculate replenishment quantity
        shortage = max(0, demand - sku.retail_stock)
        replenishment = min(shortage, sku.current_stock)
        
        # Update stock levels
        sku.retail_stock += replenishment
        sku.current_stock -= replenishment
        location.current_stock[sku_id] = sku.current_stock
        
        return replenishment

    def step(self, action):
        rewards = np.zeros(len(self.skus))
        info = {}
        supplier_loads = defaultdict(float)
        stockouts = {}  # Initialize stockouts dictionary for all SKUs
        service_levels = {}  # Track service levels for all SKUs
        
        # Split action into order quantities and lead time reductions
        num_skus = len(self.skus)
        order_quantities = action[:num_skus]
        lead_time_reductions = action[num_skus:]
        
        # Calculate time since last decision for each SKU
        current_time = self.config['current_time']
        time_deltas = {
            sku_id: max(self.config['min_decision_interval'], current_time - sku.last_decision_time)
            for sku_id, sku in self.skus.items()
        }
        
        for i, (sku_id, sku) in enumerate(self.skus.items()):
            # Calculate demand since last decision
            period_demand = self.calculate_demand_for_period(
                sku,
                sku.last_decision_time,
                current_time
            )
            
            # Update total demand
            sku.total_demand += period_demand
            
            # Handle retail level transactions
            fulfilled_retail_demand = min(period_demand, sku.retail_stock)
            retail_stockout = period_demand - fulfilled_retail_demand
            
            # Update service level metrics
            sku.fulfilled_demand += fulfilled_retail_demand
            if retail_stockout > 0:
                sku.stockout_occasions += 1
            
            stockouts[sku_id] = retail_stockout  # Track stockout for each SKU
            
            # Calculate service levels (β / fill rate)
            if sku.total_demand > 0:
                service_levels[sku_id] = sku.fulfilled_demand / sku.total_demand
            else:
                service_levels[sku_id] = 1.0
            
            # Update retail stock
            sku.retail_stock -= fulfilled_retail_demand
            
            # Replenish retail from warehouse if needed
            if sku.retail_stock < sku.safety_stock:
                replenished = self._replenish_retail(sku_id, sku.safety_stock - sku.retail_stock)
            
            # Process warehouse level operations
            if sku.open_pos > 0 and self._is_delivery_due(sku, current_time):
                sku.replenishment_cycles += 1  # Count completed replenishment cycles
                self.skus[sku_id].current_stock += sku.open_pos
                self.inventory_locations[sku.inventory_location].current_stock[sku_id] += sku.open_pos
                supplier_loads[sku.supplier] += sku.open_pos
                self.skus[sku_id].open_pos = 0
            
            # Process new orders and lead time reductions
            if order_quantities[i] > 0:
                current_rop = self.calculate_rop(sku_id)
                # Check if we're below reorder point AND have capacity for new orders
                if sku.current_stock < current_rop and sku.current_stock < sku.max_stock:
                    best_supplier = self.select_best_supplier(sku_id)
                    
                    base_order = max(order_quantities[i], sku.min_order_qty)
                    if period_demand > sku.base_demand * time_deltas[sku_id] * 1.5:
                        base_order *= 1.5
                    
                    # Ensure order won't exceed max stock capacity
                    available_capacity = sku.max_stock - (sku.current_stock + sku.open_pos)
                    order_qty = min(base_order, available_capacity)
                    
                    if order_qty > 0:
                        self.skus[sku_id].supplier = best_supplier
                        self.skus[sku_id].open_pos += order_qty
                        supplier_loads[best_supplier] += order_qty
                        
                        # Apply lead time reduction if requested
                        lead_time_reduction = int(lead_time_reductions[i])
                        if lead_time_reduction > 0:
                            # Calculate cost of lead time reduction
                            reduction_cost = lead_time_reduction * self.config['lead_time_reduction_cost']
                            # Apply reduction if supplier reliability is good enough
                            supplier_reliability = self.suppliers[best_supplier]['reliability']
                            if supplier_reliability >= 0.8:  # Only allow reduction for reliable suppliers
                                original_lead_time = self.skus[sku_id].lead_time_days
                                reduced_lead_time = max(
                                    self.config['min_lead_time'],
                                    original_lead_time - lead_time_reduction
                                )
                                self.skus[sku_id].lead_time_days = reduced_lead_time
                                # Apply cost penalty for lead time reduction
                                rewards[i] -= reduction_cost
                        
                        self._update_delivery_date(sku_id)
                        self.adjust_delivery_date(sku_id, period_demand)
            
            # Store previous demand and update last decision time
            self.skus[sku_id].previous_demand = period_demand
            self.skus[sku_id].last_decision_time = current_time
            
            # Calculate rewards
            rewards[i] = self.calculate_reward(
                sku_id,
                retail_stockout,
                self.skus[sku_id].current_stock,
                period_demand / time_deltas[sku_id],  # Convert to daily rate
                self._get_current_lead_time(sku_id),
                sku.lead_time_days
            )
        
        # Update supplier metrics
        for supplier_id, load in supplier_loads.items():
            self.suppliers[supplier_id]['current_load'] = min(1.0, load / 1000)
            self.suppliers[supplier_id]['reliability'] = max(0.7, 0.95 - 0.1 * self.suppliers[supplier_id]['current_load'])
        
        # Update info dictionary
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
            'lead_times': {sku_id: sku.lead_time_days for sku_id, sku in self.skus.items()}
        }
        
        # Advance simulation time by minimum decision interval
        self.config['current_time'] += self.config['min_decision_interval']
        
        # Check terminal conditions
        done = False
        if any(sku.retail_stock <= 0 for sku in self.skus.values()):
            done = True
            rewards -= 20  # Terminal state penalty for retail stockout
        
        return self._get_state(), np.sum(rewards), done, info
    
    def reset(self):
        # Reset simulation time
        self.config['current_time'] = 0.0
        
        # Reset SKUs to initial state
        base_date = datetime.now()
        for sku_id, sku in self.skus.items():
            self.skus[sku_id].current_stock = sku.reorder_point
            self.skus[sku_id].open_pos = 0
            self.skus[sku_id].retail_stock = int(sku.safety_stock * 1.5)
            self.skus[sku_id].last_decision_time = 0.0
            self.skus[sku_id].last_order_date = base_date
            self.skus[sku_id].next_delivery_date = base_date + timedelta(days=sku.lead_time_days)
            # Reset service level metrics
            self.skus[sku_id].total_demand = 0.0
            self.skus[sku_id].fulfilled_demand = 0.0
            self.skus[sku_id].stockout_occasions = 0
            self.skus[sku_id].replenishment_cycles = 0
            self.skus[sku_id].previous_demand = 0.0
        
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
        """Update next delivery date based on supplier lead time"""
        sku = self.skus[sku_id]
        supplier = self.suppliers[sku.supplier]
        lead_time = np.random.randint(*supplier['lead_time_range'])
        self.skus[sku_id].lead_time_days = lead_time
        self.skus[sku_id].last_order_date = datetime.now()
        self.skus[sku_id].next_delivery_date = self.skus[sku_id].last_order_date + timedelta(days=lead_time)

    def _get_current_lead_time(self, sku_id: str) -> int:
        """Get current lead time for a SKU"""
        return self.skus[sku_id].lead_time_days
    
    def _get_days_to_delivery(self, sku_id: str) -> int:
        """Calculate days until next delivery"""
        # In a real implementation, calculate from current date
        return self.skus[sku_id].lead_time_days 