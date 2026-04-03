import numpy as np
from gym import spaces

def None_to_empty_list(var):
   
    if isinstance(var, (list, tuple, np.ndarray)):
        return var
    if var is None:
        return []
    return [var]

class BaseActDecoder:
    def __init__(self, act_params, temp_db, simulator):
        self.temp_db   = temp_db
        self.simulator = simulator
        
        # Define truck-specific actions
        self.truck_discrete_outputs = act_params.get('truck_discrete_outputs', [
            'truck_target_node' 
        ])
        self.truck_binary_discrete = act_params.get('truck_binary_discrete', [
            'truck_wait'  
        ])
        
        # Define drone-specific actions 
        self.drone_discrete_outputs = act_params.get('drone_discrete_outputs', [
            'drone_service_node',     
            'drone_rendezvous_truck'   
        ])
        self.drone_binary_discrete = act_params.get('drone_binary_discrete', [
            'drone_continue',  
        ])

        # Validate allowed actions
        allowed_truck_d = {'truck_target_node'}
        allowed_truck_b = {'truck_wait'}
        allowed_drone_d = {'drone_service_node', 'drone_rendezvous_truck',}
        allowed_drone_b = {'drone_continue'}
        
        if not set(self.truck_discrete_outputs).issubset(allowed_truck_d):
            raise ValueError(f"Unsupported truck discrete outputs: {self.truck_discrete_outputs}")
        if not set(self.truck_binary_discrete).issubset(allowed_truck_b):
            raise ValueError(f"Unsupported truck binary outputs: {self.truck_binary_discrete}")
        if not set(self.drone_discrete_outputs).issubset(allowed_drone_d):
            raise ValueError(f"Unsupported drone discrete outputs: {self.drone_discrete_outputs}")  
        if not set(self.drone_binary_discrete).issubset(allowed_drone_b):
            raise ValueError(f"Unsupported drone binary outputs: {self.drone_binary_discrete}")

        self.truck_act_spaces = []
        self.drone_act_spaces = []
        self.truck_func_dict = {}
        self.drone_func_dict = {}


    def reset(self):
      
        pass

    def finish_init(self):
        total_nodes = self.temp_db.num_nodes 
        num_trucks = self.temp_db.num_trucks
        
        # print(f"ActionDecoder.finish_init: total_nodes={total_nodes} (fixed), trucks={num_trucks}")
        
        for key in self.truck_discrete_outputs:
            self.truck_act_spaces.append(spaces.Discrete(total_nodes))
            self.truck_func_dict[key] = lambda a, k=key: (np.argmax(a) if hasattr(a, '__len__') else int(a))
        
        for key in self.truck_binary_discrete:
            self.truck_act_spaces.append(spaces.Discrete(2))
            self.truck_func_dict[key] = lambda a, k=key: int(a)
        
        for key in self.drone_discrete_outputs:
            if key == 'drone_rendezvous_truck':
                self.drone_act_spaces.append(spaces.Discrete(num_trucks))
            else:
                self.drone_act_spaces.append(spaces.Discrete(total_nodes))
            self.drone_func_dict[key] = lambda a, k=key: (np.argmax(a) if hasattr(a, '__len__') else int(a))
        
        for key in self.drone_binary_discrete:
            self.drone_act_spaces.append(spaces.Discrete(2))
            self.drone_func_dict[key] = lambda a, k=key: int(a)
        
        # print(f"Truck action spaces: {len(self.truck_act_spaces)} heads, node dim: {total_nodes}")
        # print(f"Drone action spaces: {len(self.drone_act_spaces)} heads, node dim: {total_nodes}")


    def action_space(self):
        """Return action spaces for all agents (trucks first, then drones)"""
        num_trucks = self.temp_db.num_trucks
        num_drones = self.temp_db.num_drones

        action_spaces = []
        
        for _ in range(num_trucks):
            action_spaces.append(spaces.Tuple(self.truck_act_spaces))
            
        for _ in range(num_drones):
            action_spaces.append(spaces.Tuple(self.drone_act_spaces))
            
        return spaces.Tuple(action_spaces)

    def _get_agent_type(self, agent_idx):
        """Determine if agent is truck or drone based on index"""
        num_trucks = self.temp_db.num_trucks
        if agent_idx < num_trucks:
            return 'truck', agent_idx  # truck index
        else:
            return 'drone', agent_idx - num_trucks  # drone index
