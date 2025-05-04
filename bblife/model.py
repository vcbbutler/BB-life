import torch
import numpy as np
from .constants import DEFAULT_SIZE, DEFAULT_INITIAL_DENSITY, DEFAULT_MUTATION_RATE

class GameOfLife:
    def __init__(self, size=DEFAULT_SIZE, initial_density=DEFAULT_INITIAL_DENSITY, random_seed=None, device='cuda', 
                 mutation_rate=DEFAULT_MUTATION_RATE):
        self.size = size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.mutation_rate = mutation_rate
        
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Initialize the grid and age grid
        self.grid = torch.bernoulli(torch.full((size, size), initial_density, device=self.device))
        self.age_grid = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        
        # Initialize history grids (t-2, t-3, t-4)
        self.grid_t_minus_2 = torch.zeros_like(self.grid, device=self.device) 
        self.grid_t_minus_3 = torch.zeros_like(self.grid, device=self.device)
        self.grid_t_minus_4 = torch.zeros_like(self.grid, device=self.device)
        
        # Create convolution kernel for counting neighbors
        self.kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Pre-allocate tensors for rules to avoid memory allocation
        self.underpop = torch.tensor(0.0, device=self.device)
        self.overpop = torch.tensor(0.0, device=self.device)
        self.reproduce = torch.tensor(1.0, device=self.device)
    
    def update(self):
        # Store the grid state from t-1 before it's updated
        grid_t_minus_1 = self.grid.clone() 
        # Store grid state from t-4 for the stability check below
        grid_t_minus_4_snapshot = self.grid_t_minus_4.clone() 
        
        # Shift history grids forward (t-4 becomes t-3, t-3 becomes t-2, t-2 becomes t-1)
        # The current self.grid (t-1) becomes grid_t_minus_2 for the *next* iteration's check
        self.grid_t_minus_4 = self.grid_t_minus_3
        self.grid_t_minus_3 = self.grid_t_minus_2
        self.grid_t_minus_2 = grid_t_minus_1 

        # Pad the grid (state at t-1) with circular boundaries
        grid_float = grid_t_minus_1.float() # Use the t-1 state for neighbor calculation
        padded_grid = torch.nn.functional.pad(
            grid_float.unsqueeze(0).unsqueeze(0), 
            (1, 1, 1, 1),                      
            mode='circular'
        ) 
        
        # Count neighbors based on t-1 state
        neighbors = torch.nn.functional.conv2d(
            padded_grid, 
            self.kernel,
            padding=0 
        ).squeeze() 
        
        # Apply standard Game of Life rules based on t-1 state
        is_alive_t_minus_1 = (grid_t_minus_1 == 1.0)
        is_dead_t_minus_1 = ~is_alive_t_minus_1 

        survives = is_alive_t_minus_1 & ((neighbors == 2) | (neighbors == 3))
        births = is_dead_t_minus_1 & (neighbors == 3)

        # Calculate potential new grid state at time t (before any mutations)
        new_grid_potential = torch.zeros_like(self.grid)
        new_grid_potential[survives | births] = 1.0
        
        # 1. Identify mutation triggers based on age of potentially live cells
        is_alive_potential = (new_grid_potential == 1.0)
        # Calculate age-dependent mutation probability map (log scale, capped at 1.0)
        # Using age_grid from *previous* step before update for consistency
        mutation_prob_map = torch.clamp(self.mutation_rate * torch.log1p(self.age_grid), 0.0, 1.0) 
        
        # Determine which triggers fire based on age probability for potentially live cells
        active_triggers = is_alive_potential & (torch.rand_like(new_grid_potential, device=self.device) < mutation_prob_map)
        
        # Initialize the grid for this step with the potential state
        new_grid = new_grid_potential.clone()
        
        # 2. Determine mutation targets (self or neighbor) and apply flips if any triggers fired
        if torch.any(active_triggers):
            trigger_coords = active_triggers.nonzero(as_tuple=False) # Get coords on device
            cells_to_flip = torch.zeros_like(new_grid, dtype=torch.bool, device=self.device)

            # Define neighbor offsets (reuse existing logic)
            neighbor_offsets = torch.tensor([
                [-1, -1], [-1, 0], [-1, 1],
                [ 0, -1],          [ 0, 1],
                [ 1, -1], [ 1, 0], [ 1, 1]
            ], dtype=torch.long, device=self.device) # Move offsets to device

            # Randomly decide target (self or neighbor) for each trigger
            # Create random tensor for decisions (0 = self, 1 = neighbor)
            target_decisions = torch.rand(len(trigger_coords), device=self.device) < 0.5
            
            # Get coordinates for self-mutations
            self_mutation_coords = trigger_coords[target_decisions]
            if len(self_mutation_coords) > 0:
                cells_to_flip[self_mutation_coords[:, 0], self_mutation_coords[:, 1]] = True

            # Get coordinates for neighbor-mutations
            neighbor_mutation_triggers = trigger_coords[~target_decisions]
            if len(neighbor_mutation_triggers) > 0:
                # Choose random neighbor offset for each neighbor mutation trigger
                offset_indices = torch.randint(0, 8, (len(neighbor_mutation_triggers),), device=self.device)
                offsets = neighbor_offsets[offset_indices]
                
                # Calculate neighbor coordinates with toroidal wrapping
                neighbor_coords_r = (neighbor_mutation_triggers[:, 0] + offsets[:, 0]) % self.size
                neighbor_coords_c = (neighbor_mutation_triggers[:, 1] + offsets[:, 1]) % self.size
                
                # Mark neighbors to flip (handle potential duplicates safely)
                # Use advanced indexing
                cells_to_flip[neighbor_coords_r, neighbor_coords_c] = True
            
            # Apply the flips to the new_grid (0 -> 1, 1 -> 0)
            new_grid[cells_to_flip] = 1.0 - new_grid[cells_to_flip]

        # Update age grid based on the final mutated new_grid
        self.age_grid = torch.where(
            new_grid == 1.0,
            self.age_grid + 1,
            torch.tensor(0.0, device=self.device)
        )
        
        # Update the main grid state to the final mutated state for the next iteration
        self.grid = new_grid
    
    def get_grid(self):
        return self.grid.cpu().numpy()
    
    def get_age_grid(self):
        return self.age_grid.cpu().numpy() 