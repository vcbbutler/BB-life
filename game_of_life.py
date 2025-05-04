import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GameOfLife:
    def __init__(self, size=100, random_seed=None):
        self.size = size
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Initialize the grid on GPU
        self.grid = torch.randint(0, 2, (size, size), dtype=torch.float32, device='cuda')
        
        # Create convolution kernel for counting neighbors
        self.kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32, device='cuda').view(1, 1, 3, 3)
        
        # Pad the grid for convolution
        self.padded_grid = torch.zeros((size + 2, size + 2), dtype=torch.float32, device='cuda')
    
    def update(self):
        # Update the padded grid
        self.padded_grid[1:-1, 1:-1] = self.grid
        
        # Count neighbors using convolution
        neighbors = torch.nn.functional.conv2d(
            self.padded_grid.view(1, 1, self.size + 2, self.size + 2),
            self.kernel,
            padding=0
        ).squeeze()
        
        # Apply Game of Life rules
        # Rule 1: Any live cell with fewer than 2 live neighbors dies (underpopulation)
        # Rule 2: Any live cell with 2 or 3 live neighbors lives
        # Rule 3: Any live cell with more than 3 live neighbors dies (overpopulation)
        # Rule 4: Any dead cell with exactly 3 live neighbors becomes alive (reproduction)
        self.grid = torch.where(
            (self.grid == 1) & ((neighbors < 2) | (neighbors > 3)),
            torch.tensor(0.0, device='cuda'),
            torch.where(
                (self.grid == 0) & (neighbors == 3),
                torch.tensor(1.0, device='cuda'),
                self.grid
            )
        )
    
    def get_grid(self):
        return self.grid.cpu().numpy()

def animate_game(size=100, frames=100, interval=100):
    game = GameOfLife(size=size, random_seed=42)
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        game.update()
        ax.imshow(game.get_grid(), cmap='binary')
        ax.set_title(f'Game of Life - Frame {frame}')
        ax.axis('off')
    
    anim = FuncAnimation(fig, update, frames=frames, interval=interval)
    plt.show()

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU instead.")
        device = 'cpu'
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Run the animation
    animate_game(size=100, frames=100, interval=100) 