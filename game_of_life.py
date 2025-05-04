import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

class GameOfLife:
    def __init__(self, size=100, random_seed=None):
        self.size = size
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Initialize the grid and age grid on GPU
        self.grid = torch.randint(0, 2, (size, size), dtype=torch.float32, device='cuda')
        self.age_grid = torch.zeros((size, size), dtype=torch.float32, device='cuda')
        
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
        new_grid = torch.where(
            (self.grid == 1) & ((neighbors < 2) | (neighbors > 3)),
            torch.tensor(0.0, device='cuda'),
            torch.where(
                (self.grid == 0) & (neighbors == 3),
                torch.tensor(1.0, device='cuda'),
                self.grid
            )
        )
        
        # Update age grid
        self.age_grid = torch.where(
            new_grid == 1,
            self.age_grid + 1,
            torch.tensor(0.0, device='cuda')
        )
        
        self.grid = new_grid
    
    def get_grid(self):
        return self.grid.cpu().numpy()
    
    def get_age_grid(self):
        return self.age_grid.cpu().numpy()

def animate_game(size=100, frames=200, interval=50):
    game = GameOfLife(size=size, random_seed=42)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.style.use('dark_background')

    def update(frame):
        ax.clear()
        game.update()
        grid = game.get_grid()
        age_grid = game.get_age_grid()
        
        # Get coordinates of live cells
        xs, ys = np.where(grid == 1)
        zs = np.full_like(xs, 0.5, dtype=float)  # All spheres at z=0.5
        
        # Get ages of live cells
        ages = age_grid[grid == 1]
        
        # Create color map based on age
        colors = plt.cm.viridis(ages / 10)  # Normalize ages to [0,1] range
        
        # Draw spheres for live cells with age-based colors
        ax.scatter(xs, ys, zs, s=60, c=colors, edgecolors='white', alpha=0.9, marker='o', depthshade=True)
        
        # Set limits and appearance
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_zlim(0, 1.5)
        ax.set_title('Go and live among yourself', fontsize=14, pad=20)
        ax.set_xlabel('X', labelpad=10)
        ax.set_ylabel('Y', labelpad=10)
        ax.set_zlabel('Alive', labelpad=10)
        ax.view_init(elev=30, azim=frame * 1.8)
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
    plt.show()

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU instead.")
        device = 'cpu'
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Run the animation
    animate_game(size=100, frames=200, interval=50) 