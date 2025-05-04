import torch
import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
import time
import argparse
from vispy.color import ColorArray

class GameOfLife:
    def __init__(self, size=200, random_seed=None):
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
        
        # Pre-allocate padded grid to avoid memory allocation during updates
        self.padded_grid = torch.zeros((size + 2, size + 2), dtype=torch.float32, device='cuda')
        
        # Pre-allocate tensors for rules to avoid memory allocation
        self.underpop = torch.tensor(0.0, device='cuda')
        self.overpop = torch.tensor(0.0, device='cuda')
        self.reproduce = torch.tensor(1.0, device='cuda')
    
    def update(self):
        # Update the padded grid
        self.padded_grid[1:-1, 1:-1] = self.grid
        
        # Count neighbors using convolution
        neighbors = torch.nn.functional.conv2d(
            self.padded_grid.view(1, 1, self.size + 2, self.size + 2),
            self.kernel,
            padding=0
        ).squeeze()
        
        # Apply Game of Life rules using pre-allocated tensors
        new_grid = torch.where(
            (self.grid == 1) & ((neighbors < 2) | (neighbors > 3)),
            self.underpop,
            torch.where(
                (self.grid == 0) & (neighbors == 3),
                self.reproduce,
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

def animate_game(size=200, frames=200, interval=50, random_seed=None, frame_skip=1):
    game = GameOfLife(size=size, random_seed=random_seed)
    
    # Create a canvas and view
    canvas = vispy.scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = size * 1.5

    # Create scatter plot
    scatter = visuals.Markers()
    view.add(scatter)

    # Initialize with empty data
    pos = np.zeros((1, 3))
    colors = np.array([(0, 0, 0, 0)])  # Transparent
    scatter.set_data(pos, edge_color='white', face_color=colors, size=10)

    # Create floor
    floor_vertices = np.array([
        [0, 0, 0],
        [size, 0, 0],
        [size, size, 0],
        [0, size, 0]
    ])
    floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
    floor = visuals.Mesh(vertices=floor_vertices, faces=floor_faces, color=(0.5, 0.5, 0.5, 0.2))
    view.add(floor)

    # Set up the view
    view.camera.set_range()
    view.camera.elevation = 20
    view.camera.azimuth = -45
    view.camera.distance = size * 1.5

    # Create color map for ages
    def get_color(age):
        age = min(age / 15.0, 1.0)
        if age < 0.33:
            return (0, 0.5, 0, 0.9)  # Green
        elif age < 0.66:
            return (0.5, 0.5, 0, 0.9)  # Yellow
        elif age < 0.9:
            return (0.8, 0.3, 0, 0.9)  # Red
        else:
            return (0.5, 0.3, 0, 0.9)  # Brown

    def update(ev):
        # Update game state multiple times per frame if frame_skip > 1
        for _ in range(frame_skip):
            game.update()
            
        grid = game.get_grid()
        age_grid = game.get_age_grid()
        
        # Get coordinates of live cells
        live_xs, live_ys = np.where(grid == 1)
        live_zs = np.full_like(live_xs, 0.1, dtype=float)
        
        # Get ages of live cells
        live_ages = age_grid[grid == 1]
        
        # Create positions array
        pos = np.column_stack((live_xs, live_ys, live_zs))
        
        # Create colors array
        colors = np.array([get_color(age) for age in live_ages])
        
        # Update scatter plot
        scatter.set_data(pos, edge_color='white', face_color=colors, size=10)
        
        # Force redraw
        canvas.update()

    # Create timer
    timer = vispy.app.Timer(interval=interval/1000.0)  # Convert ms to seconds
    timer.connect(update)
    timer.start()

    # Run the app
    vispy.app.run()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Game of Life with customizable seeding')
    parser.add_argument('--seed', type=str, default='time',
                      help='Seed type: "none" for no seed, "time" for time-based seed, or a number for custom seed')
    parser.add_argument('--interval', type=int, default=50,
                      help='Animation interval in milliseconds')
    parser.add_argument('--frame-skip', type=int, default=1,
                      help='Number of game updates per frame')
    args = parser.parse_args()

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU instead.")
        device = 'cpu'
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Handle seed selection
    if args.seed.lower() == 'none':
        random_seed = None
        print("Running with no seed (completely random)")
    elif args.seed.lower() == 'time':
        random_seed = int(time.time())
        print(f"Using time-based seed: {random_seed}")
    else:
        try:
            random_seed = int(args.seed)
            print(f"Using custom seed: {random_seed}")
        except ValueError:
            print("Invalid seed value. Using time-based seed instead.")
            random_seed = int(time.time())
            print(f"Using time-based seed: {random_seed}")
    
    # Run the animation with command line arguments
    animate_game(size=200, frames=200, interval=args.interval, random_seed=random_seed, frame_skip=args.frame_skip) 