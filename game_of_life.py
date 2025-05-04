import torch
import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
import time
import argparse
from vispy.color import ColorArray
from PyQt5.QtCore import Qt

DEFAULT_SIZE = 500
DEFAULT_INTERVAL = 60 

class GameOfLife:
    def __init__(self, size=DEFAULT_SIZE, random_seed=None, device='cuda'):
        self.size = size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Initialize the grid and age grid on specified device
        self.grid = torch.randint(0, 2, (size, size), dtype=torch.float32, device=self.device)
        self.age_grid = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        
        # Create convolution kernel for counting neighbors
        self.kernel = torch.tensor([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        
        # Pre-allocate padded grid to avoid memory allocation during updates
        self.padded_grid = torch.zeros((size + 2, size + 2), dtype=torch.float32, device=self.device)
        
        # Pre-allocate tensors for rules to avoid memory allocation
        self.underpop = torch.tensor(0.0, device=self.device)
        self.overpop = torch.tensor(0.0, device=self.device)
        self.reproduce = torch.tensor(1.0, device=self.device)
    
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
            torch.tensor(0.0, device=self.device)
        )
        
        self.grid = new_grid
    
    def get_grid(self):
        return self.grid.cpu().numpy()
    
    def get_age_grid(self):
        return self.age_grid.cpu().numpy()

def animate_game(size=DEFAULT_SIZE, interval=DEFAULT_INTERVAL, random_seed=None, frame_skip=1, device='cuda'):
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if size <= 0:
        raise ValueError("size must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")
        
    game = GameOfLife(size=size, random_seed=random_seed, device=device)
    
    # Create a canvas and view
    canvas = vispy.scene.SceneCanvas(keys='interactive', size=(1920, 1080), resizable=True, show=True)
    canvas.native.setWindowState(Qt.WindowMaximized)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = size * 1

    # Create text display for generation counter
    text = visuals.Text('Generation: 0', pos=(100, 50), color='white', font_size=12, parent=canvas.scene)
    text.order = 1  # Ensure text is drawn on top

    # Create scatter plot
    scatter = visuals.Markers()
    view.add(scatter)

    # Initialize with empty data
    pos = np.zeros((1, 3))
    colors = np.array([(0, 0, 0, 0)])  # Transparent
    scatter.set_data(pos, edge_color=None, face_color=colors, size=10)

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
        # Use a non-linear scale: very quick initial phase, longer taper
        if age < 20:  # Quick initial phase (0-10)
            normalized_age = age / 20.0
            if normalized_age < 0.1:
                return (0.0, 0.8, 0, 0.9)  # Less green, more yellow-green

            elif normalized_age < 0.4:
                return (0.6, 0.8, 0, 0.9)  # Yellow-green
            elif normalized_age < 0.6:
                return (0.8, 0.8, 0, 0.9)  # Yellow
            elif normalized_age < 0.8:
                return (1.0, 0.6, 0, 0.9)  # Orange-yellow
            else:
                return (1.0, 0.4, 0, 0.9)  # Orange
        elif age < 50:  # Middle phase (50-200)
            normalized_age = (age - 50) / 50.0
            if normalized_age < 0.5:
                return (1.0, 0.3, 0, 0.9)  # Red-orange
            else:
                return (0.8, 0.2, 0, 0.9)  # Brown
        elif age < 100:
            return (0.4, 0.1, 0, 0.9)  # Dark brown
        else:
            return (0.2, 0.05, 0, 0.9)  # Very dark brown

    def update(ev):
        # Update game state multiple times per frame if frame_skip > 1
        for _ in range(frame_skip):
            game.update()
            
        grid = game.get_grid()
        age_grid = game.get_age_grid()
        
        # Update generation counter
        text.text = f'Generation: {int(age_grid.max())}'
        
        # Get coordinates of live cells
        live_xs, live_ys = np.where(grid == 1)
        live_zs = np.full_like(live_xs, 0.1, dtype=float)
        
        # Get ages of live cells
        live_ages = age_grid[grid == 1]
        
        # Create positions array
        pos = np.column_stack((live_xs, live_ys, live_zs))
        
        # Create colors array
        colors = np.array([get_color(age) for age in live_ages])

        # --- Scale marker size with distance from camera ---
        # Get camera position in world coordinates
        cam = view.camera
        cam_pos = np.array(cam.transform.map([0, 0, cam.distance, 1])[:3])
        # Compute distance from each point to camera
        if len(pos) > 0:
            distances = np.linalg.norm(pos - cam_pos, axis=1)
            # Inverse scale: closer = bigger, farther = smaller
            sizes = np.clip(6000 / (distances + 1), 4, 16)  # Increased scaling and min/max for larger points
        else:
            sizes = 6
        # --------------------------------------------------

        # Update scatter plot
        scatter.set_data(pos, edge_color=None, face_color=colors, size=sizes)
        
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
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                      help=f'Size of the grid (default: {DEFAULT_SIZE})')
    parser.add_argument('--seed', type=str, default='time',
                      help='Seed type: "none" for no seed, "time" for time-based seed, or a number for custom seed')
    parser.add_argument('--interval', type=int, default=DEFAULT_INTERVAL,
                      help=f'Animation interval in milliseconds (default: {DEFAULT_INTERVAL})')
    parser.add_argument('--frame-skip', type=int, default=1,
                      help='Number of game updates per frame')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run on: "cuda" or "cpu" (default: cuda)')
    args = parser.parse_args()

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
    animate_game(size=args.size, interval=args.interval, random_seed=random_seed, 
                frame_skip=args.frame_skip, device=args.device) 