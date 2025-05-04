import torch
import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
import time
import argparse
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSpinBox, QDoubleSpinBox, QPushButton, 
                            QComboBox, QGroupBox, QFormLayout)
from vispy.color import ColorArray

# Debug CUDA configuration
print("\n=== CUDA Configuration ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"cuDNN version: {torch.backends.cudnn.version() if torch.cuda.is_available() else 'N/A'}")
print(f"GPU device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
if torch.cuda.is_available():
    print(f"Current GPU device: {torch.cuda.current_device()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
print("========================\n")

DEFAULT_SIZE = 250
DEFAULT_INTERVAL = 30
DEFAULT_INITIAL_DENSITY = 0.5

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game of Life Settings")
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Grid Settings
        grid_group = QGroupBox("Grid Settings")
        grid_layout = QFormLayout()
        
        self.size_spin = QSpinBox()
        self.size_spin.setRange(100, 2000)
        self.size_spin.setValue(DEFAULT_SIZE)
        self.size_spin.setSingleStep(100)
        grid_layout.addRow("Grid Size:", self.size_spin)
        
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 0.9)
        self.density_spin.setValue(DEFAULT_INITIAL_DENSITY)
        self.density_spin.setSingleStep(0.1)
        self.density_spin.setDecimals(2)
        grid_layout.addRow("Initial Density:", self.density_spin)
        
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)
        
        # Animation Settings
        anim_group = QGroupBox("Animation Settings")
        anim_layout = QFormLayout()
        
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 1000)
        self.interval_spin.setValue(DEFAULT_INTERVAL)
        self.interval_spin.setSingleStep(10)
        anim_layout.addRow("Update Interval (ms):", self.interval_spin)
        
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 10)
        self.frame_skip_spin.setValue(1)
        anim_layout.addRow("Frame Skip:", self.frame_skip_spin)
        
        anim_group.setLayout(anim_layout)
        layout.addWidget(anim_group)
        
        # Device Settings
        device_group = QGroupBox("Device Settings")
        device_layout = QFormLayout()
        
        self.device_combo = QComboBox()
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self.device_combo.addItem("CUDA (GPU)", "cuda")
            self.device_combo.addItem("CPU", "cpu")
            self.device_combo.setCurrentIndex(0)  # Set CUDA as default
        else:
            self.device_combo.addItem("CPU (CUDA not available)", "cpu")
        device_layout.addRow("Device:", self.device_combo)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Start Simulation")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

class GameOfLife:
    def __init__(self, size=DEFAULT_SIZE, initial_density=DEFAULT_INITIAL_DENSITY, random_seed=None, device='cuda'):
        self.size = size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        if random_seed is not None:
            torch.manual_seed(random_seed)
        
        # Initialize the grid with specified density
        self.grid = torch.bernoulli(torch.full((size, size), initial_density, device=self.device))
        self.age_grid = torch.zeros((size, size), dtype=torch.float32, device=self.device)
        
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
        # Pad the grid with circular boundaries using torch.nn.functional.pad
        # Ensure grid is float32 for padding if it isn't already
        grid_float = self.grid.float() 
        padded_grid = torch.nn.functional.pad(
            grid_float.unsqueeze(0).unsqueeze(0), # Add batch and channel dims
            (1, 1, 1, 1),                      # Pad by 1 on all sides (left, right, top, bottom)
            mode='circular'
        ) # Keep batch and channel dims for conv2d
        
        # Count neighbors using convolution
        neighbors = torch.nn.functional.conv2d(
            padded_grid, 
            self.kernel,
            padding=0 # No padding needed as it's handled above
        ).squeeze() # Remove batch and channel dims after conv
        
        # Apply Game of Life rules using pre-allocated tensors
        # Ensure comparison tensors match the device and potentially dtype of neighbors and grid
        # (Assuming grid elements are 0.0 or 1.0 after bernoulli/update)
        is_alive = (self.grid == 1.0)
        is_dead = ~is_alive # Equivalent to self.grid == 0.0

        survives = is_alive & ((neighbors == 2) | (neighbors == 3))
        births = is_dead & (neighbors == 3)

        new_grid = torch.zeros_like(self.grid)
        new_grid[survives | births] = 1.0
        
        # Update age grid
        self.age_grid = torch.where(
            new_grid == 1.0,
            self.age_grid + 1,
            torch.tensor(0.0, device=self.device)
        )
        
        self.grid = new_grid
    
    def get_grid(self):
        return self.grid.cpu().numpy()
    
    def get_age_grid(self):
        return self.age_grid.cpu().numpy()

class SparseGameOfLife:
    """A memory-efficient implementation of Game of Life for very large grids.
    Instead of storing the full grid, it only tracks live cells and their neighbors.
    This is much more efficient for large, sparse grids (low density of live cells)."""
    
    def __init__(self, size=DEFAULT_SIZE, initial_density=DEFAULT_INITIAL_DENSITY, random_seed=None, device='cpu'):
        self.size = size
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        if random_seed is not None:
            np.random.seed(random_seed)  # Using numpy random for sparse initialization
        
        # Initialize with sparse representation
        # We'll use a dictionary to store live cells and their ages
        # Keys are (x, y) coordinates, values are ages
        self.live_cells = {}
        
        # Initialize with random live cells based on density
        num_cells = int(size * size * initial_density)
        for _ in range(num_cells):
            x, y = np.random.randint(0, size, 2)
            self.live_cells[(int(x), int(y))] = 0  # Age starts at 0
    
    def _get_neighbors(self, x, y):
        """Get the coordinates of all 8 neighbors with toroidal wrapping."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the cell itself
                nx = (x + dx) % self.size
                ny = (y + dy) % self.size
                neighbors.append((nx, ny))
        return neighbors
    
    def update(self):
        """Update the game state using the rules of Conway's Game of Life."""
        # Count all neighbors of live cells
        neighbor_counts = {}
        cells_to_check = set()
        
        # Count neighbors and collect all cells that need to be checked
        for cell in self.live_cells:
            x, y = cell
            # Add neighbors to the count
            for nx, ny in self._get_neighbors(x, y):
                neighbor_counts[(nx, ny)] = neighbor_counts.get((nx, ny), 0) + 1
                cells_to_check.add((nx, ny))
            # Also check the live cell itself
            cells_to_check.add(cell)
        
        # Apply the rules
        new_live_cells = {}
        for cell in cells_to_check:
            x, y = cell
            count = neighbor_counts.get(cell, 0)
            is_alive = cell in self.live_cells
            
            # Apply Game of Life rules
            if is_alive and (count == 2 or count == 3):
                # Cell stays alive
                new_live_cells[cell] = self.live_cells[cell] + 1  # Increment age
            elif not is_alive and count == 3:
                # Dead cell becomes alive
                new_live_cells[cell] = 0  # New cell starts with age 0
        
        # Update the state
        self.live_cells = new_live_cells
    
    def get_grid(self):
        """Convert sparse representation to dense grid for visualization."""
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        for (x, y) in self.live_cells:
            grid[x, y] = 1
        return grid
    
    def get_age_grid(self):
        """Convert sparse age representation to dense grid for visualization."""
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        for (x, y), age in self.live_cells.items():
            grid[x, y] = age
        return grid

def animate_game(size=DEFAULT_SIZE, interval=DEFAULT_INTERVAL, initial_density=DEFAULT_INITIAL_DENSITY, frame_skip=1, device='cuda', use_sparse=False):
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if size <= 0:
        raise ValueError("size must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")
    
    # Use sparse algorithm for very large grids or when explicitly requested
    if use_sparse or size > 2000:
        print("Using sparse algorithm for efficient large grid processing")
        game = SparseGameOfLife(size=size, initial_density=initial_density, random_seed=None, device=device)
    else:
        game = GameOfLife(size=size, initial_density=initial_density, random_seed=None, device=device)
    
    # Create a canvas and view
    canvas = vispy.scene.SceneCanvas(keys='interactive', size=(1920, 1080), resizable=True, show=True)
    canvas.native.setWindowState(Qt.WindowMaximized)
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = size * 1

    # Create text display for generation counter and status
    text = visuals.Text('Generation: 0\nLive Cells: 0\nPress SPACE to start', pos=(100, 50), color='white', font_size=12, parent=canvas.scene)
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

    # Initialize simulation state
    running = False
    generation = 0
    
    # Stability detection variables
    previous_cell_count = -1
    stable_generations = 0
    STABILITY_THRESHOLD = 5  # Number of consecutive generations with same cell count to consider stable

    def update(ev):
        nonlocal running, generation, previous_cell_count, stable_generations
        if not running:
            return
        
        # Check if simulation has reached stability
        if stable_generations >= STABILITY_THRESHOLD:
            running = False
            text.text = f'STABLE AFTER {generation} GENERATIONS\nLive Cells: {previous_cell_count}\nPress SPACE to restart'
            print(f"Simulation stabilized after {generation} generations with {previous_cell_count} cells")
            return
            
        # Update game state multiple times per frame if frame_skip > 1
        for _ in range(frame_skip):
            game.update()
            generation += 1
            
        grid = game.get_grid()
        age_grid = game.get_age_grid()
        
        # Count live cells
        live_cells = int(grid.sum())
        
        # Check for stability
        if live_cells == previous_cell_count:
            stable_generations += 1
        else:
            stable_generations = 0
            previous_cell_count = live_cells
        
        # Update generation counter and live cell count
        text.text = f'Generation: {generation}\nLive Cells: {live_cells}'
        
        # Get coordinates of live cells
        live_xs, live_ys = np.where(grid == 1)
        live_zs = np.full_like(live_xs, 0.1, dtype=float)
        
        # Get ages of live cells
        live_ages = age_grid[grid == 1]
        
        # Create positions array
        pos = np.column_stack((live_xs, live_ys, live_zs))
        
        # Create colors array using ColorArray
        colors = ColorArray([get_color(age) for age in live_ages])

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

    def on_key_press(event):
        nonlocal running, generation, previous_cell_count, stable_generations
        if event.key == ' ':
            if not running:
                # Reset stability detection if restarting
                stable_generations = 0
                previous_cell_count = -1
                
            running = not running
            if running:
                live_cells = int(game.get_grid().sum())
                text.text = f'Generation: {generation}\nLive Cells: {live_cells}'
            else:
                live_cells = int(game.get_grid().sum())
                text.text = f'Generation: {generation}\nLive Cells: {live_cells}\nPress SPACE to start'

    # Connect key press event
    canvas.events.key_press.connect(on_key_press)

    # Create timer
    timer = vispy.app.Timer(interval=interval/1000.0)  # Convert ms to seconds
    timer.connect(update)
    timer.start()

    # Run the app
    vispy.app.run()

def main():
    # Create Qt application
    app = QApplication([])
    
    # Debug CUDA information
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Show settings dialog
    settings = SettingsDialog()
    if settings.exec_() != QDialog.Accepted:
        return
    
    # Get settings values
    size = settings.size_spin.value()
    initial_density = settings.density_spin.value()
    interval = settings.interval_spin.value()
    frame_skip = settings.frame_skip_spin.value()
    device = settings.device_combo.currentText()
    
    # Determine if we should use sparse algorithm
    use_sparse = size > 2000
    
    # Run the animation with settings
    animate_game(
        size=size,
        interval=interval,
        initial_density=initial_density,
        frame_skip=frame_skip,
        device=device,
        use_sparse=use_sparse
    )

if __name__ == "__main__":
    main() 