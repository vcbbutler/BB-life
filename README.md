# Game of Life 3D Visualization

A high-performance 3D visualization of Conway's Game of Life using PyTorch for computation and VisPy for rendering. The simulation runs on GPU for optimal performance.

## Features

- GPU-accelerated Game of Life simulation using PyTorch
- Real-time 3D visualization with VisPy
- Interactive camera controls (rotate, zoom, pan)
- Age-based cell coloring
- Configurable simulation speed and frame skip
- Customizable random seeding

## Requirements

- Python 3.8+
- CUDA-capable GPU (NVIDIA)
- PyTorch with CUDA support
- VisPy
- PyQt6

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vcbbutler/BB-life
cd BB-life
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the simulation with default settings:
```bash
python game_of_life.py
```

### Command Line Arguments

- `--seed`: Seed type for random initialization
  - `none`: No seed (completely random)
  - `time`: Time-based seed (default)
  - `<number>`: Custom seed value
- `--interval`: Animation interval in milliseconds (default: 50)
- `--frame-skip`: Number of game updates per frame (default: 1)

Examples:
```bash
# Fast simulation with smooth visualization
python game_of_life.py --interval 50 --frame-skip 2

# Custom seed with slower updates
python game_of_life.py --seed 42 --interval 100

# Very fast simulation
python game_of_life.py --interval 50 --frame-skip 3
```

## Controls

- **Mouse Left Button**: Rotate view
- **Mouse Right Button**: Pan view
- **Mouse Wheel**: Zoom in/out
- **Space**: Pause/Resume animation

## Performance Tips

- Increase `frame-skip` for faster simulation
- Adjust `interval` to control visualization smoothness
- Larger values for both parameters will result in faster simulation but less smooth visualization
- The simulation automatically uses GPU acceleration if available

## License

[Your chosen license] 