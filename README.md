# 3D Game of Life Simulation

A GPU-accelerated 3D visualization of Conway's Game of Life using PyTorch and Matplotlib. The simulation features cell aging with color transitions and interactive 3D visualization.

## Features

- GPU-accelerated computation using PyTorch
- 3D visualization with rotating view
- Cell aging system with color transitions
- Customizable random seeding
- Interactive visualization with Matplotlib

## Requirements

- Python 3.x
- PyTorch (with CUDA support recommended)
- NumPy
- Matplotlib

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install torch numpy matplotlib
```

## Usage

Run the simulation with different seed options:

1. Time-based seed (default):
```bash
python game_of_life.py
```

2. No seed (completely random):
```bash
python game_of_life.py --seed none
```

3. Custom seed value:
```bash
python game_of_life.py --seed 42
```

## Visualization Details

- Cells are represented as spheres in 3D space
- Colors change as cells age (green → yellow → red → brown)
- The view rotates automatically to provide a 3D perspective
- Simulation runs for 200 frames with 50ms intervals
- Grid size is 100x100 by default

## Controls

- Close the visualization window to stop the simulation
- The simulation will automatically rotate to show the 3D perspective

## Notes

- If CUDA is not available, the simulation will run on CPU
- The seed value is printed to the console, allowing you to recreate interesting patterns
- Cell colors transition based on their age, providing visual feedback on cell longevity 