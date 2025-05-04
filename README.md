# CUDA-Accelerated Game of Life

This is a Python implementation of Conway's Game of Life that uses CUDA acceleration through PyTorch for better performance.

## Requirements

- Python 3.7+
- CUDA-capable GPU (RTX 4090 in this case)
- PyTorch with CUDA support
- NumPy
- Matplotlib

## Installation

1. Create a new conda environment (optional but recommended):
```bash
conda create -n gameoflife python=3.9
conda activate gameoflife
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Game

Simply run the Python script:
```bash
python game_of_life.py
```

## Features

- GPU-accelerated computation using PyTorch
- Real-time visualization using Matplotlib
- Configurable grid size and animation parameters
- Efficient neighbor counting using convolution operations

## Customization

You can modify the following parameters in the `game_of_life.py` file:
- Grid size (default: 100x100)
- Number of frames (default: 100)
- Animation interval in milliseconds (default: 100)
- Random seed for initial state (default: 42) 