import torch
import argparse
from PyQt5.QtWidgets import QApplication, QDialog
import vispy

from bblife.model import GameOfLife
from bblife.view import animate_game
from bblife.settings import SettingsDialog
from bblife.constants import (DEFAULT_SIZE, DEFAULT_INTERVAL, DEFAULT_INITIAL_DENSITY, 
                          DEFAULT_MUTATION_RATE, DEFAULT_FRAME_SKIP)

def print_cuda_info():
    """Print information about CUDA configuration."""
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Conway's Game of Life with VisPy Visualization")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Grid size (width and height)")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL, help="Update interval in milliseconds")
    parser.add_argument("--density", type=float, default=DEFAULT_INITIAL_DENSITY, help="Initial density of live cells (0.1 to 0.9)")
    parser.add_argument("--frame_skip", type=int, default=DEFAULT_FRAME_SKIP, help="Number of game updates per frame")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                       choices=['cuda', 'cpu'], help="Computation device ('cuda' or 'cpu')")
    parser.add_argument("--mutation_rate", type=float, default=DEFAULT_MUTATION_RATE, 
                       help="Base mutation rate, scaled by cell age (log)")
    parser.add_argument("--no_gui", action='store_true', 
                       help="Run simulation directly with command-line args, skipping GUI")
    
    args = parser.parse_args()
    
    # Print CUDA information
    print_cuda_info()
    
    # Create Qt application
    app = QApplication([])
    
    # Decide whether to use GUI or command-line args directly
    if args.no_gui:
        # Use command-line arguments directly
        size = args.size
        initial_density = args.density
        interval = args.interval
        frame_skip = args.frame_skip
        device_text = args.device # 'cuda' or 'cpu'
        mutation_rate = args.mutation_rate
        
        # Validate device selection again based on availability
        if device_text == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU.")
            device_text = 'cpu'
        
        print("\n--- Running with Command-Line Settings --- ")
        print(f"Size: {size}, Density: {initial_density:.2f}, Interval: {interval}ms, Frame Skip: {frame_skip}")
        print(f"Device: {device_text}, Mutation Rate: {mutation_rate:.4f}")
        print("----------------------------------------\n")
    else:
        # Show settings dialog, initializing with args
        settings = SettingsDialog()
        settings.size_spin.setValue(args.size)
        settings.density_spin.setValue(args.density)
        settings.interval_spin.setValue(args.interval)
        settings.frame_skip_spin.setValue(args.frame_skip)
        settings.mutation_rate_spin.setValue(args.mutation_rate)
        # Set device combo based on arg and availability
        if args.device == 'cuda' and torch.cuda.is_available():
            settings.device_combo.setCurrentIndex(settings.device_combo.findData('cuda'))
        else:
            settings.device_combo.setCurrentIndex(settings.device_combo.findData('cpu'))
            
        if settings.exec_() != QDialog.Accepted:
            return
        
        # Get settings values from dialog
        size = settings.size_spin.value()
        initial_density = settings.density_spin.value()
        interval = settings.interval_spin.value()
        frame_skip = settings.frame_skip_spin.value()
        device_text = settings.device_combo.currentData() # Get 'cuda' or 'cpu' from data
        mutation_rate = settings.mutation_rate_spin.value()
    
    # Initialize the game model
    game = GameOfLife(
        size=size, 
        initial_density=initial_density, 
        random_seed=None, 
        device=device_text, 
        mutation_rate=mutation_rate
    )
    
    # Run the animation with settings
    animate_game(
        game=game,
        size=size,
        interval=interval,
        frame_skip=frame_skip
    )

if __name__ == "__main__":
    main() 