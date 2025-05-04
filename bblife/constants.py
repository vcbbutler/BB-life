# Default simulation parameters
DEFAULT_SIZE = 256
DEFAULT_INTERVAL = 30
DEFAULT_INITIAL_DENSITY = 0.5
DEFAULT_MUTATION_RATE = 0.001
DEFAULT_FRAME_SKIP = 1

# Stability detection
STABILITY_THRESHOLD = 50

# Visualization settings
WINDOW_SIZE = (1920, 1080)
MARKER_MIN_SIZE = 4
MARKER_MAX_SIZE = 16
MARKER_SCALE_FACTOR = 6000
FLOOR_COLOR = (0.5, 0.5, 0.5, 0.2)

# Age-based color mapping thresholds
AGE_YOUNG_THRESHOLD = 20
AGE_MIDDLE_THRESHOLD = 50
AGE_OLD_THRESHOLD = 100

# Color definitions for cells by age
COLORS = {
    'very_young': (0.0, 0.8, 0, 0.9),      # Green
    'young': (0.6, 0.8, 0, 0.9),           # Yellow-green
    'young_adult': (0.8, 0.8, 0, 0.9),     # Yellow
    'adult': (1.0, 0.6, 0, 0.9),           # Orange-yellow
    'mature': (1.0, 0.4, 0, 0.9),          # Orange
    'middle_aged': (1.0, 0.3, 0, 0.9),     # Red-orange
    'older': (0.8, 0.2, 0, 0.9),           # Brown
    'old': (0.4, 0.1, 0, 0.9),             # Dark brown
    'ancient': (0.2, 0.05, 0, 0.9)         # Very dark brown
} 