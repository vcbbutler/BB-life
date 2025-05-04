import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy.color import ColorArray
from PyQt5.QtCore import Qt
import vispy.app

from .constants import (DEFAULT_INTERVAL, STABILITY_THRESHOLD, WINDOW_SIZE, 
                      MARKER_MIN_SIZE, MARKER_MAX_SIZE, MARKER_SCALE_FACTOR,
                      FLOOR_COLOR, AGE_YOUNG_THRESHOLD, AGE_MIDDLE_THRESHOLD, 
                      AGE_OLD_THRESHOLD, COLORS)

def animate_game(game, size, interval=DEFAULT_INTERVAL, frame_skip=1):
    """
    Create and manage the 3D visualization of the Game of Life simulation.
    
    Args:
        game: An instance of GameOfLife or SparseGameOfLife
        size: Grid size
        interval: Update interval in milliseconds
        frame_skip: Number of game updates per frame
    """
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")
    if size <= 0:
        raise ValueError("size must be positive")
    if interval <= 0:
        raise ValueError("interval must be positive")
    
    # Create a canvas and view
    canvas = vispy.scene.SceneCanvas(keys='interactive', size=WINDOW_SIZE, resizable=True, show=True)
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
    floor = visuals.Mesh(vertices=floor_vertices, faces=floor_faces, color=FLOOR_COLOR)
    view.add(floor)

    # Set up the view
    view.camera.set_range()
    view.camera.elevation = 20
    view.camera.azimuth = -45
    view.camera.distance = size * 1.5

    # Create color map for ages
    def get_color(age):
        # Use a non-linear scale: very quick initial phase, longer taper
        if age < AGE_YOUNG_THRESHOLD:  # Quick initial phase (0-20)
            normalized_age = age / AGE_YOUNG_THRESHOLD
            if normalized_age < 0.1:
                return COLORS['very_young']
            elif normalized_age < 0.3:
                return COLORS['young']
            elif normalized_age < 0.6:
                return COLORS['young_adult']
            elif normalized_age < 0.8:
                return COLORS['adult']
            else:
                return COLORS['mature']
        elif age < AGE_MIDDLE_THRESHOLD:  # Middle phase (20-50)
            normalized_age = (age - AGE_YOUNG_THRESHOLD) / (AGE_MIDDLE_THRESHOLD - AGE_YOUNG_THRESHOLD)
            if normalized_age < 0.5:
                return COLORS['middle_aged']
            else:
                return COLORS['older']
        elif age < AGE_OLD_THRESHOLD:
            return COLORS['old']
        else:
            return COLORS['ancient']

    # Initialize simulation state
    running = False
    generation = 0
    
    # Stability detection variables
    previous_cell_count = -1
    stable_generations = 0

    def update(ev):
        nonlocal running, generation, previous_cell_count, stable_generations
        if not running:
            return
        
        # Check if simulation has reached stability
        if stable_generations >= STABILITY_THRESHOLD:
            running = False
            text.text = f'STABLE AFTER {generation} GENERATIONS\nLive Cells: {previous_cell_count}\nPress SPACE to restart'
            print(f"Simulation stabilized after {generation - STABILITY_THRESHOLD} generations with {previous_cell_count} cells")
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
            sizes = np.clip(MARKER_SCALE_FACTOR / (distances + 1), MARKER_MIN_SIZE, MARKER_MAX_SIZE)
        else:
            sizes = MARKER_MIN_SIZE
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