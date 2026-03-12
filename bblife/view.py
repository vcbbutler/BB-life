import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy.color import ColorArray
from vispy.util import keys
from vispy.scene.cameras.perspective import PerspectiveCamera
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QAction, QDialog, QMenu, QMessageBox
import vispy.app

from .model import GameOfLife
from .settings import SettingsDialog
from .constants import (DEFAULT_INTERVAL, STABILITY_THRESHOLD, WINDOW_SIZE, 
                      MARKER_MIN_SIZE, MARKER_MAX_SIZE, MARKER_SCALE_FACTOR,
                      FLOOR_COLOR, AGE_YOUNG_THRESHOLD, AGE_MIDDLE_THRESHOLD, 
                      AGE_OLD_THRESHOLD, COLORS)


class PanningTurntableCamera(vispy.scene.cameras.TurntableCamera):
    """
    Turntable camera variant:
    - RMB drag pans (instead of zoom)
    - Scroll wheel keeps zoom behavior
    """

    def _translate_from_drag(self, p1, p2):
        norm = np.mean(self._viewbox.size)
        if self._event_value is None or len(self._event_value) == 2:
            self._event_value = self.center
        dist = (p1 - p2) / norm * self._scale_factor
        dist[1] *= -1
        dx, dy, dz = self._dist_to_trans(dist)
        ff = self._flip_factors
        up, forward, right = self._get_dim_vectors()
        dx, dy, dz = right * dx + forward * dy + up * dz
        dx, dy, dz = ff[0] * dx, ff[1] * dy, dz * ff[2]
        c = self._event_value
        self.center = c[0] + dx, c[1] + dy, c[2] + dz

    def viewbox_mouse_event(self, event):
        if event.handled or not self.interactive:
            return

        # Keep PerspectiveCamera wheel zoom behavior.
        PerspectiveCamera.viewbox_mouse_event(self, event)

        if event.type == "mouse_release":
            self._event_value = None
        elif event.type == "mouse_press":
            event.handled = True
        elif event.type == "mouse_move":
            if event.press_event is None:
                return
            if 1 in event.buttons and 2 in event.buttons:
                return

            modifiers = event.mouse_event.modifiers
            p1 = event.mouse_event.press_event.pos
            p2 = event.mouse_event.pos
            d = p2 - p1

            if 1 in event.buttons and not modifiers:
                self._update_rotation(event)
            elif 2 in event.buttons and not modifiers:
                # Remap RMB drag to translate/pan.
                self._translate_from_drag(p1, p2)
            elif 1 in event.buttons and keys.SHIFT in modifiers:
                self._translate_from_drag(p1, p2)
            elif 2 in event.buttons and keys.SHIFT in modifiers:
                if self._event_value is None:
                    self._event_value = self._fov
                fov = self._event_value - d[1] / 5.0
                self.fov = min(180.0, max(0.0, fov))

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
    view.camera = PanningTurntableCamera()
    view.camera.interactive = True
    view.camera.fov = 45
    view.camera.distance = size * 1

    # Create text display for generation counter and status
    text = visuals.Text(
        "Generation: 0\nLive Cells: 0\nPress SPACE to start",
        pos=(16, 16),
        color="white",
        font_size=12,
        anchor_x="left",
        anchor_y="top",
        parent=canvas.scene,
    )
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

    def build_status_text(live_cells, metrics, running_state):
        controls = "SPACE start/stop | R reset/settings | E menu | WASD pan | RMB drag pan"
        state_text = "Running" if running_state else "Paused"
        return (
            f"Generation: {generation}\n"
            f"Live Cells: {live_cells}\n"
            f"Rule: {metrics.get('rule', 'n/a')} ({state_text})\n"
            f"H_occ: {metrics.get('entropy_proxy', 0.0):.4f}\n"
            f"Var(pop): {metrics.get('population_variance', 0.0):.2f}\n"
            f"Period: {metrics.get('period_hint', 'none')}\n"
            f"{controls}"
        )

    def sync_menu_actions():
        start_action.setEnabled(not running)
        stop_action.setEnabled(running)

    def update_floor_visual(new_size):
        nonlocal floor
        floor.parent = None
        floor_vertices = np.array([
            [0, 0, 0],
            [new_size, 0, 0],
            [new_size, new_size, 0],
            [0, new_size, 0]
        ])
        floor_faces = np.array([[0, 1, 2], [0, 2, 3]])
        floor = visuals.Mesh(vertices=floor_vertices, faces=floor_faces, color=FLOOR_COLOR)
        view.add(floor)
        view.camera.set_range()
        view.camera.distance = new_size * 1.5

    def start_simulation():
        nonlocal running, previous_cell_count, stable_generations
        if running:
            return
        stable_generations = 0
        previous_cell_count = -1
        running = True
        live_cells = int(game.get_grid().sum())
        metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
        text.text = build_status_text(live_cells, metrics, running)
        sync_menu_actions()

    def stop_simulation():
        nonlocal running
        if not running:
            return
        running = False
        live_cells = int(game.get_grid().sum())
        metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
        text.text = build_status_text(live_cells, metrics, running)
        sync_menu_actions()

    def reset_simulation():
        nonlocal running, generation, previous_cell_count, stable_generations
        running = False
        generation = 0
        stable_generations = 0
        previous_cell_count = -1
        if hasattr(game, "reset"):
            game.reset()
        metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
        live_cells = int(game.get_grid().sum())
        text.text = build_status_text(live_cells, metrics, running)
        sync_menu_actions()

    def open_settings_dialog():
        nonlocal game, size, interval, frame_skip, generation, previous_cell_count, stable_generations
        was_running = running
        stop_simulation()

        settings = SettingsDialog(parent=canvas.native)
        settings.size_spin.setValue(size)
        settings.density_spin.setValue(float(getattr(game, "initial_density", 0.5)))
        settings.interval_spin.setValue(interval)
        settings.frame_skip_spin.setValue(frame_skip)
        settings.mutation_rate_spin.setValue(float(getattr(game, "mutation_rate", 0.001)))

        current_rule = getattr(game, "rule", "B3/S23")
        settings.set_rule_and_sync_preset(current_rule)

        current_device = getattr(game, "device", "cpu")
        device_index = settings.device_combo.findData(current_device)
        if device_index >= 0:
            settings.device_combo.setCurrentIndex(device_index)

        if settings.exec_() != QDialog.Accepted:
            if was_running:
                start_simulation()
            return

        new_size = settings.size_spin.value()
        new_density = settings.density_spin.value()
        new_interval = settings.interval_spin.value()
        new_frame_skip = settings.frame_skip_spin.value()
        new_mutation_rate = settings.mutation_rate_spin.value()
        new_rule = settings.get_rule_value()
        new_device = settings.device_combo.currentData() or current_device

        try:
            normalized_rule = GameOfLife._normalize_rule(new_rule)
        except ValueError as exc:
            QMessageBox.warning(canvas.native, "Invalid Rule", str(exc))
            if was_running:
                start_simulation()
            return

        rebuild_required = (
            new_size != size
            or float(new_density) != float(getattr(game, "initial_density", new_density))
            or new_device != getattr(game, "device", new_device)
            or float(new_mutation_rate) != float(getattr(game, "mutation_rate", new_mutation_rate))
            or normalized_rule != getattr(game, "rule", normalized_rule)
        )

        interval = new_interval
        frame_skip = new_frame_skip
        timer.stop()
        timer.interval = interval / 1000.0
        timer.start()

        if rebuild_required:
            try:
                game = GameOfLife(
                    size=new_size,
                    initial_density=new_density,
                    random_seed=getattr(game, "random_seed", None),
                    device=new_device,
                    mutation_rate=new_mutation_rate,
                    rule=normalized_rule
                )
            except Exception as exc:
                QMessageBox.warning(canvas.native, "Settings Error", str(exc))
                if was_running:
                    start_simulation()
                return

            size = new_size
            generation = 0
            stable_generations = 0
            previous_cell_count = -1
            update_floor_visual(size)

        live_cells = int(game.get_grid().sum())
        metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
        text.text = build_status_text(live_cells, metrics, running)
        sync_menu_actions()
        if was_running:
            start_simulation()

    def open_main_menu():
        sync_menu_actions()
        center_pos = canvas.native.rect().center()
        main_menu.exec_(canvas.native.mapToGlobal(center_pos))

    def pan_camera(dx=0.0, dy=0.0):
        center = np.array(view.camera.center, dtype=float)
        center[0] += dx
        center[1] += dy
        view.camera.center = tuple(center)

    def update(ev):
        nonlocal running, generation, previous_cell_count, stable_generations
        if not running:
            return
        
        # Check if simulation has reached stability
        if stable_generations >= STABILITY_THRESHOLD:
            running = False
            metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
            text.text = (
                f"STABLE AFTER {generation} GENERATIONS\n"
                f"Live Cells: {previous_cell_count}\n"
                f"Period: {metrics.get('period_hint', 'none')}\n"
                "Press SPACE to restart or R to reset"
            )
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
        metrics = game.get_metrics() if hasattr(game, "get_metrics") else {}
        
        # Check for stability
        if live_cells == previous_cell_count:
            stable_generations += 1
        else:
            stable_generations = 0
            previous_cell_count = live_cells
        
        # Update generation counter and live cell count
        text.text = build_status_text(live_cells, metrics, running)
        
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
        key_text = (getattr(event, "text", "") or "").lower()
        key_name = (getattr(getattr(event, "key", None), "name", "") or "").lower()
        pan_step = max(1.0, size * 0.03)

        if event.key == ' ' or key_text == ' ' or key_name == 'space':
            if running:
                stop_simulation()
            else:
                start_simulation()
        elif key_text == 'r' or key_name == 'r':
            open_settings_dialog()
        elif key_text == 'e' or key_name == 'e':
            open_main_menu()
        elif key_text == 'a' or key_name == 'a':
            pan_camera(dx=-pan_step)
        elif key_text == 'd' or key_name == 'd':
            pan_camera(dx=pan_step)
        elif key_text == 'w' or key_name == 'w':
            pan_camera(dy=pan_step)
        elif key_text == 's' or key_name == 's':
            pan_camera(dy=-pan_step)

    # Connect key press event
    canvas.events.key_press.connect(on_key_press)

    # Add a lightweight main menu via right-click context menu.
    main_menu = QMenu("Main", canvas.native)
    start_action = QAction("Start", canvas.native)
    stop_action = QAction("Stop", canvas.native)
    reset_action = QAction("Reset", canvas.native)
    settings_action = QAction("Settings...", canvas.native)
    start_action.triggered.connect(start_simulation)
    stop_action.triggered.connect(stop_simulation)
    reset_action.triggered.connect(reset_simulation)
    settings_action.triggered.connect(open_settings_dialog)
    main_menu.addAction(start_action)
    main_menu.addAction(stop_action)
    main_menu.addAction(reset_action)
    main_menu.addSeparator()
    main_menu.addAction(settings_action)

    sync_menu_actions()

    # Create timer
    timer = vispy.app.Timer(interval=interval/1000.0)  # Convert ms to seconds
    timer.connect(update)
    timer.start()

    # Run the app
    vispy.app.run() 