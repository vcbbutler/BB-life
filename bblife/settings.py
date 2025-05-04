import torch
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, 
                           QLabel, QSpinBox, QDoubleSpinBox, QPushButton, 
                           QComboBox, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt
from .constants import (DEFAULT_SIZE, DEFAULT_INTERVAL, DEFAULT_INITIAL_DENSITY, 
                      DEFAULT_MUTATION_RATE, DEFAULT_FRAME_SKIP)

class PowerOfTwoSpinBox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRange(8, 1000000)  # Allow very large sizes, minimum 8
        self.setValue(256)  # Default to 256
    
    def stepBy(self, steps):
        # Double or halve the value when stepping
        if steps > 0:
            # Going up, double the value
            for _ in range(steps):
                self.setValue(self.value() * 2)
        else:
            # Going down, halve the value
            for _ in range(-steps):
                self.setValue(max(8, self.value() // 2))
                
    def validate(self, text, pos):
        # Allow any integer input
        return (QSpinBox.validate(self, text, pos))

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
        
        self.size_spin = PowerOfTwoSpinBox()
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
        self.frame_skip_spin.setValue(DEFAULT_FRAME_SKIP)
        anim_layout.addRow("Frame Skip:", self.frame_skip_spin)
        
        # --- Mutation Rate ---
        self.mutation_rate_spin = QDoubleSpinBox()
        self.mutation_rate_spin.setRange(0.0, 0.1)
        self.mutation_rate_spin.setValue(DEFAULT_MUTATION_RATE)
        self.mutation_rate_spin.setSingleStep(0.001)
        self.mutation_rate_spin.setDecimals(4)
        anim_layout.addRow("Mutation Rate:", self.mutation_rate_spin)
        
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