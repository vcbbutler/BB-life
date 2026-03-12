import re
import torch
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)
from .constants import (
    DEFAULT_INTERVAL,
    DEFAULT_INITIAL_DENSITY,
    DEFAULT_MUTATION_RATE,
    DEFAULT_FRAME_SKIP,
    DEFAULT_RULE,
    RULE_PRESETS,
)

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
    CUSTOM_PRESET_NAME = "Custom"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Game of Life Settings")
        self._preset_rules_by_name = dict(RULE_PRESETS)
        self._normalized_rule_to_preset_name = {
            self._normalize_rule(rule): name
            for name, rule in self._preset_rules_by_name.items()
        }
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

        self.rule_preset_combo = QComboBox()
        for preset_name, rule_value in RULE_PRESETS.items():
            self.rule_preset_combo.addItem(preset_name, rule_value)
        self.rule_preset_combo.addItem(self.CUSTOM_PRESET_NAME, self.CUSTOM_PRESET_NAME)
        self.rule_preset_combo.setToolTip(
            "Selecting a preset fills the editable rule below."
        )
        anim_layout.addRow("Rule Preset:", self.rule_preset_combo)

        self.rule_input = QLineEdit()
        self.rule_input.setText(DEFAULT_RULE)
        self.rule_input.setToolTip(
            "Editable B/S rule. Editing this field switches preset to Custom when unmatched."
        )
        anim_layout.addRow("Rule (B/S, editable):", self.rule_input)

        default_preset_index = self.rule_preset_combo.findText("Conway Life")
        if default_preset_index < 0:
            preset_name = self._normalized_rule_to_preset_name.get(
                self._normalize_rule(DEFAULT_RULE)
            )
            default_preset_index = self.rule_preset_combo.findText(preset_name) if preset_name else 0
        self.rule_preset_combo.setCurrentIndex(default_preset_index)
        self._on_rule_preset_changed(default_preset_index)

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
        self.rule_preset_combo.currentIndexChanged.connect(self._on_rule_preset_changed)
        self.rule_input.textEdited.connect(self._on_rule_text_edited)

    @staticmethod
    def _normalize_rule(rule: str) -> str:
        match = re.fullmatch(r"B([0-8]*)/S([0-8]*)", rule.upper().strip())
        if not match:
            raise ValueError("Invalid rule")

        birth_digits = "".join(str(value) for value in sorted({int(char) for char in match.group(1)}))
        survive_digits = "".join(str(value) for value in sorted({int(char) for char in match.group(2)}))
        return f"B{birth_digits}/S{survive_digits}"

    def _set_custom_preset(self) -> None:
        custom_index = self.rule_preset_combo.findData(self.CUSTOM_PRESET_NAME)
        if custom_index >= 0 and custom_index != self.rule_preset_combo.currentIndex():
            self.rule_preset_combo.setCurrentIndex(custom_index)

    def _on_rule_preset_changed(self, _index: int) -> None:
        selected_rule = self.rule_preset_combo.currentData()
        if selected_rule == self.CUSTOM_PRESET_NAME:
            return
        self.rule_input.setText(selected_rule)

    def _on_rule_text_edited(self, text: str) -> None:
        self.sync_preset_from_rule(text)

    def sync_preset_from_rule(self, rule_value: str) -> None:
        try:
            normalized_rule = self._normalize_rule(rule_value)
        except ValueError:
            self._set_custom_preset()
            return

        matching_preset_name = self._normalized_rule_to_preset_name.get(normalized_rule)
        if not matching_preset_name:
            self._set_custom_preset()
            return

        preset_index = self.rule_preset_combo.findText(matching_preset_name)
        if preset_index >= 0 and preset_index != self.rule_preset_combo.currentIndex():
            self.rule_preset_combo.setCurrentIndex(preset_index)

    def set_rule_and_sync_preset(self, rule_value: str) -> None:
        self.rule_input.setText(rule_value)
        self.sync_preset_from_rule(rule_value)

    def get_rule_value(self) -> str:
        return self.rule_input.text().strip()