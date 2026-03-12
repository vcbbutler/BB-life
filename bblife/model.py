import math
import re
from collections import deque
from typing import Dict, Optional, Set, Tuple, Union

import numpy as np
import torch

from .constants import (
    DEFAULT_INITIAL_DENSITY,
    DEFAULT_MUTATION_RATE,
    DEFAULT_RULE,
    DEFAULT_SIZE,
    METRICS_WINDOW,
)


class GameOfLife:
    def __init__(
        self,
        size: int = DEFAULT_SIZE,
        initial_density: float = DEFAULT_INITIAL_DENSITY,
        random_seed: Optional[int] = None,
        device: str = "cuda",
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        rule: str = DEFAULT_RULE,
    ):
        self.size = size
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.initial_density = initial_density
        self.random_seed = random_seed
        self.mutation_rate = mutation_rate
        self.rule = self._normalize_rule(rule)

        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Initialize the grid and age grid.
        self.grid = torch.bernoulli(torch.full((size, size), initial_density, device=self.device))
        self.age_grid = torch.zeros((size, size), dtype=torch.float32, device=self.device)

        # Initialize history grids (t-2, t-3, t-4).
        self.grid_t_minus_2 = torch.zeros_like(self.grid, device=self.device)
        self.grid_t_minus_3 = torch.zeros_like(self.grid, device=self.device)
        self.grid_t_minus_4 = torch.zeros_like(self.grid, device=self.device)

        # Create convolution kernel for counting neighbors.
        self.kernel = torch.tensor(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 1, 3, 3)

        # Neighbor offsets are reused during mutation.
        self.neighbor_offsets = torch.tensor(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ],
            dtype=torch.long,
            device=self.device,
        )

        birth_set, survive_set = self._parse_rule(self.rule)
        self.birth_mask = self._build_rule_mask(birth_set)
        self.survive_mask = self._build_rule_mask(survive_set)

        self.generation = 0
        self.population_history = deque(maxlen=METRICS_WINDOW)
        self.metrics = {
            "rule": self.rule,
            "live_count": 0,
            "live_fraction": 0.0,
            "population_variance": 0.0,
            "entropy_proxy": 0.0,
            "period_hint": "none",
            "generation": 0,
        }

    def reset(self) -> None:
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)

        self.grid = torch.bernoulli(
            torch.full((self.size, self.size), self.initial_density, device=self.device)
        )
        self.age_grid = torch.zeros((self.size, self.size), dtype=torch.float32, device=self.device)
        self.grid_t_minus_2 = torch.zeros_like(self.grid, device=self.device)
        self.grid_t_minus_3 = torch.zeros_like(self.grid, device=self.device)
        self.grid_t_minus_4 = torch.zeros_like(self.grid, device=self.device)
        self.generation = 0
        self.population_history.clear()
        self.metrics = {
            "rule": self.rule,
            "live_count": int(self.grid.sum().item()),
            "live_fraction": float(self.grid.float().mean().item()),
            "population_variance": 0.0,
            "entropy_proxy": 0.0,
            "period_hint": "none",
            "generation": 0,
        }

    @staticmethod
    def _parse_rule(rule: str) -> Tuple[Set[int], Set[int]]:
        match = re.fullmatch(r"B([0-8]*)/S([0-8]*)", rule.upper().strip())
        if not match:
            raise ValueError(f"Invalid rule '{rule}'. Expected format like B3/S23.")

        birth_set = {int(char) for char in match.group(1)}
        survive_set = {int(char) for char in match.group(2)}
        return birth_set, survive_set

    @classmethod
    def _normalize_rule(cls, rule: str) -> str:
        birth_set, survive_set = cls._parse_rule(rule)
        birth_digits = "".join(str(value) for value in sorted(birth_set))
        survive_digits = "".join(str(value) for value in sorted(survive_set))
        return f"B{birth_digits}/S{survive_digits}"

    def _build_rule_mask(self, active_counts: Set[int]) -> torch.Tensor:
        mask = torch.zeros(9, dtype=torch.bool, device=self.device)
        if active_counts:
            indices = torch.tensor(sorted(active_counts), dtype=torch.long, device=self.device)
            mask[indices] = True
        return mask

    def update(self):
        # Store the grid state from t-1 before it is updated.
        grid_t_minus_1 = self.grid.clone()
        # Keep the pre-shift t-4 state for period-4 detection.
        grid_t_minus_4_snapshot = self.grid_t_minus_4.clone()

        # Shift history grids forward (t-4 <- t-3 <- t-2 <- t-1).
        self.grid_t_minus_4 = self.grid_t_minus_3
        self.grid_t_minus_3 = self.grid_t_minus_2
        self.grid_t_minus_2 = grid_t_minus_1

        # Pad the grid (state at t-1) with circular boundaries.
        padded_grid = torch.nn.functional.pad(
            grid_t_minus_1.float().unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1),
            mode="circular",
        )

        # Count neighbors based on t-1 state.
        neighbors = torch.nn.functional.conv2d(
            padded_grid,
            self.kernel,
            padding=0,
        ).squeeze()
        neighbor_counts = neighbors.to(torch.long)

        is_alive_t_minus_1 = grid_t_minus_1 == 1.0
        is_dead_t_minus_1 = ~is_alive_t_minus_1

        survives = is_alive_t_minus_1 & self.survive_mask[neighbor_counts]
        births = is_dead_t_minus_1 & self.birth_mask[neighbor_counts]

        # Calculate potential new grid state at time t (before mutations).
        new_grid_potential = torch.zeros_like(self.grid)
        new_grid_potential[survives | births] = 1.0

        # Identify mutation triggers based on age of potentially live cells.
        is_alive_potential = new_grid_potential == 1.0
        mutation_prob_map = torch.clamp(self.mutation_rate * torch.log1p(self.age_grid), 0.0, 1.0)
        active_triggers = is_alive_potential & (
            torch.rand_like(new_grid_potential, device=self.device) < mutation_prob_map
        )

        new_grid = new_grid_potential.clone()

        # Determine mutation targets (self or neighbor) and apply flips.
        if torch.any(active_triggers):
            trigger_coords = active_triggers.nonzero(as_tuple=False)
            cells_to_flip = torch.zeros_like(new_grid, dtype=torch.bool, device=self.device)

            target_decisions = torch.rand(len(trigger_coords), device=self.device) < 0.5

            self_mutation_coords = trigger_coords[target_decisions]
            if len(self_mutation_coords) > 0:
                cells_to_flip[self_mutation_coords[:, 0], self_mutation_coords[:, 1]] = True

            neighbor_mutation_triggers = trigger_coords[~target_decisions]
            if len(neighbor_mutation_triggers) > 0:
                offset_indices = torch.randint(0, 8, (len(neighbor_mutation_triggers),), device=self.device)
                offsets = self.neighbor_offsets[offset_indices]

                neighbor_coords_r = (neighbor_mutation_triggers[:, 0] + offsets[:, 0]) % self.size
                neighbor_coords_c = (neighbor_mutation_triggers[:, 1] + offsets[:, 1]) % self.size
                cells_to_flip[neighbor_coords_r, neighbor_coords_c] = True

            new_grid[cells_to_flip] = 1.0 - new_grid[cells_to_flip]

        # Update age grid based on the final mutated new grid.
        self.age_grid = torch.where(
            new_grid == 1.0,
            self.age_grid + 1,
            torch.tensor(0.0, device=self.device),
        )

        # Update the main grid state.
        self.grid = new_grid
        self.generation += 1
        self._update_metrics(grid_t_minus_4_snapshot)

    def _update_metrics(self, grid_t_minus_4_snapshot: torch.Tensor) -> None:
        live_count = int(self.grid.sum().item())
        total_cells = self.size * self.size
        live_fraction = live_count / total_cells

        self.population_history.append(live_count)
        if len(self.population_history) > 1:
            population_variance = float(np.var(np.array(self.population_history, dtype=np.float32)))
        else:
            population_variance = 0.0

        if live_fraction <= 0.0 or live_fraction >= 1.0:
            entropy_proxy = 0.0
        else:
            entropy_proxy = -(
                live_fraction * math.log2(live_fraction)
                + (1.0 - live_fraction) * math.log2(1.0 - live_fraction)
            )

        period_hint = "none"
        if self.generation >= 2 and torch.equal(self.grid, self.grid_t_minus_3):
            period_hint = "p2"
        elif self.generation >= 3 and torch.equal(self.grid, self.grid_t_minus_4):
            period_hint = "p3"
        elif self.generation >= 4 and torch.equal(self.grid, grid_t_minus_4_snapshot):
            period_hint = "p4"

        self.metrics = {
            "rule": self.rule,
            "live_count": live_count,
            "live_fraction": live_fraction,
            "population_variance": population_variance,
            "entropy_proxy": entropy_proxy,
            "period_hint": period_hint,
            "generation": self.generation,
        }

    def get_grid(self):
        return self.grid.cpu().numpy()

    def get_age_grid(self):
        return self.age_grid.cpu().numpy()

    def get_metrics(self) -> Dict[str, Union[str, int, float]]:
        return dict(self.metrics)