"""Configuration for the preprocessing pipeline."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PreprocessConfig:
    mode: Literal["passthrough", "best_frame", "enhance", "fusion"] = "passthrough"
    max_frames: int = 5
    time_budget_ms: float = 100.0
    gamma: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)
    fusion_method: Literal["mean", "median"] = "mean"
    ecc_min_correlation: float = 0.8
