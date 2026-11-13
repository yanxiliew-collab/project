"""Data contracts shared by preprocessing components."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class TrackFrame:
    """A frame and the tracked region of interest within it."""

    frame_idx: int
    timestamp: float
    full_frame: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class PreprocessResult:
    """The preprocessed image and processing diagnostics."""

    image: Optional[np.ndarray]
    diagnostics: dict[str, Any]
