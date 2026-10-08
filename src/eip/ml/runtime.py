from dataclasses import dataclass
from typing import Protocol
import numpy as np

from eip.algorithms import RunContext


@dataclass(frozen=True)
class ModelSpec:
    path: str
    input_axes: str
    dtype: str
    channel_order: str
    normalization: str
    output_kind: str
    sha256: str


class ModelRuntime(Protocol):
    """Extension contract only; this release does not claim a model implementation."""
    def load(self, model: ModelSpec, device: str, context: RunContext) -> None: ...
    def predict(self, batch: np.ndarray, context: RunContext) -> np.ndarray: ...
    def close(self) -> None: ...
