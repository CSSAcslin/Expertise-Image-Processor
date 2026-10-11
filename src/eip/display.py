from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DisplaySettings:
    low: float | None = None
    high: float | None = None
    gamma: float = 1.0
    colormap: str = "gray"


def display_range(array: np.ndarray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if not finite.size:
        return 0.0, 1.0
    # Bound the work needed for automatic display, without changing data.
    samples = finite[::max(1, finite.size // 200_000)]
    low, high = np.percentile(samples, [0.5, 99.5])
    if high <= low:
        high = low + 1
    return float(low), float(high)


def to_display(array: np.ndarray, settings: DisplaySettings) -> np.ndarray:
    low, high = display_range(array) if settings.low is None or settings.high is None else (settings.low, settings.high)
    if not np.isfinite([low, high, settings.gamma]).all() or high <= low or settings.gamma <= 0:
        raise ValueError("显示上限必须大于下限，伽马必须为正数")
    value = np.asarray(array, dtype=np.float32)
    value = np.clip((value - low) / (high - low), 0, 1)
    value = np.nan_to_num(value, nan=0, posinf=1, neginf=0)
    value = np.power(value, 1 / settings.gamma)
    if value.ndim == 3:
        return np.ascontiguousarray(np.rint(value[..., :3] * 255).astype(np.uint8))
    if settings.colormap == "heat":
        rgb = np.stack([np.clip(value * 3, 0, 1), np.clip(value * 3 - 1, 0, 1), np.clip(value * 3 - 2, 0, 1)], axis=-1)
    else:
        rgb = np.repeat(value[..., None], 3, axis=-1)
    return np.ascontiguousarray(np.rint(rgb * 255).astype(np.uint8))


def labels_to_display(array: np.ndarray) -> np.ndarray:
    """Discrete palette; IDs remain intact and do not pass through intensity LUTs."""
    invalid = ~np.isfinite(array)
    labels = np.where(invalid,0,array).astype(np.uint64)
    channels = [((labels * factor + offset) % 191 + 64).astype(np.uint8) for factor, offset in [(53, 7), (97, 31), (131, 67)]]
    rgb = np.stack(channels, axis=-1)
    rgb[labels == 0] = 0
    rgb[invalid] = [128,64,148]
    return np.ascontiguousarray(rgb)
