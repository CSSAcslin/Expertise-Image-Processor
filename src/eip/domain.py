from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import numpy as np


@dataclass(frozen=True)
class SourceSpec:
    path: str
    kind: str = "auto"
    interval: float | None = None

    def __post_init__(self):
        if self.kind not in {"auto", "image", "sequence", "video", "correlation"}:
            raise ValueError("不支持的数据源类型")
        if self.interval is not None and (not math.isfinite(self.interval) or self.interval <= 0):
            raise ValueError("序列时间间隔必须为正数")

    @property
    def key(self) -> str:
        value = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(value.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class DataSpec:
    name: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    frame_count: int | None
    duration: float | None
    rate: float | None
    timestamp_source: str
    color: str
    revision: str
    notes: tuple[str, ...] = ()
    frame_indexed: bool = False
    sample_bits: int | None = None


@dataclass(frozen=True)
class FrameRef:
    index: int | None
    time: float | None
    pts: int | None = None
    source_number: int | None = None
    trigger_time: float | None = None


@dataclass
class FrameData:
    pixels: np.ndarray
    ref: FrameRef


@dataclass(frozen=True)
class ROI:
    x: int
    y: int
    width: int
    height: int

    def validate(self, shape: tuple[int, ...]):
        h, w = shape[:2]
        if any(type(value) is not int for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("ROI 坐标和尺寸必须是整数像素")
        if min(self.x, self.y) < 0 or min(self.width, self.height) < 1:
            raise ValueError("ROI 必须是非空的图像区域")
        if self.x + self.width > w or self.y + self.height > h:
            raise ValueError("ROI 超出当前数据尺寸，请重新选择")

    def crop(self, array: np.ndarray) -> np.ndarray:
        self.validate(array.shape)
        return array[self.y:self.y + self.height, self.x:self.x + self.width].copy()


@dataclass(frozen=True)
class Selection:
    mode: str = "all"
    start: float = 0
    end: float | None = None
    current: FrameRef | None = None
    roi: ROI | None = None

    def __post_init__(self):
        if self.mode not in {"all", "range", "current"}:
            raise ValueError("未知处理范围")
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("处理起点不能为负数")
        if self.mode == "current" and self.current is None:
            raise ValueError("尚未定位当前帧")
        if self.mode == "range" and (self.end is None or not math.isfinite(self.end) or self.end <= self.start):
            raise ValueError("处理终点必须大于起点（终点不包含在范围中）")


@dataclass(frozen=True)
class StepSpec:
    algorithm: str
    parameters: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    manifest: str | None = None
    code_hash: str | None = None
    version: str = "1.0"


def validate_pixels(array: np.ndarray) -> None:
    if not isinstance(array, np.ndarray) or array.ndim not in {2, 3}:
        raise ValueError("当前框架支持 YX 或 YXS 图像；多维数据需要读取适配器")
    if array.ndim == 3 and array.shape[2] not in {3, 4}:
        raise ValueError("三维数组必须明确为 RGB/RGBA，不能隐式当作科研通道")
    if array.dtype.kind not in "bufi" or min(array.shape[:2]) < 1:
        raise ValueError("图像必须包含非空的实数像素")


def file_revision(paths: list[Path]) -> str:
    """Fast stat fingerprint, deliberately not a content hash."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def source_from_dict(value: dict) -> SourceSpec:
    return SourceSpec(**value)


def selection_from_dict(value: dict) -> Selection:
    value = dict(value)
    if value.get("roi") is not None:
        value["roi"] = ROI(**value["roi"])
    if value.get("current") is not None:
        value["current"] = FrameRef(**value["current"])
    return Selection(**value)
