from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import math
import re
import csv
import json

import numpy as np
from PIL import Image
import tifffile

from eip.domain import DataSpec, FrameData, FrameRef, Selection, SourceSpec, file_revision, validate_pixels

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".cine"}


def natural_key(path: Path):
    return [(0, int(part)) if part.isdigit() else (1, part.lower()) for part in re.split(r"(\d+)", path.name)]


def read_image(path: Path, page: int = 0) -> np.ndarray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        with tifffile.TiffFile(path) as file:
            pixels = file.pages[page].asarray()
    else:
        with Image.open(path) as image:
            if image.mode in {"P", "CMYK", "YCbCr", "LA"}:
                image = image.convert("RGB")
            pixels = np.array(image)
    validate_pixels(pixels)
    pixels.setflags(write=False)
    return pixels


class ImageReader:
    def __init__(self, source: SourceSpec):
        self.source = source
        self.path = Path(source.path)
        self.files = sorted((p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key) if source.kind == "sequence" else [self.path]
        if not self.files:
            raise ValueError("目录没有支持的图像文件")
        self.pages = 1
        if source.kind != "sequence" and self.path.suffix.lower() in {".tif", ".tiff"}:
            with tifffile.TiffFile(self.path) as file:
                if len(file.series) != 1:
                    raise ValueError("当前版本不支持多 series TIFF，请导出明确的图像序列")
                self.pages = len(file.pages)
                axes = file.series[0].axes
                if axes not in {"YX", "YXS", "IYX", "IYXS", "QYX", "QYXS", "TYX", "TYXS"}:
                    raise ValueError(f"TIFF 轴为 {axes}；当前版本仅支持二维图像及明确的页序列")
        self.count = len(self.files) if source.kind == "sequence" else self.pages
        first = read_image(self.files[0])
        self._budget = 128 * 1024 * 1024
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict([(0, first)] if first.nbytes <= self._budget else [])
        self._cache_size = first.nbytes if first.nbytes <= self._budget else 0
        interval = source.interval
        duration = self.count * interval if interval is not None else None
        self.timestamps = None
        self.source_numbers = self.trigger_times = None
        notes = ["序列采用文件名自然排序；时间间隔为用户指定值" if interval is not None else "目录采用自然排序；无确认的时间间隔"] if source.kind == "sequence" else []
        fingerprint_files = list(self.files)
        if source.kind == "sequence" and (self.path / "measurements.csv").is_file() and (self.path.parent / "run.json").is_file():
            run = json.loads((self.path.parent / "run.json").read_text(encoding="utf-8"))
            item = next((value for value in run.get("results", []) if value.get("output_directory") == self.path.name), None)
            if item and item.get("status") == "succeeded" and item.get("completed_frames") == self.count:
                with (self.path / "measurements.csv").open(encoding="utf-8-sig", newline="") as file:
                    rows = list(csv.DictReader(file))
                if len(rows) != self.count or [int(row["output_index"]) for row in rows] != list(range(self.count)):
                    raise ValueError("输出序列与时间记录不匹配")
                self.timestamps = [float(row["time_seconds"]) if row["time_seconds"] else None for row in rows]
                self.source_numbers = [int(row["source_frame_number"]) if row.get("source_frame_number") else None for row in rows]
                self.trigger_times = [float(row["trigger_time_seconds"]) if row.get("trigger_time_seconds") else None for row in rows]
                if any(value is not None and not math.isfinite(value) for value in [*self.timestamps, *self.trigger_times]):
                    raise ValueError("输出记录包含非法时间值")
                notes = ["输出帧序号为序列索引；源时间来自运行记录，来源映射见 measurements.csv"]
                fingerprint_files += [self.path / "measurements.csv", self.path.parent / "run.json"]
        if self.count > 1 and interval is None and self.timestamps is None and source.kind != "sequence":
            notes.append("TIFF 页序列没有确认的时间间隔；仅按页序号定位")
        self.meta = DataSpec(self.path.name, "sequence" if self.count > 1 or source.kind == "sequence" else "image", first.shape, str(first.dtype), "YX" if first.ndim == 2 else "YXS", self.count, duration, 1 / interval if interval else None, "run_record" if self.timestamps is not None else "user" if interval else "unavailable", "gray" if first.ndim == 2 else "RGB/RGBA", file_revision(fingerprint_files), tuple(notes))

    def read_frame(self, index: int) -> FrameData:
        if index < 0 or index >= self.count:
            raise IndexError("帧序号超出数据范围")
        if index not in self._cache:
            pixels = read_image(self.files[index] if self.source.kind == "sequence" else self.path, 0 if self.source.kind == "sequence" else index)
            if pixels.shape != self.meta.shape or str(pixels.dtype) != self.meta.dtype:
                raise ValueError(f"第 {index + 1} 帧的尺寸或类型与第一帧不一致")
            while self._cache and self._cache_size + pixels.nbytes > self._budget:
                _, old = self._cache.popitem(last=False)
                self._cache_size -= old.nbytes
            # A single oversized image is not retained in the cache.
            if pixels.nbytes <= self._budget:
                self._cache[index] = pixels
                self._cache_size += pixels.nbytes
        else:
            pixels = self._cache[index]
            self._cache.move_to_end(index)
        time = self.timestamps[index] if self.timestamps is not None else index * self.source.interval if self.source.interval else None
        return FrameData(pixels, FrameRef(index, time, source_number=self.source_numbers[index] if self.source_numbers else None, trigger_time=self.trigger_times[index] if self.trigger_times else None))

    def iter_frames(self, selection: Selection):
        if selection.mode == "current":
            if selection.current.index is None:
                raise ValueError("图像序列缺少当前帧索引")
            yield self.read_frame(selection.current.index)
            return
        start, end = (int(selection.start), int(selection.end)) if selection.mode == "range" else (0, self.count)
        if not 0 <= start < end <= self.count:
            raise ValueError("序列范围超出实际帧数")
        for index in range(start, end):
            yield self.read_frame(index)

    def close(self):
        self._cache.clear()


class VideoReader:
    def __init__(self, source: SourceSpec):
        try:
            import av
        except ImportError as error:
            raise ValueError("视频预览需要 PyAV。请按 README 安装项目依赖。") from error
        self.av = av
        self.source = source
        self.path = Path(source.path)
        self.container = av.open(str(self.path))
        try:
            if not self.container.streams.video:
                raise ValueError("文件不含视频流")
            self.stream = self.container.streams.video[0]
            self._iterator = iter(self.container.decode(self.stream))
            self._known: OrderedDict[int, int] = OrderedDict()
            self._next_index: int | None = 0
            first_av = next(self._iterator)
            self._origin = self.stream.start_time if self.stream.start_time is not None else (first_av.pts or 0)
            bits = max((component.bits for component in first_av.format.components), default=8)
            if first_av.format.name.startswith("gray"):
                self.pixel_format = first_av.format.name
            elif bits <= 8:
                self.pixel_format = "rgb24"
            elif bits <= 16:
                self.pixel_format = "rgb48le"
            else:
                raise ValueError("当前视频适配器无法保证该像素格式的数值精度")
            first = self._convert(first_av)
            self._first = first
            duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration is not None else (self.container.duration / av.time_base if self.container.duration else None)
            rate = float(self.stream.average_rate) if self.stream.average_rate else None
            self.meta = DataSpec(self.path.name, "video", first.pixels.shape, str(first.pixels.dtype), "YX" if first.pixels.ndim == 2 else "YXS", self.stream.frames or None, duration, rate, "stream_pts", self.pixel_format, file_revision([self.path]), (f"流 0；{first_av.format.name} → {self.pixel_format} 解码像素；时间相对流起点", "容器帧数为元信息；未知位置不估算帧序号"))
        except Exception:
            self.container.close()
            raise

    def _convert(self, frame) -> FrameData:
        pixels = frame.to_ndarray(format=self.pixel_format)
        validate_pixels(pixels)
        pixels.setflags(write=False)
        index = self._next_index if self._next_index is not None else self._known.get(frame.pts)
        if index is not None:
            self._next_index = index + 1
            if frame.pts is not None:
                self._known[frame.pts] = index
                if len(self._known) > 8192:
                    self._known.popitem(last=False)
        time = float((frame.pts - self._origin) * self.stream.time_base) if frame.pts is not None else None
        return FrameData(pixels, FrameRef(index, time, frame.pts))

    def first(self) -> FrameData:
        return self._first

    def read_next(self) -> FrameData:
        return self._convert(next(self._iterator))

    def _seek(self, time: float):
        if not math.isfinite(time) or time < 0:
            raise ValueError("视频时间必须是非负有限数值")
        stamp = self._origin + math.floor(time / float(self.stream.time_base))
        self.container.seek(stamp, backward=True, any_frame=False, stream=self.stream)
        self._iterator = iter(self.container.decode(self.stream))
        self._next_index = 0 if time == 0 else None

    def read_at_time(self, time: float) -> FrameData:
        self._seek(time)
        last = None
        for frame in self._iterator:
            converted = self._convert(frame)
            last = converted
            if converted.ref.time is None:
                raise ValueError("视频缺少帧时间戳，无法可靠按时间定位")
            if converted.ref.time + 1e-9 >= time:
                return converted
        if last is not None:
            return last
        raise ValueError("目标位置没有可解码帧")

    def read_previous(self, current_time: float) -> FrameData:
        if current_time <= 0:
            return self.read_at_time(0)
        self._seek(max(0, current_time - float(self.stream.time_base)))
        last = None
        for frame in self._iterator:
            converted = self._convert(frame)
            if converted.ref.time is None:
                raise ValueError("缺少时间戳，无法定位上一帧")
            if converted.ref.time >= current_time - 1e-9:
                break
            last = converted
        # Seek again so forward stepping continues immediately after the returned frame.
        return self.read_at_time(last.ref.time if last is not None else 0)

    def iter_frames(self, selection: Selection):
        if selection.mode == "current":
            if selection.current.time is None:
                raise ValueError("视频当前帧没有可靠的时间戳")
            frame = self.read_at_time(selection.current.time)
            if selection.current.pts is not None and frame.ref.pts != selection.current.pts:
                raise ValueError("源数据改变或定位失败，当前帧身份不匹配")
            yield frame
            return
        # Sequential decoding from the stream start gives authoritative ordinal indices,
        # including VFR, without retaining the full video in memory.
        with self.av.open(str(self.path)) as container:
            stream = container.streams.video[0]
            for index, av_frame in enumerate(container.decode(stream)):
                self._next_index = index
                frame = self._convert(av_frame)
                if selection.mode == "range":
                    if frame.ref.time is None:
                        raise ValueError("视频缺少时间戳，无法按时间段处理")
                    if frame.ref.time < selection.start:
                        continue
                    if frame.ref.time >= selection.end:
                        break
                yield frame

    def close(self):
        self.container.close()


def open_source(source: SourceSpec):
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"数据源不存在：{path}")
    kind = source.kind
    if kind == "auto":
        if path.is_dir():
            raise ValueError("图像序列需显式选择目录及时间间隔")
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            kind = "image"
        elif path.suffix.lower() in VIDEO_EXTENSIONS:
            kind = "video"
        else:
            raise ValueError("不支持此格式；当前支持 PNG/JPEG/BMP/TIFF、CINE 与常规视频")
    source = replace(source, kind=kind)
    if kind == "correlation":
        from eip.modes.split_correlation.storage import CorrelationReader
        return CorrelationReader(source)
    if kind == "video" and path.suffix.lower() == ".cine":
        from eip.cine_reader import CineReader
        return CineReader(source)
    return VideoReader(source) if kind == "video" else ImageReader(source)
