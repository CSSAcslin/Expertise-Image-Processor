"""Read supported monochrome Phantom CINEs without loading their pixel stream.

Binary layout: Vision Research's Cine File Format (release 705), sections
3.1--3.5 and 5.1. Packed samples remain their stored 10-bit code values;
this reader performs no implicit linearization or camera image processing.
"""
from __future__ import annotations

from pathlib import Path
import math
import struct

import numpy as np

from eip.domain import DataSpec, FrameData, FrameRef, Selection, SourceSpec, file_revision


TICKS_PER_SECOND = 1 << 32


def unpack10(payload: bytes, count: int) -> np.ndarray:
    """Four consecutive MSB-first 10-bit samples in each five-byte group."""
    if count % 4 or len(payload) != count * 5 // 4:
        raise ValueError("CINE 10 位打包数据长度与图像尺寸不匹配")
    groups = np.frombuffer(payload, dtype=np.uint8).reshape(-1, 5).astype(np.uint16)
    pixels = np.empty((len(groups), 4), dtype=np.uint16)
    pixels[:, 0] = (groups[:, 0] << 2) | (groups[:, 1] >> 6)
    pixels[:, 1] = ((groups[:, 1] & 63) << 4) | (groups[:, 2] >> 4)
    pixels[:, 2] = ((groups[:, 2] & 15) << 6) | (groups[:, 3] >> 2)
    pixels[:, 3] = ((groups[:, 3] & 3) << 8) | groups[:, 4]
    return pixels.reshape(-1)


class CineReader:
    def __init__(self, source: SourceSpec):
        self.source, self.path = source, Path(source.path)
        self.file = self.path.open("rb")
        self.file_size = self.path.stat().st_size
        self.position = -1
        try:
            self._parse()
        except Exception:
            self.file.close()
            raise

    def _read(self, offset, size):
        if offset < 0 or size < 0 or offset + size > self.file_size:
            raise ValueError("CINE 文件被截断或包含非法偏移")
        self.file.seek(offset)
        data = self.file.read(size)
        if len(data) != size:
            raise ValueError("CINE 文件读取不完整")
        return data

    def _parse(self):
        values = struct.unpack("<HHHHiIiIIIIQ", self._read(0, 44))
        marker, header_size, compression, version, first_recorded, total, self.first_number, self.count, bitmap_offset, setup_offset, offsets_offset, trigger = values
        if marker != 0x4943 or header_size < 44 or version not in {0, 1}:
            raise ValueError("不是受支持的 Phantom CINE 文件（版本 0 / 1）")
        if compression not in {0, 2}:
            raise ValueError("当前 CINE 适配器支持灰度原始数据，不支持压缩 CCI")
        if not (header_size <= bitmap_offset < setup_offset < offsets_offset):
            raise ValueError("CINE 头部结构顺序无效")
        if not 1 <= self.count <= 8_000_000 or total < self.count:
            raise ValueError("CINE 帧数无效或索引超过当前内存预算")
        if self.count * 24 > 128 * 1024 * 1024:
            raise ValueError("CINE 帧索引及时间表超过 128 MiB 预算")
        size, self.width, height, planes, self.storage_bits, self.packing, self.image_size, *_ = struct.unpack("<IiiHHIIiiII", self._read(bitmap_offset, 40))
        self.height = abs(height)
        self.bottom_up = height > 0 and self.packing == 0
        if size < 40 or bitmap_offset + size > setup_offset or planes != 1 or self.width <= 0 or not self.height:
            raise ValueError("CINE 图像尺寸或平面数量无效")
        if self.storage_bits not in {8, 16} or self.packing not in {0, 256}:
            raise ValueError("0.3.1 支持 8 / 16 位容器和 10 位 Packed 灰度 CINE；该像素类型尚未支持")
        setup_head = self._read(setup_offset, 144)
        mark, length = struct.unpack_from("<HH", setup_head, 140)
        if mark != 0x5453 or length < 144 or setup_offset + length > offsets_offset:
            raise ValueError("CINE 相机设置结构无效")
        setup = self._read(setup_offset, length)
        if compression == 2 and length < 812:
            raise ValueError("此 RAW CINE 未提供可确认的颜色类型，不能作为灰度数据导入")
        def field(offset, default=0):
            return struct.unpack_from("<I", setup, offset)[0] if offset + 4 <= length else default
        self.sample_bits = field(896, self.storage_bits)
        if field(808) != 0:
            raise ValueError("该 CINE 是彩色 / Bayer 数据，需专用颜色适配；当前小版本只支持灰度")
        if self.packing == 256 and (self.storage_bits != 16 or self.sample_bits != 10):
            raise ValueError("当前支持保存值为 10 位的 Packed CINE；该文件需要其他位深或显式线性化适配")
        if not 1 <= self.sample_bits <= self.storage_bits:
            raise ValueError("CINE 有效位深无效")
        count = self.width * self.height
        self.row_bytes = ((self.width * self.storage_bits + 31) // 32) * 4
        expected = count * 5 // 4 if self.packing == 256 else self.row_bytes * self.height
        if (self.packing == 256 and count % 4) or self.image_size != expected or expected > 256 * 1024 * 1024:
            raise ValueError("CINE 像素长度不匹配或单帧超过当前读取预算")
        offset_dtype = "<u8" if version == 1 else "<u4"
        offset_bytes = 8 if version == 1 else 4
        self.offsets = np.frombuffer(self._read(offsets_offset, self.count * offset_bytes), dtype=offset_dtype).copy()
        if np.any(self.offsets < offsets_offset + self.count * offset_bytes) or np.any(self.offsets > self.file_size - 8) or np.any(self.offsets[1:] <= self.offsets[:-1]):
            raise ValueError("CINE 图像偏移表无效")
        self.ticks = None
        old_times = None
        cursor = setup_offset + length
        blocks = 0
        while cursor < offsets_offset:
            blocks += 1
            if blocks > 4096:
                raise ValueError("CINE 扩展块数量超过当前读取上限")
            if cursor + 8 > offsets_offset:
                raise ValueError("CINE 时间 / 扩展数据块被截断")
            block_size, block_type, _ = struct.unpack("<IHH", self._read(cursor, 8))
            if block_size < 8 or cursor + block_size > offsets_offset:
                raise ValueError("CINE 扩展数据块长度无效")
            if block_type == 1002:
                if self.ticks is not None or block_size - 8 != self.count * 8:
                    raise ValueError("CINE 逐帧时间记录与帧数不匹配")
                self.ticks = np.frombuffer(self._read(cursor + 8, self.count * 8), dtype="<u8").copy()
            elif block_type == 1001 and self.ticks is None:
                start = self.first_number - first_recorded
                if start < 0 or start + self.count > total or block_size - 8 < total * 8:
                    raise ValueError("CINE 旧版时间记录与保存范围不匹配")
                old_times = np.frombuffer(self._read(cursor + 8 + start * 8, self.count * 8), dtype="<u8").copy()
            cursor += block_size
        if self.ticks is None:
            self.ticks = old_times
        self.trigger_ticks = trigger & ~3
        self.times = None
        if self.ticks is not None:
            self.ticks &= np.uint64(0xFFFFFFFFFFFFFFFC)  # Low two bits are synchronization/event flags.
            if np.any(self.ticks[1:] < self.ticks[:-1]):
                raise ValueError("CINE 时间记录不是递增序列，无法可靠按时间定位")
            self.times = (self.ticks - self.ticks[0]).astype(np.float64) / TICKS_PER_SECOND
        rate = field(768, struct.unpack_from("<H", setup)[0]) or None
        duration = None
        if self.times is not None:
            differences = np.diff(self.times)
            positive = differences[differences > 0]
            tail = float(np.median(positive)) if positive.size else (1 / rate if rate else 0)
            duration = float(self.times[-1] + tail)
        self.meta = DataSpec(self.path.name, "video", (self.height, self.width), "uint16" if self.storage_bits == 16 else "uint8", "YX", self.count, duration, rate, "cine_time64" if self.times is not None else "unavailable", "gray", file_revision([self.path]), (f"Phantom CINE v{version}；采集源帧 {self.first_number} ～ {self.first_number + self.count - 1}", f"保存值 {self.sample_bits} bit；{'Packed 解码为原始码值' if self.packing == 256 else '未打包灰度值'}；不隐式应用相机图像处理", "按文件偏移表精确定位；显示帧号从 1 开始；时间相对文件首帧，另保留触发时间差" if self.times is not None else "无逐帧时间记录；仍可按精确帧号定位，按时间范围处理不可用"), frame_indexed=True, sample_bits=self.sample_bits)

    def read_frame(self, index: int) -> FrameData:
        if type(index) is not int or not 0 <= index < self.count:
            raise ValueError("CINE 帧号超出范围")
        offset = int(self.offsets[index])
        annotation = struct.unpack("<I", self._read(offset, 4))[0]
        if annotation < 8 or offset + annotation > self.file_size:
            raise ValueError("CINE 帧注释长度无效")
        size = struct.unpack("<I", self._read(offset + annotation - 4, 4))[0]
        if size != self.image_size:
            raise ValueError("CINE 帧数据大小与头部不一致")
        if index + 1 < self.count and offset + annotation + size > int(self.offsets[index + 1]):
            raise ValueError("CINE 帧数据重叠或被截断")
        data = self._read(offset + annotation, size)
        if self.packing == 256:
            pixels = unpack10(data, self.width * self.height).reshape(self.height, self.width)
        else:
            dtype = np.uint8 if self.storage_bits == 8 else np.dtype("<u2")
            padded = np.frombuffer(data, dtype=dtype).reshape(self.height, self.row_bytes // (self.storage_bits // 8))
            pixels = padded[:, :self.width]
            if self.bottom_up:
                pixels = pixels[::-1]
            pixels = pixels.copy()
        pixels.setflags(write=False)
        self.position = index
        ticks = int(self.ticks[index]) if self.ticks is not None else None
        return FrameData(pixels, FrameRef(index, float(self.times[index]) if self.times is not None else None, ticks - int(self.ticks[0]) if ticks is not None else None, self.first_number + index, (ticks - self.trigger_ticks) / TICKS_PER_SECOND if ticks is not None else None))

    def read_at_time(self, time):
        if not math.isfinite(time) or time < 0:
            raise ValueError("定位时间必须是非负有限数值")
        if self.times is None:
            raise ValueError("此 CINE 没有逐帧时间记录，请使用帧号定位")
        index = min(self.count - 1, int(np.searchsorted(self.times, time - 1e-12)))
        return self.read_frame(index)

    def read_next(self):
        if self.position + 1 >= self.count:
            raise StopIteration
        return self.read_frame(self.position + 1)

    def read_previous(self, time):
        if self.times is None:
            return self.read_frame(max(0, self.position - 1))
        return self.read_frame(max(0, int(np.searchsorted(self.times, time - 1e-12)) - 1))

    def iter_frames(self, selection: Selection):
        if selection.mode == "current":
            ref = selection.current
            frame = self.read_frame(ref.index) if ref.index is not None else self.read_at_time(ref.time)
            if ref.pts is not None and ref.pts != frame.ref.pts:
                raise ValueError("CINE 当前帧身份不匹配，请重新定位")
            yield frame
        else:
            start, end = 0, self.count
            if selection.mode == "range":
                if self.times is None:
                    raise ValueError("此 CINE 缺少逐帧时间，不能按秒选择处理范围；可处理当前帧或全部数据")
                start = int(np.searchsorted(self.times, selection.start - 1e-12))
                end = int(np.searchsorted(self.times, selection.end - 1e-12))
            for index in range(start, end):
                yield self.read_frame(index)

    def close(self):
        self.file.close()
