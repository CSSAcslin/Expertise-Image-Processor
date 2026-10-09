from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import tifffile
import struct


@pytest.fixture
def scientific_image(tmp_path):
    pixels = np.arange(48, dtype=np.uint16).reshape(6, 8) * 1000
    path = tmp_path / "scientific.tif"
    tifffile.imwrite(path, pixels)
    return path, pixels


@pytest.fixture
def vfr_video(tmp_path):
    path = tmp_path / "vfr.mkv"
    stamps = [0, 40, 110, 160, 300]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "gray16le"
        stream.time_base = Fraction(1, 1000)
        stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(stamps):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32), 1000 * (index + 1), dtype=np.uint16), format="gray16le")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path, [stamp / 1000 for stamp in stamps]


@pytest.fixture
def cine_factory(tmp_path):
    """Small independent CINE writer: bit-string packing, integer TIME64 ticks."""
    def create(pixels=None, *, packed=True, bits=10, version=1, timestamps=True,
               time_type=1002, top_down=False, ticks=None):
        if pixels is None:
            pixels = (np.arange(72, dtype=np.uint16).reshape(3, 3, 8) * 23) % 1024
            pixels[0, 0, :4] = [0, 1, 511, 1023]
        count, height, width = pixels.shape
        storage_bits = 16 if packed else pixels.dtype.itemsize * 8
        rows = ((width * storage_bits + 31) // 32) * 4
        payloads = []
        for array in pixels:
            if packed:
                bit_string = ''.join(format(int(value), '010b') for value in array.flat)
                payloads.append(bytes(int(bit_string[i:i + 8], 2) for i in range(0, len(bit_string), 8)))
            else:
                array = array if top_down else array[::-1]
                payloads.append(b''.join(row.astype('<u1' if storage_bits == 8 else '<u2').tobytes() + bytes(rows - width * storage_bits // 8) for row in array))
        setup = bytearray(900)
        struct.pack_into('<HH', setup, 140, 0x5453, len(setup))
        struct.pack_into('<I', setup, 768, 100_000)
        struct.pack_into('<I', setup, 896, bits)
        epoch = 1_700_000_000 << 32
        if ticks is None:
            ticks = [epoch + round(i * (1 << 32) / 100_000) for i in range(count)]
        clear_ticks = [value & ~3 for value in ticks]
        first_number = -2
        trigger = clear_ticks[min(2, count - 1)]
        saved_ticks = ticks if time_type == 1002 else [epoch - (1 << 20), *ticks]
        tags = struct.pack('<IHH', 8 + 8 * len(saved_ticks), time_type, 0) + b''.join(struct.pack('<Q', value | 3) for value in saved_ticks) if timestamps else b''
        offsets_offset = 84 + len(setup) + len(tags)
        offset_width = 8 if version == 1 else 4
        cursor = offsets_offset + count * offset_width
        offsets, frames = [], []
        for payload in payloads:
            offsets.append(cursor)
            frames.append(struct.pack('<II', 8, len(payload)) + payload)
            cursor += len(frames[-1])
        header = struct.pack('<HHHHiIiIIIIQ', 0x4943, 44, 0, version,
                             first_number if time_type == 1002 else first_number - 1,
                             count if time_type == 1002 else count + 1, first_number,
                             count, 44, 84, offsets_offset, trigger | 3)
        bitmap = struct.pack('<IiiHHIIiiII', 40, width, -height if top_down else height,
                             1, storage_bits, 256 if packed else 0, len(payloads[0]), 0, 0, 0, 1 << bits)
        path = tmp_path / f'fixture-{len(list(tmp_path.glob("*.cine")))}.cine'
        path.write_bytes(header + bitmap + setup + tags + b''.join(struct.pack('<Q' if version == 1 else '<I', value) for value in offsets) + b''.join(frames))
        return path, pixels, clear_ticks, offsets
    return create
