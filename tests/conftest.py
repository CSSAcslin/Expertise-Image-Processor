from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import tifffile


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
