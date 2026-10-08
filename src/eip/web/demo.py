from pathlib import Path
from fractions import Fraction

import av
import numpy as np
import tifffile


def create_demo(directory: Path):
    """Real, explicitly labelled synthetic files for exploring the workflow."""
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / "示例-16位显微图.tif"
    video = directory / "示例-粒子运动.mp4"
    y, x = np.mgrid[0:480, 0:640]

    def field(t):
        value = 1200 + 700 * (x / 640 + y / 480)
        for i in range(18):
            cx = (i * 131 + 70 + t * (i % 3 + 1) * 18) % 640
            cy = (i * 83 + 60 + np.sin(t + i) * 22) % 480
            radius = 9 + i % 6 * 3
            value = value + (18000 + i * 1500) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        return np.clip(value, 0, 65535).astype(np.uint16)

    if not image.exists():
        tifffile.imwrite(image, field(0), photometric="minisblack", metadata={"axes": "YX", "description": "Synthetic demonstration, not experimental data"})
    if not video.exists():
        with av.open(str(video), "w") as container:
            stream = container.add_stream("mpeg4", rate=12)
            stream.width, stream.height, stream.pix_fmt = 640, 480, "yuv420p"
            for i in range(48):
                gray = np.clip(field(i / 12) / 160, 0, 255).astype(np.uint8)
                rgb = np.stack([gray, (gray.astype(float) * 0.9).astype(np.uint8), (gray.astype(float) * 0.7).astype(np.uint8)], axis=-1)
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = i
                frame.time_base = Fraction(1, 12)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    return [image, video]
