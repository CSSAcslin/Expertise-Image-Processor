import numpy as np
import pytest
import tifffile
import av

from eip.domain import Selection, SourceSpec
from eip.readers import open_source


def test_uint16_reader_preserves_pixels(scientific_image):
    path, pixels = scientific_image
    reader = open_source(SourceSpec(str(path)))
    try:
        assert reader.meta.axes == "YX" and reader.meta.dtype == "uint16"
        np.testing.assert_array_equal(reader.read_frame(0).pixels, pixels)
        assert reader.read_frame(0).ref.time is None
        assert not reader.read_frame(0).pixels.flags.writeable
    finally:
        reader.close()


def test_sequence_natural_order_and_explicit_interval(tmp_path):
    for index in [10, 2, 1]:
        tifffile.imwrite(tmp_path / f"frame{index}.tif", np.full((3, 4), index, dtype=np.uint16))
    reader = open_source(SourceSpec(str(tmp_path), "sequence", 0.2))
    try:
        assert [frame.pixels[0, 0] for frame in reader.iter_frames(Selection())] == [1, 2, 10]
        assert reader.read_frame(2).ref.time == pytest.approx(0.4)
        assert reader.meta.timestamp_source == "user"
    finally:
        reader.close()


def test_sequence_mismatched_frame_fails_at_read(tmp_path):
    tifffile.imwrite(tmp_path / "1.tif", np.zeros((2, 2), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "2.tif", np.zeros((3, 2), dtype=np.uint16))
    reader = open_source(SourceSpec(str(tmp_path), "sequence", 1))
    try:
        with pytest.raises(ValueError, match="不一致"):
            reader.read_frame(1)
    finally:
        reader.close()


def test_video_real_pts_and_high_depth(vfr_video):
    path, stamps = vfr_video
    reader = open_source(SourceSpec(str(path)))
    try:
        frames = list(reader.iter_frames(Selection()))
        assert [frame.ref.time for frame in frames] == pytest.approx(stamps)
        assert [frame.ref.index for frame in frames] == list(range(5))
        assert all(frame.pixels.dtype == np.uint16 for frame in frames)
        assert [frame.pixels[0, 0] for frame in frames] == [1000, 2000, 3000, 4000, 5000]
    finally:
        reader.close()


def test_video_seek_next_and_previous_use_actual_time(vfr_video):
    path, _ = vfr_video
    reader = open_source(SourceSpec(str(path)))
    try:
        chosen = reader.read_at_time(0.08)
        assert chosen.ref.time == pytest.approx(0.11)
        assert chosen.pixels[0, 0] == 3000
        assert chosen.ref.index in {None, 2}  # Either unknown or established by sequential decoding, never FPS estimation.
        assert reader.read_next().ref.time == pytest.approx(0.16)
        assert reader.read_previous(0.16).ref.time == pytest.approx(0.11)
        frames = list(reader.iter_frames(Selection("range", 0.08, 0.3)))
        assert [frame.ref.time for frame in frames] == pytest.approx([0.11, 0.16])
    finally:
        reader.close()


def test_bad_and_missing_source_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        open_source(SourceSpec(str(tmp_path / "missing.tif")))
    path = tmp_path / "bad.tif"
    path.write_bytes(b"not an image")
    with pytest.raises(Exception):
        open_source(SourceSpec(str(path)))


def test_compressed_mp4_gop_seek_matches_sequential_decode(tmp_path):
    path = tmp_path / "gop.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=12)
        stream.width, stream.height = 64, 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 12
        for index in range(30):
            pixels = np.full((48, 64, 3), index * 7, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    sequential = open_source(SourceSpec(str(path)))
    try:
        expected = list(sequential.iter_frames(Selection()))
    finally:
        sequential.close()
    random = open_source(SourceSpec(str(path)))
    try:
        chosen = random.read_at_time(expected[19].ref.time)
        np.testing.assert_array_equal(chosen.pixels, expected[19].pixels)
        assert chosen.ref.pts == expected[19].ref.pts
        previous = random.read_previous(chosen.ref.time)
        np.testing.assert_array_equal(previous.pixels, expected[18].pixels)
        assert random.read_next().ref.pts == expected[19].ref.pts
    finally:
        random.close()
