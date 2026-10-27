"""Persistent TIFF reads must preserve lazy loading, pixels and handle lifetime."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.readers import open_source
from eip.resources import configure_reader


def stack(tmp_path, count=41):
    values = np.arange(count*5*8, dtype=np.uint16).reshape(count, 5, 8)
    path = tmp_path/"stack.tif"
    tifffile.imwrite(path, values, photometric="minisblack", metadata=dict(axes="TYX"))
    return path, values


def test_one_data_handle_for_sequential_and_random_tiff_reads(tmp_path, monkeypatch):
    path, values = stack(tmp_path)
    reader = open_source(SourceSpec(str(path), interval=.02))
    configure_reader(reader, 0)
    original = tifffile.TiffFile
    opened = []

    def tracked(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(tifffile, "TiffFile", tracked)
    try:
        for index in [*range(len(values)), 25, 0, 40, 1, 9]:
            frame = reader.read_frame(index)
            np.testing.assert_array_equal(frame.pixels, values[index])
            assert not frame.pixels.flags.writeable
            assert frame.ref.index == index
            assert frame.ref.time == pytest.approx(index*.02)
        assert len(opened) == 1
        assert reader._cache_size == 0 and not reader._cache
        assert not reader._tiff.pages.cache
        # The pinned tifffile version retains only page 0, plus compact offsets.
        assert len(reader._tiff.pages._loaded) == 1
    finally:
        reader.close()
    assert opened[0].filehandle.closed
    reader.close()
    with pytest.raises(ValueError, match="关闭"):
        reader.read_frame(0)


def test_tiff_reads_and_close_are_serialized_across_threads(tmp_path):
    path, values = stack(tmp_path)
    reader = open_source(SourceSpec(str(path)))
    configure_reader(reader, values[0].nbytes*2)
    try:
        indices = [int(i) for i in np.random.default_rng(19).integers(0, len(values), 150)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            frames = list(pool.map(reader.read_frame, indices))
        for index, frame in zip(indices, frames):
            np.testing.assert_array_equal(frame.pixels, values[index])
        assert reader._cache_size <= reader._budget
    finally:
        reader.close()


def test_directory_tiffs_do_not_retain_one_handle_per_file(tmp_path):
    for index in range(5):
        tifffile.imwrite(tmp_path/f"{index}.tif", np.full((5, 8), index, dtype=np.uint16))
    reader = open_source(SourceSpec(str(tmp_path), "sequence", .01))
    configure_reader(reader, 0)
    try:
        for index in range(5):
            assert np.all(reader.read_frame(index).pixels == index)
        assert reader._tiff is None
    finally:
        reader.close()


def test_revision_check_still_detects_source_change_with_open_handle(tmp_path):
    path, _ = stack(tmp_path)
    reader = open_source(SourceSpec(str(path)))
    configure_reader(reader, 0)
    try:
        before = reader.meta.revision
        reader.read_frame(2)
        with path.open("ab") as file:
            file.write(b"changed after import")
        verify = open_source(SourceSpec(str(path)))
        try:
            assert verify.meta.revision != before
        finally:
            verify.close()
    finally:
        reader.close()
