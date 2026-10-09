import csv
import io
import struct

import numpy as np
import pytest
import tifffile

from eip.domain import FrameRef, Selection, SourceSpec
from eip.readers import open_source
from test_web import client, wait_job, submit


@pytest.mark.parametrize('version,time_type', [(1, 1002), (0, 1001)])
def test_cine_packed_values_time_flags_and_precise_frame_identity(cine_factory, version, time_type):
    path, pixels, ticks, _ = cine_factory(version=version, time_type=time_type)
    reader = open_source(SourceSpec(str(path)))
    try:
        assert reader.meta.frame_indexed and reader.meta.sample_bits == 10
        assert reader.meta.dtype == 'uint16' and reader.meta.frame_count == 3
        assert reader.meta.timestamp_source == 'cine_time64'
        frames = list(reader.iter_frames(Selection()))
        np.testing.assert_array_equal(np.stack([f.pixels for f in frames]), pixels)
        assert not frames[0].pixels.flags.writeable
        assert [f.ref.source_number for f in frames] == [-2, -1, 0]
        assert [f.ref.pts for f in frames] == [t - ticks[0] for t in ticks]
        assert frames[1].ref.time == (ticks[1] - ticks[0]) / 2**32
        assert frames[0].ref.trigger_time == (ticks[0] - ticks[2]) / 2**32
        assert frames[2].ref.trigger_time == 0
        assert reader.read_at_time(frames[1].ref.time).ref.index == 1
        assert reader.read_at_time(999).ref.index == 2
        assert reader.read_previous(frames[2].ref.time).ref.index == 1
        assert reader.read_next().ref.index == 2
        with pytest.raises(StopIteration):
            reader.read_next()
        selected = list(reader.iter_frames(Selection('range', frames[1].ref.time, frames[2].ref.time)))
        assert len(selected) == 1 and selected[0].ref == frames[1].ref
        assert list(reader.iter_frames(Selection('current', current=frames[1].ref)))[0].ref == frames[1].ref
        with pytest.raises(ValueError, match='身份'):
            list(reader.iter_frames(Selection('current', current=FrameRef(1, frames[1].ref.time, 100))))
        with pytest.raises(ValueError, match='范围'):
            reader.read_frame(3)
    finally:
        reader.close()


@pytest.mark.parametrize('top_down', [False, True])
def test_cine_uint8_padding_and_orientation(cine_factory, top_down):
    array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    path, _, _, _ = cine_factory(array, packed=False, bits=8, top_down=top_down)
    reader = open_source(SourceSpec(str(path)))
    try:
        np.testing.assert_array_equal(reader.read_frame(1).pixels, array[1])
        assert reader.meta.dtype == 'uint8'
    finally:
        reader.close()


def test_cine_uint16_12bit_codes_are_not_shifted(cine_factory):
    array = np.array([[[0, 1, 4095], [2048, 16, 500]]], dtype=np.uint16)
    path, _, _, _ = cine_factory(array, packed=False, bits=12)
    reader = open_source(SourceSpec(str(path)))
    try:
        np.testing.assert_array_equal(reader.read_frame(0).pixels, array[0])
        assert reader.meta.sample_bits == 12
    finally:
        reader.close()


def test_cine_without_time_table_still_has_exact_frames(cine_factory):
    path, pixels, _, _ = cine_factory(timestamps=False)
    reader = open_source(SourceSpec(str(path)))
    try:
        assert reader.meta.timestamp_source == 'unavailable' and reader.meta.duration is None
        frame = reader.read_frame(2)
        assert frame.ref == FrameRef(2, None, None, 0, None)
        np.testing.assert_array_equal(frame.pixels, pixels[2])
        assert len(list(reader.iter_frames(Selection()))) == 3
        with pytest.raises(ValueError, match='帧号'):
            reader.read_at_time(0)
        with pytest.raises(ValueError, match='缺少逐帧时间'):
            list(reader.iter_frames(Selection('range', 0, 1)))
    finally:
        reader.close()


@pytest.mark.parametrize('offset,fmt,value,message', [
    (0, '<H', 0, 'Phantom'), (6, '<H', 9, '版本'), (4, '<H', 1, 'CCI'),
    (44 + 14, '<H', 24, '像素类型'), (84 + 808, '<I', 1, 'Bayer'),
    (84 + 896, '<I', 12, '线性化'), (20, '<I', 0, '帧数'),
    (84 + 900, '<I', 7, '长度'), (44 + 20, '<I', 0, '像素长度'),
])
def test_cine_rejects_unsupported_or_malformed_headers(cine_factory, offset, fmt, value, message):
    path, _, _, _ = cine_factory()
    data = bytearray(path.read_bytes())
    struct.pack_into(fmt, data, offset, value)
    path.write_bytes(data)
    with pytest.raises(ValueError, match=message):
        open_source(SourceSpec(str(path)))


def test_cine_checks_truncation_offsets_annotation_and_time_order(cine_factory):
    path, _, _, _ = cine_factory()
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises(ValueError, match='截断'):
        open_source(SourceSpec(str(path)))
    path, _, _, offsets = cine_factory()
    data = bytearray(path.read_bytes())
    table = struct.unpack_from('<I', data, 32)[0]
    struct.pack_into('<Q', data, table, 0)
    path.write_bytes(data)
    with pytest.raises(ValueError, match='偏移表'):
        open_source(SourceSpec(str(path)))
    path, _, _, offsets = cine_factory()
    data = bytearray(path.read_bytes())
    struct.pack_into('<I', data, offsets[1], 7)
    path.write_bytes(data)
    reader = open_source(SourceSpec(str(path)))
    try:
        with pytest.raises(ValueError, match='注释'):
            reader.read_frame(1)
    finally:
        reader.close()
    path, _, _, _ = cine_factory(ticks=[10_000, 30_000, 20_000])
    with pytest.raises(ValueError, match='递增'):
        open_source(SourceSpec(str(path)))


def test_web_cine_roi_export_and_chained_frame_metadata(client, cine_factory):
    path, pixels, ticks, _ = cine_factory()
    assert client.get('/api/path', params={'path': str(path)}).json()['supported_data']
    response = client.post('/api/sources/open', json={'path': str(path)})
    assert response.status_code == 200, response.text
    data = response.json()
    source = data['source']
    assert source['meta']['frame_indexed']
    frame = client.post(f'/api/sources/{source["id"]}/frame', json={'action': 'index', 'index': 1}).json()
    assert frame['ref']['source_number'] == -1
    assert client.get(f'/api/frames/{frame["token"]}/pixel?x=3&y=1').json()['value'] == int(pixels[1, 1, 3])
    trial = wait_job(client, submit(client, source, frame))
    assert trial['status'] == 'succeeded', trial
    formal = wait_job(client, submit(client, source, frame, mode='run'))
    assert formal['status'] == 'succeeded', formal
    image = client.get(f'/api/jobs/{formal["id"]}/download?file=image').content
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(image)), pixels[1, 1:3, 2:5] * 2 + 7)
    text = client.get(f'/api/jobs/{formal["id"]}/download?file=csv').content.decode('utf-8-sig')
    row = next(csv.DictReader(io.StringIO(text)))
    assert row['source_frame'] == '1' and row['source_frame_number'] == '-1'
    assert float(row['trigger_time_seconds']) == (ticks[1] - ticks[2]) / 2**32
    result = next(s for s in client.get('/api/state').json()['sources'] if s['role'] == 'result')
    mapped = client.post(f'/api/sources/{result["id"]}/frame', json={'action': 'match', 'reference': frame['ref']}).json()
    assert mapped['ref']['source_number'] == -1 and mapped['ref']['trigger_time'] == frame['ref']['trigger_time']
    derived = client.post(f'/api/sources/{result["id"]}/promote', json={}).json()
    assert derived['frame']['ref']['source_number'] == -1
    assert derived['source']['lineage'][0]['run_id'] == formal['id']


def test_web_cine_missing_timestamps_supports_frame_mode_only(client, cine_factory):
    path, pixels, _, _ = cine_factory(timestamps=False)
    response = client.post('/api/sources/open', json={'path': str(path)})
    assert response.status_code == 200, response.text
    data = response.json()
    identifier = data['source']['id']
    frame = client.post(f'/api/sources/{identifier}/frame', json={'action': 'index', 'index': 2}).json()
    assert frame['ref']['index'] == 2 and frame['ref']['time'] is None
    assert client.post(f'/api/sources/{identifier}/frame', json={'action': 'time', 'time': 0}).status_code == 400
    response = client.post('/api/jobs', json={'source_ids': [identifier], 'mode': 'run', 'steps': [], 'selection': {'mode': 'range', 'start': 0, 'end': 1}})
    assert response.status_code == 400 and '逐帧时间' in response.json()['detail']
    formal = wait_job(client, submit(client, data['source'], frame, mode='run', selection={'mode': 'all', 'roi': None}, steps=[]))
    assert formal['status'] == 'succeeded' and formal['completed'] == formal['total'] == 3
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(client.get(f'/api/jobs/{formal["id"]}/download?file=image').content)), pixels[0])
