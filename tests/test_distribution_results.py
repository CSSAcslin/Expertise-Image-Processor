"""Distribution output stays a bin-count vector from inspection to export."""
import io
from collections import OrderedDict

import numpy as np
import pytest

from eip.module_registry import new_node
from test_sequence_web import client, open_tiff, submit
from test_axis_modules import Reader
from eip.sequence_engine import SequenceEngine


@pytest.mark.parametrize("output", ["main", "side"])
def test_distribution_inspection_and_saved_histogram(client, tmp_path, output):
    values = np.array([
        [[0, 1, 2, 3], [4, 5, np.nan, 9]],
        [[0, 0, 1, 2], [2, 3, 4, np.inf]],
        [[np.nan, -1, 5, 9], [np.inf, np.nan, -np.inf, 6]],
    ])
    expected = np.array([[2, 3], [3, 4], [0, 0]], dtype=np.int64)
    source = open_tiff(client, tmp_path, values)
    node = new_node("distribution")
    node["parameters"].update(bins=2, minimum=0, maximum=4, output=output)
    side_id = f"{node['id']}:data" if output=="side" else None
    request = dict(source_id=source["id"], steps=[node], sample_index=1, side_output=side_id)
    response = client.post("/api/modules/inspect", json=request)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["kind"] == "signals" and data["profile"]
    assert data["contract"]["full_axes"] == "TB"
    assert data["contract"]["shape"] == [2] and data["contract"]["length"] == 3
    assert data["contract"]["semantic"] == "distribution"
    assert data["values"] == [[3, 4]]
    assert data["reference"]["index"] == 1
    assert data["node_id"] == node["id"]
    assert data["mappings"][node["id"]]["edges"] == [0, 2, 4]
    assert client.get("/api/jobs").json() == []
    missing = client.post("/api/modules/inspect", json=dict(request, side_output="missing:data"))
    assert missing.status_code == 400
    if output=="side":
        main = client.post("/api/modules/inspect", json=dict(request, side_output=None)).json()
        assert main["kind"] == "image" and main["contract"]["full_axes"] == "TYX"

    job = submit(client, source, [node])
    assert job["status"] == "succeeded", job
    item = job["run"]["results"][0]
    result = item["side_outputs"][0] if side_id else item
    assert result["contract"]["full_axes"] == "TB"
    assert result["contract"]["dtype"] == "int64"
    params = dict(side_output=side_id) if side_id else {}
    download = f"/api/jobs/{job['id']}/download"
    saved = client.get(download, params=dict(params, file="array"))
    np.testing.assert_array_equal(np.load(io.BytesIO(saved.content), allow_pickle=False), expected)
    for index in (0, 1, 2):
        shown = client.post(f"/api/jobs/{job['id']}/sequence", json=dict(params, index=index, profile=True)).json()
        assert shown["kind"] == "signals" and shown["indices"] == [index]
        assert shown["values"] == [expected[index].tolist()]
    graphic = client.get(download, params=dict(params, file="svg"))
    assert graphic.status_code == 200, graphic.text
    assert all(label in graphic.text for label in ("Intensity", "Frequency", "Histogram: sample 1"))
    png = client.get(download, params=dict(params, file="png"))
    assert png.status_code == 200 and png.content.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("minimum,maximum,edges,expected", [
    (None, None, [2, 6, 10], [[2, 0], [0, 2]]),
    (0, None, [0, 5, 10], [[2, 0], [0, 2]]),
    (None, 12, [2, 7, 12], [[2, 0], [1, 1]]),
])
def test_auto_bounds_use_complete_edited_input_and_match_trial(client, tmp_path, minimum, maximum, edges, expected):
    values = np.array([
        [[-999, 0, 1, 999], [-99, 30, 40, 99]],
        [[0, 2, 4, 8], [-99, 30, 40, 99]],
        [[-4, 6, 10, 12], [-99, 30, 40, 99]],
    ], dtype=float)
    source = open_tiff(client, tmp_path, values)
    crop, roi, histogram = [new_node(a) for a in ("time_crop", "roi_crop", "distribution")]
    crop["parameters"].update(first=2, last=3)
    roi["parameters"].update(x=1, y=0, width=2, height=1)
    histogram["parameters"].update(bins=2, minimum=minimum, maximum=maximum)
    nodes = [crop, roi, histogram]
    side_id = f"{histogram['id']}:data"
    for index in (0, 1):
        response = client.post("/api/modules/inspect", json=dict(source_id=source["id"], steps=nodes, side_output=side_id, sample_index=index))
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["values"] == [expected[index]]
        mapping = data["mappings"][histogram["id"]]
        assert mapping["edges"] == edges
        assert mapping["detected"] == dict(minimum=2, maximum=10, samples=2)
        assert mapping["range_scope"] == "complete_upstream_sequence"
    assert client.get("/api/jobs").json() == []
    for mode in ("trial", "run"):
        job = submit(client, source, nodes, mode=mode, sample_index=1)
        assert job["status"] == "succeeded", job
        side = job["run"]["results"][0]["side_outputs"][0]
        assert side["mappings"][histogram["id"]]["edges"] == edges
        assert side["module"]["parameters"]["minimum"] == minimum
        assert side["module"]["parameters"]["maximum"] == maximum
        response = client.get(f"/api/jobs/{job['id']}/download", params=dict(file="array", side_output=side_id))
        actual = np.load(io.BytesIO(response.content), allow_pickle=False)
        np.testing.assert_array_equal(actual, expected[1:] if mode=="trial" else expected)


@pytest.mark.parametrize("minimum,maximum,edges,expected", [
    (None, None, [-.5, 0, .5], [0, 8]),
    (0, None, [0, .25, .5], [8, 0]),
    (None, 0, [-.5, -.25, 0], [0, 8]),
])
def test_constant_auto_bounds_remain_a_valid_histogram(tmp_path, minimum, maximum, edges, expected):
    node = new_node("distribution")
    node["parameters"].update(bins=2, minimum=minimum, maximum=maximum, output="main")
    engine = SequenceEngine(Reader(np.zeros((2, 2, 4))), [node], workdir=tmp_path)
    try:
        np.testing.assert_array_equal(engine.get(1, 0), expected)
        assert engine.mappings[node["id"]]["edges"] == edges
    finally:
        engine.close()


@pytest.mark.parametrize("values,minimum,maximum,message", [
    (np.full((2, 2, 4), np.nan), None, None, "没有有限值"),
    (np.zeros((2, 2, 4)), 5, None, "上界须大于下界"),
    (np.ones((2, 2, 4)), None, -5, "上界须大于下界"),
])
def test_auto_bounds_report_unavailable_or_conflicting_ranges(tmp_path, values, minimum, maximum, message):
    node = new_node("distribution")
    node["parameters"].update(minimum=minimum, maximum=maximum, output="main")
    engine = SequenceEngine(Reader(values), [node], workdir=tmp_path)
    try:
        with pytest.raises(ValueError, match=message):
            engine.get(1, 0)
    finally:
        engine.close()


def test_auto_bounds_after_correlation_keep_the_same_scope_in_inspection(tmp_path):
    values = (np.arange(12*2*4, dtype=float).reshape(12, 2, 4)%13)+1
    align = new_node("alignment", shape=(2, 4))
    correlation = new_node("correlation_map", frames=12)
    correlation["parameters"].update(window=3, min_samples=2, stride=1, map_lag=0)
    histogram = new_node("distribution")
    histogram["parameters"].update(bins=4, minimum=None, maximum=None)
    nodes = [align, correlation, histogram]
    engines = [SequenceEngine(Reader(values), nodes, workdir=tmp_path, single=single) for single in (False, True)]
    try:
        side_id = f"{histogram['id']}:data"
        assert engines[0].output.length == engines[1].output.length == 10
        for engine in engines:
            engine.get_side_output(side_id, 0)
        np.testing.assert_allclose(engines[0].mappings[histogram["id"]]["edges"], engines[1].mappings[histogram["id"]]["edges"])
    finally:
        for engine in engines:
            engine.close()


def test_auto_bounds_ignore_nonfinite_values(tmp_path):
    values = np.array([[[np.nan, np.inf, -np.inf, -5], [0, 2, 7, 4]]])
    node = new_node("distribution")
    node["parameters"].update(bins=2, minimum=None, maximum=None, output="main")
    engine = SequenceEngine(Reader(values), [node], workdir=tmp_path)
    try:
        np.testing.assert_array_equal(engine.get(1, 0), [2, 3])
        assert engine.mappings[node["id"]]["edges"] == [-5, 1, 7]
    finally:
        engine.close()


def test_auto_bounds_cache_reuses_a_range_and_invalidates_upstream_edits(tmp_path):
    class CountingReader(Reader):
        reads = 0

        def read_frame(self, index):
            self.reads += 1
            return super().read_frame(index)

    reader = CountingReader(np.arange(24, dtype=float).reshape(3, 2, 4))
    roi, histogram = [new_node(a) for a in ("roi_crop", "distribution")]
    roi["parameters"].update(x=0, y=0, width=2, height=1)
    histogram["parameters"].update(bins=2, minimum=None, maximum=None, output="main")
    cache = OrderedDict()
    for x, expected_reads, limits in ((0, 3, [0, 17]), (0, 1, [0, 17]), (1, 3, [1, 18])):
        roi["parameters"]["x"] = x
        reader.reads = 0
        engine = SequenceEngine(reader, [roi, histogram], workdir=tmp_path, single=True, range_cache=cache)
        try:
            engine.get(2, 1)
            assert reader.reads == expected_reads
            edges = engine.mappings[histogram["id"]]["edges"]
            assert [edges[0], edges[-1]] == limits
        finally:
            engine.close()
