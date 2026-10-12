"""Distribution output stays a bin-count vector from inspection to export."""
import io

import numpy as np
import pytest

from eip.module_registry import new_node
from test_sequence_web import client, open_tiff, submit


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
