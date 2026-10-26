"""Alignment accepts browser null limits and respects the workbench allocation."""
import io

import numpy as np
import pytest

from eip.module_contracts import Contract, compile_flow
from eip.module_registry import new_node
from test_sequence_web import client, open_tiff, submit


@pytest.mark.parametrize("execution", [{}, {"memory_mb": None, "backend": "inherit", "precision": "native"}])
def test_alignment_blank_limit_uses_inherited_allocation(execution):
    node = new_node("alignment", shape=(2048, 2048))
    node["execution"] = execution
    source = Contract("YX", (2048, 2048), None)
    assert compile_flow([node], source, memory_mb=512)["output"]["shape"] == (2, 2048, 1024)
    with pytest.raises(ValueError, match="内存预算"):
        compile_flow([node], source, memory_mb=128)


@pytest.mark.parametrize("override,inherited", [(16, 512), (512, 16)])
def test_alignment_cannot_exceed_either_memory_limit(override, inherited):
    node = new_node("alignment", shape=(1024, 1024))
    node["execution"] = dict(memory_mb=override)
    with pytest.raises(ValueError, match="内存预算"):
        compile_flow([node], Contract("YX", (1024, 1024), 5), memory_mb=inherited)


@pytest.mark.parametrize("memory_mb", [128, 512])
def test_browser_null_budget_plan_inspect_and_process(client, tmp_path, memory_mb):
    response = client.post("/api/resources", json=dict(memory_mb=memory_mb, preview_mb=16, cache_mb=0, gpu_mb=16, backend="cpu"))
    assert response.status_code == 200, response.text
    values = np.arange(5*3*8, dtype=np.uint16).reshape(5, 3, 8)
    source = open_tiff(client, tmp_path, values)
    node = new_node("alignment", shape=(3, 8))
    node["execution"] = dict(backend="inherit", memory_mb=None, precision="native")
    response = client.post("/api/modules/plan", json=dict(source_id=source["id"], steps=[node]))
    assert response.status_code == 200 and response.json()["errors"] == [], response.text
    response = client.post("/api/modules/inspect", json=dict(source_id=source["id"], steps=[node], sample_index=0))
    assert response.status_code == 200 and response.json()["contract"]["axes"] == "PYX", response.text
    for mode in ["trial", "run"]:
        job = submit(client, source, [node], mode=mode, sample_index=0)
        assert job["status"] == "succeeded", job
        response = client.get(f"/api/jobs/{job['id']}/download?file=array")
        assert response.status_code == 200, response.text
        output = np.load(io.BytesIO(response.content), allow_pickle=False)
        expected = np.stack([values[:, :, :4], values[:, :, 4:]], axis=1)
        np.testing.assert_array_equal(output, expected[:1] if mode == "trial" else expected)
