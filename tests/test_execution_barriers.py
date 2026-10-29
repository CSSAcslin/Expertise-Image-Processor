"""Opaque stage execution must not be duplicated by left/right range requests."""
from pathlib import Path

import numpy as np
import pytest

from eip.algorithms import load_manifest
from eip.module_registry import new_node
from test_gpu_pipeline_0_8_4 import HostDevice
from test_spatial_execution import alignment, make, node


def stateful_plugin():
    manifest = Path(__file__).resolve().parents[1]/"examples/plugins/invert.json"
    spec = load_manifest(str(manifest))
    return new_node(spec.id, shape=(12, 20), frames=67, plugins=(spec,))


def setup_engine(tmp_path, values, backend, optimize):
    steps = [stateful_plugin(), alignment(), node("point_extract", x=3, y=2)]
    steps[0]["execution"] = dict(backend="cpu")
    if backend == "mixed":
        steps[1]["execution"] = dict(backend="cpu")
    engine = make(tmp_path, values, steps,
                  "host_gpu" if backend == "mixed" else backend, optimize)
    calls = []
    def process(pixels, context):
        # A plugin may retain state or use an RNG. Splitting one invocation
        # into independent left/right calls would change a single sample.
        calls.append(float(pixels[0, 0]))
        return pixels+len(calls)
    engine.instances[0].instances[0].process = process
    if engine.spatial is not None:
        # A correctness boundary cannot depend on a cache being large enough
        # to keep the opaque stage's complete output, even for one frame.
        engine.spatial.raw_limit = 1
    assert engine.cache.budget == 0
    return engine, calls


@pytest.mark.parametrize("backend", ["cpu", "host_gpu", "mixed"])
@pytest.mark.parametrize("access", ["single", "batch"])
def test_opaque_plugin_runs_once_per_required_sample_with_tiny_range_cache(
        tmp_path, monkeypatch, backend, access):
    monkeypatch.setattr("eip.modes.split_correlation.backend.Backend", HostDevice)
    values = (np.arange(67, dtype=np.float32)[:, None, None]
              +np.arange(240, dtype=np.float32).reshape(12, 20)[None, ...]/100)
    original = values.copy()
    old, old_calls = setup_engine(tmp_path/"old", values, backend, False)
    new, new_calls = setup_engine(tmp_path/"new", values, backend, True)
    try:
        if access == "single":
            indices = [0, 5, 1, 0, 66]
            expected = [old.get(3, j).copy() for j in indices]
            actual = [new.get(3, j).copy() for j in indices]
        else:
            expected = [v.copy() for _, v in old.iter_outputs(range(old.output.length))]
            actual = [v.copy() for _, v in new.iter_outputs(range(new.output.length))]
        np.testing.assert_array_equal(actual, expected)
        assert new_calls == old_calls
        if access == "batch":
            assert len(new_calls) == len(values)
        elif backend == "cpu":
            assert len(new_calls) == len(indices)
        # Both channels of each paired sample came from the same invocation.
        actual = np.asarray(actual)
        np.testing.assert_allclose(actual[:, 1]-actual[:, 0], .1,
                                   rtol=0., atol=1e-5)
        assert any("upstream_complete_or_stateful_dependency" in key
                   for key in new.spatial.record()["full_barriers"])
        assert new.spatial.record()["raw_peak_bytes"] <= 1
        np.testing.assert_array_equal(values, original)
    finally:
        new.close()
        old.close()
