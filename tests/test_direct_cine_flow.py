"""Independent dedicated-flow values, finite rules, rebuilds and experiment limits."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.sequence_processing import process_sequence_job


path = Path(__file__).resolve().parents[1]/"tools/direct_cine_flow.py"
module_spec = importlib.util.spec_from_file_location("dedicated_experiment", path)
direct = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(direct)


def fixture_spec(tmp_path, count=521):
    rng = np.random.default_rng(732)
    values = rng.uniform(10, 900, (count, 6, 12))
    values[2, 1:4, 1:4] = np.nan
    values[4, 2:5, 8:11] = np.inf
    path = tmp_path/"pair.tif"
    tifffile.imwrite(path, values, photometric="minisblack", metadata={"axes": "TYX"})
    source = SourceSpec(str(path), interval=.01)
    reader = open_source(source)
    revision = reader.meta.revision
    reader.close()
    nodes = [new_node(key) for key in ("time_crop", "calibration", "alignment", "roi_crop", "time_statistics", "correlation_map")]
    nodes[1]["parameters"].update(curve="linear", input_max=4095, black_left=0, black_right=0)
    nodes[2]["parameters"].update(left=dict(x=0,y=0,width=6,height=6), right=dict(x=6,y=0,width=6,height=6),
                                    midline=[6,6], dx=1, dy=1, interpolation="nearest", mirror_x=False, mirror_y=False)
    nodes[3]["parameters"].update(x=1,y=1,width=3,height=3)
    nodes[4]["parameters"].update(mode="mean", validity="independent", output="main")
    nodes[5]["parameters"].update(window=5, stride=1, analysis="map", map_lag=0, center=None)
    spec = dict(id="dedicated-comparison", display_name="dedicated comparison", mode="run", contract_version=3, sources=[asdict(source)], revisions=[revision],
                steps=nodes, input_kinds=["image"], resources=dict(memory_mb=128, preview_mb=16, cache_mb=0, gpu_mb=64, backend="cpu"))
    return spec, values


def test_dedicated_cpu_matches_entire_workbench_across_rebuilds(tmp_path):
    spec, raw = fixture_spec(tmp_path)
    events = []
    app_spec = dict(spec, output_dir=str(tmp_path/"app"))
    process_sequence_job(app_spec, events.append, lambda:False)
    run = next(event["run"] for event in events if event["type"] == "sequence_completed")
    assert run["status"] == "succeeded",run
    expected = np.load(tmp_path/"app"/"run_dedicated-comparison"/run["results"][0]["output_directory"]/"values.npy")
    outputs = []
    for implementation in ("literal", "specialized"):
        folder = tmp_path/implementation
        result = direct.run_direct(spec, implementation=implementation, output_dir=folder, exports="full" if implementation == "specialized" else "none")
        actual = np.load(folder/"values.npy")
        np.testing.assert_array_equal(actual, expected)
        signals = np.load(folder/"signals.npy")
        roi = np.stack((raw[:,1:4,1:4],raw[:,2:5,8:11]),axis=1)/4095
        finite = np.isfinite(roi)
        counts = finite.sum(axis=(2,3))
        with np.errstate(invalid="ignore"):
            explicit = np.where(counts, np.where(finite,roi,0).sum(axis=(2,3))/np.maximum(counts,1), np.nan)
        np.testing.assert_array_equal(signals, explicit)
        assert np.isnan(signals[2,0]) and np.isfinite(signals[2,1])
        assert result["status"] == "succeeded"
        assert result["samples"] == len(raw)
        assert result["transfers"]["upload_calls"] == 0
        assert result["phases_seconds"]["setup"] > 0
        if implementation == "specialized":
            assert json.loads((folder/"dataset.json").read_text(encoding="utf-8"))["contract"] == json.loads(json.dumps(run["results"][0]["contract"]))
        outputs.append(actual)
    np.testing.assert_array_equal(*outputs)


def test_joint_valid_pair_g2_matches_independent_formula(tmp_path):
    spec, _ = fixture_spec(tmp_path, count=7)
    p = spec["steps"][-1]["parameters"]
    rng = np.random.default_rng(9)
    signal = rng.uniform(10,90,(800,2))
    signal[[3,7,21,306],0] = np.nan
    signal[[14,87,300],1] = np.inf
    expected = []
    for k in range(len(signal)-p["window"]+1):
        values = signal[k:k+p["window"]]
        values = values[np.isfinite(values).all(axis=1)]
        expected.append(np.mean(values[:,0]*values[:,1])/(np.mean(values[:,0])*np.mean(values[:,1])))
    np.testing.assert_allclose(direct.cpu_correlation(signal,p), expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("change",["curve","lag","steps","statistics","precision"])
def test_experiment_rejects_unmeasured_scientific_changes(tmp_path,change):
    spec,_ = fixture_spec(tmp_path,count=7)
    if change == "curve": spec["steps"][1]["parameters"]["curve"] = "sqrt"
    if change == "lag": spec["steps"][-1]["parameters"]["map_lag"] = 1
    if change == "steps": spec["steps"][3],spec["steps"][4] = spec["steps"][4],spec["steps"][3]
    if change == "statistics": spec["steps"][4]["parameters"]["validity"] = "joint"
    if change == "precision": spec["steps"][1].setdefault("execution",{})["precision"] = "float32"
    with pytest.raises(ValueError):
        direct.validate_spec(spec)


def test_existing_outputs_are_never_overwritten(tmp_path):
    spec,_ = fixture_spec(tmp_path,count=7)
    folder = tmp_path/"existing"
    folder.mkdir()
    sentinel = folder/"values.npy"
    sentinel.write_bytes(b"preserve")
    with pytest.raises(ValueError,match="must be empty"):
        direct.run_direct(spec,output_dir=folder,exports="none")
    assert sentinel.read_bytes() == b"preserve"


def test_real_cuda_dedicated_preprocessing_and_both_correlation_paths(tmp_path):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Real CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("Real CUDA device required")
    spec,_ = fixture_spec(tmp_path,count=270)
    cpu = tmp_path/"cpu"
    direct.run_direct(spec,output_dir=cpu,exports="none")
    expected = np.load(cpu/"values.npy")
    for implementation, alignment in (("literal","inherit"),("literal","cpu"),("specialized","inherit")):
        folder = tmp_path/f"{implementation}-{alignment}"
        result = direct.run_direct(spec,backend="gpu",implementation=implementation, alignment_backend=alignment,
                                   output_dir=folder,exports="none")
        actual = np.load(folder/"values.npy")
        np.testing.assert_allclose(actual,expected,rtol=5e-14,atol=5e-14,equal_nan=True)
        np.testing.assert_allclose(np.load(folder/"signals.npy"),np.load(cpu/"signals.npy"),rtol=5e-14,atol=5e-14,equal_nan=True)
        assert result["actual_backend"] == "gpu"
        assert result["transfers"]["download_calls"] > 0


def point_fixture_spec(tmp_path, count=521):
    spec, raw = fixture_spec(tmp_path, count=count)
    point = new_node("point_extract")
    point["parameters"].update(x=1,y=1,output="main")
    spec["steps"] = [*spec["steps"][:3], point, spec["steps"][-1]]
    spec["id"] = "point-comparison"
    return spec, raw


def test_paired_point_extraction_is_not_a_spatial_mean_and_matches_workbench(tmp_path, monkeypatch):
    spec, raw = point_fixture_spec(tmp_path)
    events = []
    process_sequence_job(dict(spec,output_dir=str(tmp_path/"app")),events.append,lambda:False)
    run = next(event["run"] for event in events if event["type"] == "sequence_completed")
    assert run["status"] == "succeeded", run
    expected = np.load(tmp_path/"app/run_point-comparison"/run["results"][0]["output_directory"]/"values.npy")
    def no_spatial_mean(*args, **kwargs):
        raise AssertionError("Point experiment must not call a spatial statistic")
    monkeypatch.setattr(direct,"mean_finite",no_spatial_mean)
    expected_signals = np.stack((raw[:,1,1],raw[:,2,8]),axis=1)/4095
    expected_signals[~np.isfinite(expected_signals)] = np.nan
    for implementation in ("literal", "specialized"):
        folder = tmp_path/implementation
        result = direct.run_direct(spec,implementation=implementation,output_dir=folder,exports="full" if implementation == "specialized" else "none")
        np.testing.assert_array_equal(np.load(folder/"signals.npy"),expected_signals)
        np.testing.assert_array_equal(np.load(folder/"values.npy"),expected)
        assert result["semantic"] == "pixel_g2"
        assert result["parameters"]["sampling"]["kind"] == "point"
        assert result["preprocessing_work"]["consumed_roi_pixels"] is None
        assert result["preprocessing_work"]["consumed_point_samples"] == 2*len(raw)
        assert result["preprocessing_work"]["sampling_sensor_coordinates"] == dict(left=[1,1],right=[2,8])
        if implementation == "specialized":
            assert json.loads((folder/"dataset.json").read_text(encoding="utf-8"))["contract"] == json.loads(json.dumps(run["results"][0]["contract"]))
            expected_folder = tmp_path/"app/run_point-comparison"/run["results"][0]["output_directory"]
            assert (folder/"references.jsonl").read_bytes() == (expected_folder/"references.jsonl").read_bytes()
            assert (folder/"measurements.csv").read_bytes() == (expected_folder/"measurements.csv").read_bytes()


def test_point_side_output_is_outside_this_experiment(tmp_path):
    spec,_ = point_fixture_spec(tmp_path,count=7)
    spec["steps"][3]["parameters"]["output"] = "side"
    with pytest.raises(ValueError,match="main-route"):
        direct.validate_spec(spec)


def test_real_cuda_point_literals_and_specialization_match_cpu_without_mean(tmp_path, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("Real CUDA device required")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("Real CUDA device required")
    spec, _ = point_fixture_spec(tmp_path,count=270)
    cpu = tmp_path/"cpu"
    direct.run_direct(spec,output_dir=cpu,exports="none")
    expected = np.load(cpu/"values.npy")
    def no_spatial_mean(*args, **kwargs):
        raise AssertionError("Point experiment must not call a spatial statistic")
    monkeypatch.setattr(direct,"mean_finite",no_spatial_mean)
    assert np.isfinite(expected).all(), "The CUDA point check must exercise finite g2, not merely matching NaNs"
    for implementation, alignment in (("literal","inherit"),("literal","cpu"),("specialized","inherit")):
        folder = tmp_path/f"{implementation}-{alignment}"
        result = direct.run_direct(spec,backend="gpu",implementation=implementation, alignment_backend=alignment,
                                   output_dir=folder,exports="none")
        np.testing.assert_array_equal(np.load(folder/"signals.npy"),np.load(cpu/"signals.npy"))
        np.testing.assert_allclose(np.load(folder/"values.npy"),expected,rtol=5e-14,atol=5e-14,equal_nan=True)
        assert result["actual_backend"] == "gpu"
        assert result["preprocessing_work"]["spatial_reduction"].startswith("none")
