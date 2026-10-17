"""Device-flow emulation, bounded large sequences and optional real CUDA parity.

The host adapter tests scheduling and numerical definitions, not CUDA performance.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from eip.algorithms import Cancelled
from eip.domain import DataSpec, FrameData, FrameRef, SourceSpec
from eip.exact_quantile import external_quantile, finite_quantile
from eip.module_contracts import Contract
from eip.module_registry import catalog, new_node
from eip.sequence_engine import SequenceEngine


class Reader:
    def __init__(self, values, contract=None):
        self.values = np.asarray(values)
        self.reads = []
        self.meta = DataSpec("fixture", "sequence", self.values.shape[1:], str(self.values.dtype), "YX",
                             len(values), None, None, "fixture", "RGB", "fixed")
        if contract:
            self.meta = replace(self.meta, contract=contract.record())
        self.source = SourceSpec("fixture", "sequence")

    def read_frame(self, i):
        self.reads.append(i)
        return FrameData(self.values[i], FrameRef(i, i*.01))


class HostDevice:
    """No CUDA claims: NumPy arrays exercising the real device scheduler."""
    def __init__(self, requested, budget):
        self.xp, self.name, self.reason = np, "cpu" if requested=="cpu" else "gpu", None

    def host(self, value):
        return np.array(value, copy=True)

    def close(self):
        pass


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setattr("eip.modes.split_correlation.backend.Backend", HostDevice)


def node(a, **p):
    n = new_node(a, shape=(4, 8), frames=21)
    n["parameters"].update(p)
    return n


def make(tmp_path, values, nodes, gpu, *, contract=None, resources=None):
    nodes = deepcopy(nodes)
    for n in nodes:
        n.setdefault("execution", {})["backend"] = "gpu" if gpu else "cpu"
    return SequenceEngine(Reader(values, contract), nodes, workdir=tmp_path,
                          resources=resources or dict(memory_mb=256, preview_mb=16, cache_mb=0, gpu_mb=128, backend="cpu"))


def compare(tmp_path, values, nodes, *, contract=None, indices=None):
    cpu, gpu = make(tmp_path/"cpu", values, nodes, False, contract=contract), make(tmp_path/"gpu", values, nodes, True, contract=contract)
    try:
        assert cpu.output.record() == gpu.output.record()
        for j in indices if indices is not None else range(cpu.output.length):
            expected, actual = cpu.get(len(nodes), j), gpu.get(len(nodes), j)
            assert expected.dtype == actual.dtype
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-11, equal_nan=True)
    finally:
        cpu.close(); gpu.close()


def test_visible_builtin_catalog_declares_device_execution():
    current = [s for s in catalog().values() if not s.get("legacy")]
    assert len(current) >= 25
    assert all(s["backends"] == ["cpu", "gpu"] for s in current)


def test_contiguous_nodes_batch_upload_and_only_download_final_values(tmp_path, device):
    values = np.arange(100)[:, None, None]+np.arange(64).reshape(8, 8)
    values = values.astype(np.float64)
    nodes = [node("gain_offset", gain=2., offset=1.), node("curve", curve="identity"),
             node("roi_crop", x=1, y=1, width=4, height=4), node("point_extract", x=0, y=1)]
    e = make(tmp_path, values, nodes, True)
    try:
        for i in range(100):
            assert e.get(4, i) == 2*(i+17)+1
        metrics = e.device.record()
        assert metrics["downloads"] == 100
        assert metrics["upload_batches"] == 4 and metrics["uploaded_samples"] == 100
        assert metrics["largest_upload_batch"] == 32
        assert metrics["cache_peak_bytes"] <= metrics["cache_budget_bytes"]
        assert len(e.source.reader.reads) == 100
    finally:
        e.close()


def test_explicit_cpu_node_is_a_real_boundary(tmp_path, device):
    values = np.arange(12*4*8, dtype=float).reshape(12, 4, 8)+1
    nodes = [node("gain_offset", gain=2), node("curve", curve="sqrt", input_max=1), node("gain_offset", offset=1)]
    for n in nodes:
        n["execution"] = dict(backend="gpu")
    nodes[1]["execution"]["backend"] = "cpu"
    e = SequenceEngine(Reader(values), nodes, workdir=tmp_path)
    try:
        for i in range(12):
            np.testing.assert_allclose(e.get(3, i), np.sqrt(values[i]*2)+1)
        assert e.device.metrics["downloads"] == 24
        assert e.device_records[nodes[1]["id"]]["actual"] == "cpu"
    finally:
        e.close()


@pytest.mark.parametrize("mode", ["mean", "sum", "min", "max", "variance", "std", "quantile", "count"])
def test_time_statistics_pair_validity_matches_cpu(tmp_path, device, mode):
    values = np.arange(7*2*3*4, dtype=float).reshape(7, 2, 3, 4)
    values[2, 0, 1, 1] = np.nan
    values[:, 1, 2, 2] = np.inf
    compare(tmp_path, values, [node("time_statistics", mode=mode, quantile=.73)], contract=Contract("PYX", (2, 3, 4), 7))


@pytest.mark.parametrize("scope", ["all", "samples", "space_time"])
@pytest.mark.parametrize("mode", ["mean", "variance", "quantile", "count"])
def test_global_statistics_batch_axes_and_empty_components(tmp_path, device, scope, mode):
    values = np.arange(35*2*3*4, dtype=float).reshape(35, 2, 3, 4)-100
    values[2, 0, 1, 1] = np.nan
    values[:, 1, 2, 2] = np.inf
    compare(tmp_path, values, [node("global_statistics", mode=mode, scope=scope, quantile=.73)], contract=Contract("PYX", (2, 3, 4), 35))


def test_ten_thousand_frames_reduce_with_bounded_batch_and_one_output_download(tmp_path, device):
    values = np.arange(10000, dtype=float).reshape(10000, 1, 1)
    e = make(tmp_path, values, [node("global_statistics", mode="variance")], True)
    try:
        assert e.get(1, 0) == pytest.approx(8333333.25)
        metrics = e.device.record()
        assert metrics["downloads"] == 1
        assert metrics["upload_batches"] == 313
        assert metrics["uploaded_samples"] == 10000
        assert metrics["cache_peak_bytes"] <= metrics["cache_budget_bytes"]
    finally:
        e.close()


def test_variance_of_small_fluctuations_on_large_baseline(tmp_path,device):
    values=(1e12+np.tile([-.25,.25],5000)).reshape(10000,1,1)
    for gpu in (False,True):
        e=make(tmp_path,values,[node("global_statistics",mode="variance")],gpu)
        try:
            assert e.get(1,0)==pytest.approx(.0625,abs=1e-13)
        finally:
            e.close()


@pytest.mark.parametrize("q", [0, .02, .5, .73, 1])
def test_external_exact_radix_selection_is_not_approximate(tmp_path, q):
    values = np.random.default_rng(51).normal(size=1100)
    values[:8] = [-0., 0., -1e300, 1e300, -5e-324, 5e-324, -1., 1.]
    spool = np.r_[values, np.full(13, np.inf)]
    expected = np.quantile(values, q)
    actual = external_quantile(spool, len(values), q, np, np.asarray, 16384, lambda:None)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)


def test_finite_quantile_axes_and_all_invalid():
    a = np.arange(24, dtype=float).reshape(2, 3, 4)
    a[:, 1] = np.nan
    a[1, 2, 3] = np.inf
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanquantile(np.where(np.isfinite(a), a, np.nan), .3, axis=(0, 2))
    np.testing.assert_allclose(finite_quantile(a, .3, (0, 2)), expected, equal_nan=True)


@pytest.mark.parametrize("dtype",[np.float64,np.uint16])
def test_mean_window_updates_only_entering_and_leaving_samples(tmp_path,dtype):
    values = np.arange(30*2*3, dtype=dtype).reshape(30, 2, 3)
    if dtype==np.float64:
        values[10, 1, 1] = np.nan
    e = make(tmp_path, values, [node("time_smooth", window=11)], False)
    try:
        for j in range(e.output.length):
            np.testing.assert_allclose(e.get(1, j), np.nanmean(values[j:j+11], axis=0))
        assert len(e.source.reader.reads) == 11+2*19
    finally:
        e.close()


def test_quantile_window_disk_ring_reads_entering_samples_and_cleans_up(tmp_path):
    values = np.arange(19*2*3, dtype=float).reshape(19, 2, 3)
    values[2, 1, 1] = np.nan
    e = make(tmp_path, values, [node("time_smooth", window=5, mode="quantile", quantile=.31)], False)
    try:
        for j in range(e.output.length):
            np.testing.assert_allclose(e.get(1, j), np.nanquantile(values[j:j+5], .31, axis=0))
        assert len(e.source.reader.reads) == 19
        e.context.cancelled = lambda:True
        with pytest.raises(Cancelled):
            e.get(1, 0)
    finally:
        e.close()
    assert not list(tmp_path.glob("window-ring-*"))


@pytest.mark.parametrize("mode", ["mean", "quantile"])
def test_gpu_window_reuse_and_random_seek(tmp_path, device, mode):
    values = np.arange(19*3*4, dtype=float).reshape(19, 3, 4)
    values[2, 1, 1] = np.nan
    compare(tmp_path, values, [node("time_smooth", window=5, mode=mode, quantile=.31)], indices=[0,1,2,10,3,14])


def test_gpu_disk_window_quantile_path(tmp_path, device):
    values = np.random.default_rng(4).normal(size=(19, 8, 8))
    values[1:4, 2, 1] = np.nan
    e = make(tmp_path, values, [node("time_smooth", window=7, mode="quantile", quantile=.37)], True)
    e.device.scratch = lambda i:16384
    try:
        for j in [0,1,2,10,4]:
            np.testing.assert_allclose(e.get(1,j), np.nanquantile(values[j:j+7], .37, axis=0), equal_nan=True)
        assert e.mappings[e.nodes[0]["id"]]["window_storage"] == "disk_ring_gpu_blocks"
    finally:
        e.close()
    assert not list(tmp_path.glob("window-ring-*"))


def test_gpu_histogram_auto_bounds_and_last_edge(tmp_path, device):
    values = np.arange(5*3*4, dtype=float).reshape(5,3,4)
    values[1,1,1] = np.nan
    compare(tmp_path, values, [node("gain_offset", gain=.7), node("distribution", minimum=None, maximum=None, bins=7, output="main")])


def test_gpu_histogram_exact_boundaries_and_neighbours(tmp_path,device):
    edges=np.linspace(-3.1,7.3,8)
    row=np.r_[edges,np.nextafter(edges,-np.inf),np.nextafter(edges,np.inf),np.nan,np.inf,-np.inf]
    values=np.tile(row,(37,1,1))
    compare(tmp_path,values,[node("distribution",minimum=-3.1,maximum=7.3,bins=7,output="main")])


def test_calibration_alignment_interpolation_and_line_extraction(tmp_path, device):
    values = np.arange(7*4*8, dtype=np.uint16).reshape(7,4,8)+1
    lut = np.arange(512, dtype=float)**1.1
    path = tmp_path/"lut.npy"; np.save(path, lut)
    calibration = node("calibration", lut_path=str(path), black_left=1, black_right=2, saturation_code=210)
    alignment = node("alignment", dx=.25, dy=.2, interpolation="bilinear")
    nodes = [calibration, alignment, node("time_bin", size=2), node("time_statistics", mode="variance")]
    compare(tmp_path, values, nodes)
    line = node("line_extract", mode="segment", x0=0, y0=0, x1=7, y1=3, samples=17, interpolation="bilinear", output="main")
    compare(tmp_path, values, [line, node("axis_slice", axis="T", index=2, output="main")])


@pytest.mark.parametrize("lag", [-1, 0, 1])
def test_g2_device_chain_and_following_window(tmp_path, device, lag):
    values = np.arange(21*4*8, dtype=float).reshape(21,4,8)+10
    values[3,1,6] = np.nan
    nodes = [node("calibration", curve="linear", input_max=1000), node("alignment"),
             node("correlation_map", window=3, map_lag=lag, stride=1), node("time_smooth", window=3)]
    compare(tmp_path, values, nodes)


def test_device_global_statistic_binding_and_side_delivery(tmp_path, device):
    values = np.arange(9*4*8, dtype=float).reshape(9,4,8)+1
    stat = node("global_statistics", mode="max", output="side")
    gain = node("gain_offset", gain={"ref":dict(node_id=stat["id"], port="value")})
    e = make(tmp_path, values, [stat,gain], True)
    try:
        np.testing.assert_allclose(e.get(2,0), values[0]*values.max())
        side = e.side_outputs()[0]
        assert e.get_side_output(side["id"],0) == values.max()
    finally:
        e.close()


def test_device_budget_is_checked_before_large_geometry(tmp_path, device):
    values = np.zeros((1,256,512), dtype=np.uint16)
    alignment = new_node("alignment", shape=(256,512))
    with pytest.raises(ValueError, match="显存预算"):
        make(tmp_path, values, [alignment], True, resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=16,backend="cpu"))


def test_saved_gpu_pipeline_and_side_output_preserve_numeric_provenance(tmp_path,device):
    import json
    import tifffile
    from dataclasses import asdict
    from eip.sequence_processing import process_sequence_job
    from eip.readers import open_source
    values=(np.arange(21*4*8,dtype=np.uint16).reshape(21,4,8)+10)
    path=tmp_path/"raw.tif"
    tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    source=SourceSpec(str(path),"image",.01)
    reader=open_source(source)
    revision=reader.meta.revision; reader.close()
    nodes=[node("gain_offset",gain=.5),node("time_smooth",window=5),
           node("data_tap",destination="file",export_format="dataset"),node("global_statistics",mode="variance")]
    for n in nodes:
        n["execution"]={"backend":"gpu","precision":"float64"}
    nodes[2]["execution"]["precision"]="native"
    payload=dict(id="emulated-device",display_name="Device schedule verification",mode="run",steps=nodes,
        sources=[asdict(source)],revisions=[revision],output_dir=str(tmp_path/"outputs"),seed=0,contract_version=3,
        resources=dict(memory_mb=256,preview_mb=16,cache_mb=0,gpu_mb=128,backend="cpu"))
    events=[]
    process_sequence_job(payload,events.append,lambda:False)
    completed=next(event for event in events if event["type"]=="sequence_completed")
    assert completed["run"]["status"]=="succeeded",completed["run"]
    part=tmp_path/"outputs/run_emulated-device/source_000"
    smoothed=np.array([values[i:i+5].mean(axis=0)*.5 for i in range(17)])
    np.testing.assert_allclose(np.load(part/"values.npy"),[smoothed.var()])
    np.testing.assert_allclose(np.load(part/"side_000/values.npy"),smoothed)
    run=json.loads((part.parent/"run.json").read_text(encoding="utf-8"))
    from eip import __version__
    assert run["app_version"]==__version__
    trace=run["results"][0]["modules"]["device_execution"]
    assert trace["intermediate_residency"] and trace["largest_upload_batch"]>1
    assert trace["cache_peak_bytes"]<=trace["cache_budget_bytes"]
    ref=json.loads((part/"references.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert ref["time"] is None and ref["contribution"]["upstream_stop"]==17
    np.testing.assert_array_equal(tifffile.imread(path),values)


def test_real_cuda_extended_pipeline_when_available(tmp_path):
    from eip.modes.split_correlation.backend import devices
    if not devices()["gpu"]:
        pytest.skip("无可用 CUDA；其余测试仅验证设备调度和参考数学，不代表 CUDA 真机实测")
    values = np.arange(21*4*8, dtype=float).reshape(21,4,8)+10
    values[3,1,6] = np.nan
    for steps in ([node("time_smooth", window=5, mode="quantile"), node("time_statistics", mode="quantile")],
                  [node("global_statistics", scope="samples", mode="variance")],
                  [node("global_statistics", scope="all", mode="quantile")],
                  [node("calibration", curve="linear", input_max=1000),node("alignment", dx=.2,dy=.2,interpolation="bilinear"),
                   node("correlation_map",window=3,stride=1,map_lag=-1),node("time_smooth",window=3)],
                  [node("distribution", minimum=None, maximum=None, bins=9, output="main")]):
        compare(tmp_path, values, steps)
    from eip.modes.split_correlation.backend import Backend
    backend=Backend("gpu",64*1024**2)
    try:
        row=np.random.default_rng(5).normal(size=1100)
        row[:4]=[-1e300,1e300,-5e-324,5e-324]
        actual=external_quantile(row,len(row),.73,backend.xp,backend.host,16384,lambda:None)
        np.testing.assert_allclose(actual,np.quantile(row,.73),rtol=1e-12)
    finally:
        backend.close()
