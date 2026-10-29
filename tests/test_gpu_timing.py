"""Bounded asynchronous instrumentation and real CUDA numerical integration."""
from types import SimpleNamespace
from dataclasses import asdict
import json

import numpy as np
import pytest

from eip.gpu_timing import GpuEventTimer, KERNEL_NAMES
from eip.performance import RunProfiler


class FakeCuda:
    def __init__(self):
        self.time = self.completed = self.synchronizations = self.records = 0
        self.fail_record = False
        self.stream = SimpleNamespace(ptr=1)

    def Event(self):
        owner = self

        class Event:
            def record(self, stream):
                if owner.fail_record:
                    raise RuntimeError("event recording unavailable")
                self.time = owner.time
                owner.records += 1

            @property
            def done(self):
                return self.time <= owner.completed

            def synchronize(self):
                owner.synchronizations += 1
                owner.completed = self.time

        return Event()

    def get_current_stream(self):
        return self.stream

    def get_elapsed_time(self, start, stop):
        return stop.time-start.time


def fake_timer(**bounds):
    cuda = FakeCuda()
    return cuda,GpuEventTimer(SimpleNamespace(cuda=cuda),**bounds)


def test_event_pool_is_bounded_and_never_synchronizes_during_launches():
    cuda,timer = fake_timer(max_pending=2)
    calls = []

    def kernel(value):
        calls.append(value)
        cuda.time += 5
        return value+1

    for index in range(5):
        assert timer.launch(KERNEL_NAMES[0],kernel,index) == index+1
    before = timer.snapshot()
    assert calls == list(range(5))  # No extra warmup numerical execution.
    assert cuda.synchronizations == 0
    assert before["warmup_excluded_calls"] == 1
    assert before["capacity_excluded_calls"] == 2
    assert before["pending_calls"] == before["allocated_event_pairs"] == 2
    assert before["device_seconds"] is None
    timer.finish()
    after = timer.snapshot()
    assert cuda.synchronizations == 1
    assert after["timed_calls"] == 2
    assert after["device_seconds"] == pytest.approx(.01)
    assert after["pending_calls"] == 0
    assert after["status"] == "finalized"
    assert timer.snapshot() == after


def test_layout_warm_registry_is_bounded_and_handles_strides():
    cuda,timer = fake_timer(max_signatures=1)
    kernel = lambda value: value
    array = np.zeros((2,3))
    timer.launch(KERNEL_NAMES[0],kernel,array)
    timer.launch(KERNEL_NAMES[0],kernel,array)
    timer.launch(KERNEL_NAMES[0],kernel,array[:,::-1])
    timer.launch(KERNEL_NAMES[0],kernel,np.zeros((3,2)))
    timer.finish()
    result = timer.snapshot()
    assert result["warmed_signatures"] == 1
    assert result["signature_excluded_calls"] == 2
    assert result["warmup_excluded_calls"] == result["timed_calls"] == 1


def test_sampling_is_explicit_and_does_not_extrapolate_unmeasured_calls():
    cuda,timer = fake_timer(max_pending=16)
    calls = []

    def kernel(value):
        calls.append(value)
        cuda.time += 1

    for index in range(100):
        timer.launch(KERNEL_NAMES[0],kernel,index)
    timer.finish()
    result = timer.snapshot()
    assert calls == list(range(100))
    assert result["initial_calls_per_kernel"] == 8
    assert result["sample_every"] == 32
    assert result["sampling_excluded_calls"] == 90
    assert result["timed_calls"] == 9
    assert result["warmup_excluded_calls"] == 1
    assert result["device_seconds"] == pytest.approx(.009)


def test_monitoring_failure_does_not_repeat_or_suppress_numerical_work():
    cuda,timer = fake_timer()
    calls = []
    kernel = lambda value: calls.append(value) or value+1
    timer.launch(KERNEL_NAMES[0],kernel,0)
    cuda.fail_record = True
    assert timer.launch(KERNEL_NAMES[0],kernel,1) == 2
    assert timer.launch(KERNEL_NAMES[0],kernel,2) == 3
    timer.finish()
    result = timer.snapshot()
    assert calls == [0,1,2]
    assert result["status"] == "error"
    assert result["monitoring_excluded_calls"] == 2
    assert result["failed_calls"] == 0
    assert "unavailable" in result["error"]


def test_numerical_error_propagates_once_even_when_event_pool_is_full():
    cuda,timer = fake_timer(max_pending=1)
    calls = []

    def kernel(value):
        calls.append(value)
        cuda.time += 1
        if value == 2:
            raise ValueError("actual kernel failure")
        return value

    timer.launch(KERNEL_NAMES[0],kernel,0)
    timer.launch(KERNEL_NAMES[0],kernel,1)
    with pytest.raises(ValueError,match="actual kernel failure"):
        timer.launch(KERNEL_NAMES[0],kernel,2)
    timer.finish()
    assert calls == [0,1,2]
    assert timer.snapshot()["failed_calls"] == 1
    assert timer.snapshot()["timed_calls"] == 1


def test_nested_launch_is_not_double_counted():
    cuda,timer = fake_timer()

    def inner(value):
        cuda.time += 2
        return value

    def outer(value):
        return timer.launch(KERNEL_NAMES[2],inner,value)

    timer.launch(KERNEL_NAMES[0],outer,0)
    timer.launch(KERNEL_NAMES[0],outer,1)
    timer.finish()
    result = timer.snapshot()
    assert result["timed_calls"] == 1
    assert result["nested_excluded_calls"] == 2
    assert result["device_seconds"] == pytest.approx(.002)


def test_cpu_and_fake_gpu_adapters_have_no_device_measurement():
    profiler = RunProfiler()
    assert profiler.gpu_timer(np) is None
    result = profiler.finish()
    assert result["gpu_timing"] is None
    assert result["gpu_device_seconds"] is None
    assert result["gpu_execution_seconds"] is None


def test_real_cuda_fused_launch_coverage_and_float64_results():
    from eip.modes.split_correlation.backend import Backend,devices
    from eip.modes.split_correlation.correlation import RollingMoments
    if not devices()["gpu"]:
        pytest.skip("需要真实 CuPy / CUDA 设备")
    backend = Backend("gpu",64*1024**2)
    profiler = RunProfiler()
    backend.gpu_timer = profiler.gpu_timer(backend.xp)
    config = SimpleNamespace(min_mean=1e-9,min_samples=2)
    baseline = np.full((2,7),10.)
    cpu = RollingMoments(baseline,config,SimpleNamespace(xp=np,host=np.asarray))
    gpu = RollingMoments(baseline,config,backend)
    cp = backend.xp
    left = cp.arange(35,dtype=cp.float64).reshape(5,7)+11
    right = left+3
    try:
        for _ in range(3):
            cpu.add_many(cp.asnumpy(left),cp.asnumpy(right))
            gpu.add_many(left,right)
            cpu.add(cp.asnumpy(left[0]),cp.asnumpy(right[0]),-1)
            gpu.add(left[0],right[0],-1)
            actual,count = gpu.result()
            expected,n = cpu.result()
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(count,n)
            np.testing.assert_array_equal(cp.asnumpy(gpu.result(device=True,count=False)[0]),expected)
        result = profiler.finish()
        coverage = result["gpu_timing"]
        assert result["gpu_device_seconds"] is not None,coverage["error"]
        assert result["gpu_device_seconds"] > 0
        assert result["gpu_execution_seconds"] is None
        assert coverage["status"] == "finalized" and coverage["pending_calls"] == 0
        assert coverage["calls"] == 12
        assert coverage["timed_calls"] == 8
        assert coverage["warmup_excluded_calls"] == 4
        assert all(row["timed_calls"] == 2 for row in coverage["kernels"].values() if row["calls"])
        assert coverage["kernels"]["eip_g2_sequence_block_float64"]["calls"] == 0
        assert sum(result["host_seconds"].values()) == pytest.approx(result["elapsed_seconds"])
        assert profiler.snapshot() == result
    finally:
        backend.close()


def cuda_payload(tmp_path):
    import tifffile
    from eip.domain import SourceSpec
    from eip.module_registry import new_node
    from eip.readers import open_source
    from eip.modes.split_correlation.backend import devices
    if not devices()["gpu"]:
        pytest.skip("需要真实 CuPy / CUDA 设备")
    path = tmp_path/"raw.tif"
    data = np.arange(21*2*4,dtype=np.uint16).reshape(21,2,4)+10
    tifffile.imwrite(path,data,photometric="minisblack",metadata={"axes":"TYX"})
    source = SourceSpec(str(path),interval=.01)
    reader = open_source(source)
    revision = reader.meta.revision
    reader.close()
    alignment = new_node("alignment")
    alignment["parameters"].update(left=dict(x=0,y=0,width=2,height=2),
                                    right=dict(x=2,y=0,width=2,height=2),midline=[2,2])
    scan = new_node("correlation_window")
    scan["parameters"].update(window_list=[2,3,5,7],lag=0,center=10)
    scan["execution"] = dict(backend="gpu")
    return dict(id="device-timing",display_name="device timing",mode="run",
                output_dir=str(tmp_path/"outputs"),sources=[asdict(source)],
                revisions=[revision],steps=[alignment,new_node("point_extract"),scan])


@pytest.mark.parametrize("interrupt",[None,"fail","cancel"])
def test_worker_persists_cuda_coverage_on_success_failure_and_cancel(tmp_path,monkeypatch,interrupt):
    import eip.sequence_processing as processing
    payload = cuda_payload(tmp_path)
    original = processing.SequenceEngine.get
    stop = False

    def get(engine,stage,index):
        nonlocal stop
        if interrupt and stage == len(engine.nodes) and index == 2:
            if interrupt == "cancel":
                stop = True
                engine.context.check_cancelled()
            raise ValueError("deliberate interruption")
        return original(engine,stage,index)

    monkeypatch.setattr(processing.SequenceEngine,"get",get)
    processing.process_sequence_job(payload,lambda event:None,lambda:stop)
    folder = "run_device-timing"+(".partial" if interrupt else "")
    run = json.loads((tmp_path/"outputs"/folder/"run.json").read_text(encoding="utf-8"))
    assert run["status"] == {None:"succeeded","fail":"failed","cancel":"cancelled"}[interrupt]
    stats = run["results"][0]["performance"]
    assert stats["schema_version"] == 2
    assert stats["gpu_execution_seconds"] is None
    assert stats["gpu_device_seconds"] > 0
    assert stats["gpu_timing"]["status"] == "finalized"
    assert stats["gpu_timing"]["pending_calls"] == 0
    assert sum(stats["host_seconds"].values()) == pytest.approx(stats["elapsed_seconds"])


def test_cleanup_disk_error_still_closes_engine_and_finalizes_cuda_events(tmp_path,monkeypatch):
    import eip.sequence_processing as processing
    payload = cuda_payload(tmp_path)
    closed = []
    original_close,original_write = processing.SequenceEngine.close,processing.write_json

    def side_failure(engine,directory,payload,item,context,emit):
        (directory/"side_000").mkdir()
        item["side_outputs"] = [dict(status="succeeded",output_directory="side_000")]
        raise ValueError("injected source failure")

    def disk_error(path,value):
        if path.name == "delivery.json":
            raise OSError("injected cleanup disk failure")
        original_write(path,value)

    def close(engine):
        original_close(engine)
        closed.append(engine)

    monkeypatch.setattr(processing,"persist_side_outputs",side_failure)
    monkeypatch.setattr(processing,"write_json",disk_error)
    monkeypatch.setattr(processing.SequenceEngine,"close",close)
    with pytest.raises(OSError,match="cleanup disk failure"):
        processing.process_sequence_job(payload,lambda event:None,lambda:False)
    assert len(closed) == 1 and not closed[0].device_backends
    run = json.loads((tmp_path/"outputs/run_device-timing.partial/run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"
    stats = run["results"][0]["performance"]
    assert stats["gpu_timing"]["status"] == "finalized"
    assert stats["gpu_timing"]["pending_calls"] == 0
    assert stats["gpu_device_seconds"] > 0
