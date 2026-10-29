from dataclasses import asdict
import json

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.performance import RunProfiler, progress_metadata
from eip.readers import open_source
from eip.sequence_processing import process_sequence_job


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_nested_host_categories_are_exclusive_and_finish_is_frozen():
    clock = Clock()
    profiler = RunProfiler(clock=clock)
    clock.advance(1)
    with profiler.measure("cpu_compute"):
        clock.advance(2)
        with profiler.measure("input_read"):
            clock.advance(3)
        with profiler.measure("cpu_compute"):
            clock.advance(4)
    with profiler.measure("gpu_submit"):
        clock.advance(5)
        with profiler.measure("download"):
            clock.advance(6)
    result = profiler.finish()
    assert result["elapsed_seconds"] == 21
    assert result["host_seconds"]["cpu_compute"] == 6
    assert result["host_seconds"]["gpu_submit"] == 5
    assert result["host_seconds"]["download"] == 6
    assert sum(result["host_seconds"].values()) == result["elapsed_seconds"]
    assert result["gpu_execution_seconds"] is None
    clock.advance(100)
    assert profiler.snapshot() == result


def test_reads_count_repetition_and_decoded_bytes_not_unique_frames():
    clock = Clock()
    profiler = RunProfiler(clock=clock)
    profiler.add_input(12)
    profiler.add_input(12)
    profiler.add_output(8)
    clock.advance(2)
    result = profiler.finish()
    assert result["input_read_samples"] == 2
    assert result["input_read_bytes"] == 24
    assert result["input_reads_per_second"] == 1
    assert result["output_per_second"] == .5
    assert "not unique" in result["semantics"]["input_read_samples"]
    assert "not physical" in result["semantics"]["input_read_bytes"]


@pytest.mark.parametrize("axis,unit",[("T","样本"),("W","窗长"),("Q","延迟点"),("N","记录")])
def test_progress_axis_is_not_mislabeled_as_frame(axis,unit):
    assert progress_metadata([dict(contract=dict(sequence_axis=axis))]) == dict(progress_axis=axis,progress_unit=unit)
    assert progress_metadata([dict(progress_axis=axis)]) == dict(progress_axis=axis,progress_unit=unit)


def test_old_and_mixed_axis_records_do_not_invent_frame_units():
    assert progress_metadata([{}])["progress_unit"] == "输出"
    assert progress_metadata([dict(progress_axis="W"),dict(progress_axis="Q")])["progress_axis"] is None


def payload(tmp_path,nodes):
    values=np.arange(21*2*4,dtype=np.uint16).reshape(21,2,4)+10
    path=tmp_path/"raw.tif"
    tifffile.imwrite(path,values,photometric="minisblack",metadata={"axes":"TYX"})
    spec=SourceSpec(str(path),interval=.01)
    reader=open_source(spec)
    revision=reader.meta.revision
    reader.close()
    return dict(id="performance",display_name="performance",mode="run",output_dir=str(tmp_path/"outputs"),sources=[asdict(spec)],revisions=[revision],steps=nodes)


def test_window_scan_persists_source_performance_and_accurate_units(tmp_path):
    alignment=new_node("alignment")
    alignment["parameters"].update(left=dict(x=0,y=0,width=2,height=2),right=dict(x=2,y=0,width=2,height=2),midline=[2,2])
    scan=new_node("correlation_window")
    scan["parameters"].update(window_list=[2,3,7],lag=0,center=10)
    events=[]
    process_sequence_job(payload(tmp_path,[alignment,new_node("point_extract"),scan]),events.append,lambda:False)
    completed=next(e for e in events if e["type"]=="sequence_completed")
    assert completed["run"]["status"] == "succeeded",completed["run"]
    run=json.loads((tmp_path/"outputs/run_performance/run.json").read_text(encoding="utf-8"))
    item=run["results"][0]
    assert item["progress_axis"] == "W"
    assert item["progress_unit"] == "窗长"
    assert item["completed_samples"] == item["completed_frames"] == 3
    stats=item["performance"]
    assert stats["output_samples"] == 3
    assert stats["input_read_samples"] > 0
    assert stats["output_bytes"] == 24
    assert stats["gpu_execution_seconds"] is None
    assert sum(stats["host_seconds"].values()) == pytest.approx(stats["elapsed_seconds"])
    progress=[e for e in events if e["type"]=="progress"]
    assert progress[-1]["completed"] == 3
    assert all(e["progress_unit"]=="窗长" for e in progress)


@pytest.mark.parametrize("cancel",[False,True])
def test_failed_and_cancelled_sources_preserve_partial_metrics(tmp_path,monkeypatch,cancel):
    import eip.sequence_processing as processing
    original=processing.SequenceEngine.iter_outputs
    count=0
    stop=False

    def interrupted(engine,indices):
        nonlocal count,stop
        for index,value in original(engine,indices):
            count+=1
            if count==2:
                if cancel:
                    stop=True
                    engine.context.check_cancelled()
                raise ValueError("intentional source failure")
            yield index,value

    monkeypatch.setattr(processing.SequenceEngine,"iter_outputs",interrupted)
    events=[]
    processing.process_sequence_job(payload(tmp_path,[new_node("gain_offset")]),events.append,lambda:stop)
    run=json.loads((tmp_path/"outputs/run_performance.partial/run.json").read_text(encoding="utf-8"))
    assert run["status"] == ("cancelled" if cancel else "failed")
    item=run["results"][0]
    assert item["completed_samples"] == 1
    assert item["performance"]["output_samples"] == 1
    assert item["performance"]["elapsed_seconds"] > 0
    if cancel:
        assert json.loads(json.dumps(next(e for e in events if e["type"]=="cancelled")["run"])) == run
