"""Independent Q provenance covers signed lag, actual clocks and unknowns."""
from dataclasses import asdict
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.reference_index import WindowTimeIndex
from eip.sequence_processing import process_sequence_job


TOOL = Path(__file__).resolve().parents[1]/"tools"/"direct_lag_references.py"
_spec = importlib.util.spec_from_file_location("direct_lag_references", TOOL)
exports = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exports)


def flow():
    nodes = [new_node(name) for name in exports.ALGORITHMS]
    nodes[2]["parameters"].update(left=dict(x=0,y=0,width=4,height=3),
                                right=dict(x=4,y=0,width=4,height=3), midline=[4,4])
    nodes[3]["parameters"].update(x=1,y=1,output="main")
    nodes[4]["parameters"].update(window=3,min_samples=2,lag_list=[-2,0,3],analysis="curve")
    return nodes


@pytest.mark.parametrize("timestamps", [True, False])
def test_direct_q_metadata_equal_workbench_without_pixel_replay(tmp_path,cine_factory,monkeypatch,timestamps):
    pixels = (np.arange(13*3*8,dtype=np.uint16).reshape(13,3,8)+10)*2
    # Real indexed ticks are deliberately irregular: no nominal-rate shortcut.
    ticks = [(1_700_000_000 << 32)+sum([0,400,800,1600,400,800,1200,1600,800,400,1200,800,400][:i+1])
             for i in range(len(pixels))]
    path,*_ = cine_factory(pixels=pixels,timestamps=timestamps,ticks=ticks)
    source = SourceSpec(str(path))
    reader = open_source(source)
    payload = dict(id="qrefs",display_name="qrefs",mode="run",contract_version=3,
                   sources=[asdict(source)],revisions=[reader.meta.revision],steps=flow(),
                   resources=dict(backend="cpu",memory_mb=256,cache_mb=32,gpu_mb=64),
                   output_dir=str(tmp_path/"workbench"))
    reader.close()
    process_sequence_job(payload,lambda event:None,lambda:False)
    baseline = tmp_path/"workbench"/"run_qrefs"/"source_000"
    result = json.loads((baseline.parent/"run.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded",result
    # Values are copied only in this export-unit test; real benchmark computes
    # its own values. This isolates independent source / clock reconstruction.
    values = np.load(baseline/"values.npy",allow_pickle=False)
    output = tmp_path/"direct"
    output.mkdir()
    np.save(output/"values.npy",values,allow_pickle=False)
    reader = open_source(source)
    monkeypatch.setattr(reader,"read_frame",lambda index:pytest.fail("metadata must not decode pixels"))
    try:
        report = exports.write_exports(output,payload,reader,values,make_plots=False)
    finally:
        reader.close()
    assert report["metadata_pixel_reads"] == 0
    assert report["center"] == 5
    assert report["left_window"] == [4,7]
    assert report["contract"] == result["results"][0]["contract"]
    assert report["contract"]["origin"] == [0,0]
    assert report["reference_metrics"]["pair_elements"] == 9
    for name in ("references.jsonl", "measurements.csv"):
        assert (output/name).read_bytes() == (baseline/name).read_bytes()
    np.testing.assert_array_equal(np.load(output/"reference_offsets.npy"),np.load(baseline/"reference_offsets.npy"))
    refs = [json.loads(line) for line in (output/"references.jsonl").read_text().splitlines()]
    assert [r["correlation"]["lag_samples"] for r in refs] == [-2,0,3]
    if timestamps:
        assert refs[0]["correlation"]["lag_seconds"] < 0
        assert refs[-1]["correlation"]["lag_seconds"] > 0
        assert refs[0]["correlation"]["lag_seconds_min"] != refs[0]["correlation"]["lag_seconds_max"]
    else:
        assert all(r["correlation"]["known_pairs"] == 0 for r in refs)


def test_partly_unknown_times_do_not_become_known_or_depend_on_intensity():
    times = np.array([0., .1, None, .5, .9, 1.5, 2.4, 3.2],dtype=object)
    summaries,metrics = exports.summarize_pair_times(times,2,5,[-2,0,2])
    assert summaries[0] == dict(lag_seconds=None,lag_seconds_min=-.4,lag_seconds_max=-.4,known_pairs=1)
    assert summaries[1] == dict(lag_seconds=None,lag_seconds_min=0.,lag_seconds_max=0.,known_pairs=2)
    assert summaries[2] == dict(lag_seconds=None,lag_seconds_min=1.,lag_seconds_max=1.5,known_pairs=2)
    assert metrics["pair_elements"] == 9 and metrics["pixel_reads"] == 0
    assert metrics["max_scratch_bytes"] <= 17*3


@pytest.mark.parametrize("times", [
    [0.,None,np.inf,2.,3.,4.,5.],
    [0.,None,np.nan,2.,3.,4.,5.],
    [0.,1.,2.,3.,np.inf,np.nan,5.],
])
def test_nonfinite_time_fallback_preserves_legacy_order_sensitive_rules(times):
    for lag in (-1,0,1):
        summaries,metrics = exports.summarize_pair_times(np.array(times,dtype=object),1,6,[lag])
        tree = WindowTimeIndex(5,4096,{})
        with np.errstate(invalid="ignore"):
            total,minimum,maximum,known = tree.summary(1,6,lag,lambda j:times[j],lambda:None)
        expected = np.array([total/5 if known == 5 else np.nan,
                             minimum if minimum is not None else np.nan,
                             maximum if maximum is not None else np.nan])
        summary = summaries[0]
        actual = np.array([summary["lag_seconds"] if summary["lag_seconds"] is not None else np.nan,
                           summary["lag_seconds_min"] if summary["lag_seconds_min"] is not None else np.nan,
                           summary["lag_seconds_max"] if summary["lag_seconds_max"] is not None else np.nan])
        np.testing.assert_equal(actual,expected)
        assert summary["known_pairs"] == known
        assert metrics["nonfinite_fallback_windows"] == 1
        tree.close()


def test_summaries_cancel_between_lags_and_reject_out_of_bounds():
    calls = []
    def check():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("cancel")
    with pytest.raises(RuntimeError,match="cancel"):
        exports.summarize_pair_times(np.arange(10,dtype=float),3,6,[-1,0,1],check=check)
    with pytest.raises(ValueError,match="outside"):
        exports.summarize_pair_times(np.arange(10,dtype=float),1,4,[-2])


def test_lag_layout_includes_zero_and_keeps_explicit_center_and_order():
    p = dict(window=3,lag_min=-3,lag_max=3,lag_step=2,lag_list=None,center=None)
    layout = exports.scan_layout(13,p)
    assert layout == dict(center=6,start=5,stop=8,window=3,lags=[-3,-1,0,1,3])
    p["center"] = 6
    layout = exports.scan_layout(13,p,lags=[3,-2,3])
    assert layout["center"] == 5 and layout["lags"] == [3,-2,3]
    with pytest.raises(ValueError,match="Center"):
        exports.scan_layout(13,p,center=0)


def test_export_rejects_q_contract_mislabeled_as_time(tmp_path):
    reader = SimpleNamespace(meta=SimpleNamespace(frame_count=13),times=None,ticks=None,
                             first_number=-2,trigger_ticks=0)
    payload = dict(id="unknown",steps=flow(),sources=[dict(path="fake.cine",kind="auto",interval=.01)])
    values = np.array([np.nan,1.2,np.inf],dtype=np.float64)
    np.save(tmp_path/"values.npy",values,allow_pickle=False)
    wrong = dict(exports.output_contract(3),sequence_axis="T")
    with pytest.raises(ValueError,match="float64"):
        exports.write_exports(tmp_path,payload,reader,values,contract=wrong,make_plots=False)
    report = exports.write_exports(tmp_path,payload,reader,values,make_plots=False)
    assert report["reference_metrics"]["time_source"] == "indexed metadata; no nominal FPS"
    refs = [json.loads(line) for line in (tmp_path/"references.jsonl").read_text().splitlines()]
    assert all(row["time"] is None and row["correlation"]["known_pairs"] == 0 for row in refs)
    with (tmp_path/"measurements.csv").open(encoding="utf-8-sig",newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["finite_count"] == "0" and rows[0]["min"] == ""
    assert rows[1]["finite_count"] == "1" and rows[1]["value_1"] == "1.2"
    assert rows[2]["invalid_count"] == "1" and rows[2]["max"] == ""
