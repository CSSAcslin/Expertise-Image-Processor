"""Independent experiment validation: signed Q, finite pairs and ordered sums."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import tifffile

from eip.domain import SourceSpec
from eip.module_registry import new_node
from eip.readers import open_source
from eip.sequence_processing import process_sequence_job


path = Path(__file__).resolve().parents[1]/"tools/direct_lag_scan.py"
module_spec = importlib.util.spec_from_file_location("dedicated_lag_experiment",path)
direct = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(direct)


def parameters(**overrides):
    return dict(window=31,min_mean=1e-9,min_samples=20,lag_min=-11,lag_max=17,
                lag_step=4,lag_list=None,center=None,analysis="curve",**overrides)


def oracle(signals,p,plan):
    """Scalar-by-scalar oracle; does not reuse experiment summation helpers."""
    base = [float(v) if np.isfinite(v) else 0. for v in signals[plan["center"]]]
    result,counts = [],[]
    for lag in plan["lags"]:
        sl=sr=slr=0.
        n=0
        for index in range(plan["start"],plan["stop"]):
            l,r = [float(v) for v in (signals[index,0],signals[index+lag,1])]
            valid = np.isfinite(l) and np.isfinite(r)
            dl,dr = (l-base[0],r-base[1]) if valid else (0.,0.)
            sl += dl
            sr += dr
            slr += dl*dr
            n += int(valid)
        ml,mr = sl/max(n,1),sr/max(n,1)
        il,ir = base[0]+ml,base[1]+mr
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            value = float(1+np.float64(slr/max(n,1)-ml*mr)/(np.float64(il)*ir))
        result.append(value if n>=p["min_samples"] and il>p["min_mean"] and ir>p["min_mean"] and np.isfinite(value) else np.nan)
        counts.append(n)
    return np.asarray(result),np.asarray(counts,dtype=np.uint32)


@pytest.mark.parametrize("invalid",[False,True])
@pytest.mark.parametrize("baseline_nonfinite",[False,True])
def test_cpu_order_matches_scalar_oracle_and_independent_formula(invalid,baseline_nonfinite):
    rng=np.random.default_rng(527)
    signals=np.ascontiguousarray(rng.uniform(.02,4,(217,2)))
    p=parameters()
    plan=direct.scan_layout(len(signals),p)
    if invalid:
        signals[[11,71,81,92,104,117],0]=np.nan
        signals[[7,77,87,97,107],1]=np.inf
    if baseline_nonfinite:
        signals[plan["center"]]=[np.nan,-np.inf]
    actual,count=direct.compute_cpu(signals,p)
    expected,expected_count=oracle(signals,p,plan)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(count,expected_count)
    formula=[]
    for lag in plan["lags"]:
        left=signals[plan["start"]:plan["stop"],0]
        right=signals[plan["start"]+lag:plan["stop"]+lag,1]
        good=np.isfinite(left)&np.isfinite(right)
        formula.append(np.mean(left[good]*right[good])/(np.mean(left[good])*np.mean(right[good])))
    np.testing.assert_allclose(actual,formula,atol=2e-13,rtol=2e-13)


def test_negative_and_positive_lags_are_independent_and_subset_keeps_full_center():
    rng=np.random.default_rng(188)
    signals=np.ascontiguousarray(rng.uniform(.1,4,(117,2)))
    p=parameters()
    full=direct.scan_layout(len(signals),p)
    selected=[17,-11,0,17]
    plan=direct.scan_layout(len(signals),p,selected)
    assert plan["center"]==full["center"]
    assert plan["lags"]==selected
    values,counts=direct.compute_cpu(signals,p,lags=selected)
    np.testing.assert_array_equal((values,counts),oracle(signals,p,plan))
    assert values[0]!=values[1]
    assert values[0]==values[3]


def test_recorded_scan_layout_and_bounded_read_union():
    p=parameters()
    p.update(window=10001,min_samples=10000,lag_min=0,lag_max=30000,lag_step=10)
    plan=direct.scan_layout(50654,p)
    assert plan["center"]==10326
    assert plan["start"]==5326 and plan["stop"]==15327
    assert plan["lags"]==list(range(0,30001,10))
    assert plan["needed_intervals"]==[[5326,45327]]
    assert plan["read_intervals"]==[[5312,45344]]
    assert plan["exact_needed_samples"]==40001 and plan["read_samples"]==40032
    sparse=direct.scan_layout(50654,p,[0,30000])
    assert sparse["exact_needed_samples"]==20002
    assert sparse["needed_intervals"]==[[5326,15327],[35326,45327]]


@pytest.mark.parametrize("kind",["all_invalid","zero_mean","too_few"])
def test_invalid_output_rules(kind):
    p=parameters()
    signal=np.ones((117,2),np.float64)
    if kind=="all_invalid":signal[:,1]=np.nan
    if kind=="zero_mean":signal[:,0]=0.
    if kind=="too_few":signal[::2,1]=np.inf
    values,count=direct.compute_cpu(signal,p)
    assert np.isnan(values).all()
    if kind=="all_invalid":assert not count.any()
    if kind=="too_few":assert np.all(count<p["min_samples"])


@pytest.mark.parametrize("change",["short_source","center","lag_bounds","window","dtype","layout"])
def test_experiment_rejects_invalid_plan_and_signal(change):
    p=parameters()
    signal=np.ones((117,2),np.float64)
    if change=="short_source":signal=signal[:35]
    if change=="center":p["center"]=2
    if change=="lag_bounds":p["lag_min"]=30
    if change=="window":p["window"]=30
    if change=="dtype":signal=signal.astype(np.float32)
    if change=="layout":signal=signal[::2]
    with pytest.raises(ValueError):direct.compute_cpu(signal,p)


def fixture_spec(tmp_path):
    rng=np.random.default_rng(991)
    raw=rng.uniform(1,3000,(117,6,12)).astype(np.float64)
    raw[52,2,2]=np.nan
    raw[69,3,9]=np.inf
    path=tmp_path/"pair.tif"
    tifffile.imwrite(path,raw,photometric="minisblack",metadata={"axes":"TYX"})
    source=SourceSpec(str(path),interval=.01)
    reader=open_source(source)
    revision=reader.meta.revision
    reader.close()
    nodes=[new_node(key) for key in ("time_crop","calibration","alignment","point_extract","correlation_curve")]
    nodes[1]["parameters"].update(curve="linear",input_max=4095,black_left=0,black_right=0)
    nodes[2]["parameters"].update(left=dict(x=0,y=0,width=6,height=6),right=dict(x=6,y=0,width=6,height=6),
                                    midline=[6,6],dx=1,dy=1,interpolation="nearest",mirror_x=False,mirror_y=False)
    nodes[3]["parameters"].update(x=2,y=2,output="main")
    nodes[4]["parameters"].update(parameters())
    spec=dict(id="lag-comparison",display_name="lag comparison",mode="run",contract_version=3,sources=[asdict(source)],revisions=[revision],
              steps=nodes,input_kinds=["image"],resources=dict(memory_mb=128,preview_mb=16,cache_mb=0,gpu_mb=64,backend="cpu"))
    return spec,raw


def test_independent_sensor_coordinates_calibration_and_complete_cpu_run(tmp_path):
    spec,raw=fixture_spec(tmp_path)
    signals,times,meta,reader=direct.prepare_signals(spec)
    try:
        assert meta["selected_sensor_coordinates"]==dict(left=[2,2],right=[3,9])
        expected=raw[:,[2,3],[2,9]]/4095
        expected[~np.isfinite(expected)]=np.nan
        indices=meta["read_indices"]
        np.testing.assert_array_equal(signals[indices],expected[indices])
        unused=np.setdiff1d(np.arange(len(raw)),indices)
        assert np.isnan(signals[unused]).all()
        assert meta["calibrated_sensor_values"]==2*len(indices)
    finally:reader.close()
    result=direct.run_direct(spec,tmp_path/"direct",backend="cpu",exports="none",warm_repeats=1)
    actual=np.load(tmp_path/"direct"/"values.npy")
    count=np.load(tmp_path/"direct"/"counts.npy")
    plan=result["plan"]
    np.testing.assert_array_equal((actual,count),oracle(signals,spec["steps"][-1]["parameters"],plan))
    assert result["elapsed_seconds"]>result["phases_seconds"]["correlation"]>0
    assert len(result["numerical_warm_seconds"])==1
    assert result["transfers"]["upload_bytes"]==0


@pytest.mark.parametrize("change",["order","target","precision","calibration","alignment","sampling"])
def test_rejects_unsupported_scientific_changes(tmp_path,change):
    spec,_=fixture_spec(tmp_path)
    spec=deepcopy(spec)
    if change=="order":spec["steps"][1:3]=reversed(spec["steps"][1:3])
    if change=="target":spec["steps"][1]["target"]="left"
    if change=="precision":spec["steps"][1].setdefault("execution",{})["precision"]="float32"
    if change=="calibration":spec["steps"][1]["parameters"]["saturation_code"]=2048
    if change=="alignment":spec["steps"][2]["parameters"]["interpolation"]="bilinear"
    if change=="sampling":spec["steps"][3]["parameters"]["output"]="side"
    with pytest.raises(ValueError):direct.validate_spec(spec)


def test_never_overwrites_existing_output(tmp_path):
    folder=tmp_path/"output"
    folder.mkdir()
    sentinel=folder/"keep"
    sentinel.write_text("unchanged")
    with pytest.raises(ValueError):direct.run_direct({},folder)
    assert sentinel.read_text()=="unchanged"


def test_complete_cine_direct_flow_matches_workbench_values_and_provenance(tmp_path,cine_factory):
    spec,raw=fixture_spec(tmp_path)
    codes=np.nan_to_num(raw,nan=0.,posinf=4095.).astype(np.uint16)
    ticks=[(1_700_000_000<<32)+i*800+i//7*400 for i in range(len(codes))]
    cine,*_=cine_factory(pixels=codes,ticks=ticks,timestamps=True,packed=False,bits=12)
    source=SourceSpec(str(cine))
    reader=open_source(source)
    spec.update(sources=[asdict(source)],revisions=[reader.meta.revision],output_dir=str(tmp_path/"processor"))
    reader.close()
    events=[]
    process_sequence_job(spec,events.append,lambda:False)
    run=next(e["run"] for e in events if e["type"]=="sequence_completed")
    assert run["status"]=="succeeded",run
    expected=tmp_path/"processor"/"run_lag-comparison"/run["results"][0]["output_directory"]
    result=direct.run_direct(spec,tmp_path/"direct-full",backend="cpu",exports="full",warm_repeats=0)
    np.testing.assert_array_equal(np.load(tmp_path/"direct-full"/"values.npy"),np.load(expected/"values.npy"))
    for name in ("references.jsonl","measurements.csv"):
        assert (tmp_path/"direct-full"/name).read_bytes()==(expected/name).read_bytes()
    assert result["artifacts"]["reference_records"]==result["output_samples"]
    assert (tmp_path/"direct-full"/"curve.png").is_file()


def test_cuda_ordered_scan_matches_cpu_with_nonfinite_signed_pairs():
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():pytest.skip("No CUDA device")
    except Exception as error:
        pytest.skip(f"CUDA unavailable: {error}")
    rng=np.random.default_rng(83)
    signals=np.ascontiguousarray(rng.uniform(.02,3,(217,2)))
    signals[[71,81,92,104,117],0]=np.nan
    signals[[77,87,97,107],1]=np.inf
    p=parameters()
    plan=direct.scan_layout(len(signals),p)
    signals[plan["center"]]=[np.nan,-np.inf]
    cpu,cpu_count=direct.compute_cpu(signals,p)
    gpu,gpu_count=direct.compute_gpu(cp.asarray(signals),p,cp=cp)
    np.testing.assert_allclose(cp.asnumpy(gpu),cpu,atol=2e-13,rtol=2e-13)
    np.testing.assert_array_equal(cp.asnumpy(gpu_count),cpu_count)
