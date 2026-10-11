"""Read-only real CINE verification for contract v3 with a bounded source crop."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from eip.domain import SourceSpec
from eip.readers import open_source
from eip.module_registry import new_node
from eip.sequence_processing import process_sequence_job


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",default="F:/Data-test/Test1/22741_1_4.cine")
    path=Path(parser.parse_args().source)
    before=path.stat()
    source=SourceSpec(str(path.resolve()))
    reader=open_source(source)
    try:
        meta=asdict(reader.meta)
        actual_times=reader.times[1000:1201].copy()
        raw=np.stack([reader.read_frame(i).pixels for i in range(1000,1201)]).astype(np.float64)
    finally:reader.close()
    out=ROOT/".test-output"/f"axis-cine-{int(time.time())}"
    out.mkdir(parents=True)
    nodes=[new_node(k,meta["shape"],meta["frame_count"]) for k in ("time_crop","calibration","alignment","spatial_roi","statistics","correlation")]
    nodes[0]["parameters"].update(first=1001,last=1201)
    nodes[1]["parameters"].update(curve="linear",input_max=1023.)
    nodes[3]["parameters"].update(x=16,y=16,width=64,height=64)
    nodes[5]["parameters"].update(window=51,stride=20,map_lag=-1,lag_list=[-2,-1,0,1,2],center=101)
    # Direct source crops and direct means are independent of the module engine.
    # Fixed linear scaling cancels from standard g², so use saved camera codes.
    g=nodes[2]["parameters"]
    left=raw[:,g["left"]["y"]+16:g["left"]["y"]+80,g["left"]["x"]+16:g["left"]["x"]+80]
    right=raw[:,g["right"]["y"]+16:g["right"]["y"]+80,g["right"]["x"]+16:g["right"]["x"]+80]
    def direct(a,b):
        return np.mean(a*b,axis=0)/(a.mean(axis=0)*b.mean(axis=0))
    report={}
    for task in ("map","curve"):
        nodes[3]["parameters"]["mode"]="mean" if task=="curve" else "pixels"
        nodes[5]["parameters"]["analysis"]=task
        payload=dict(id=task,mode="run",sources=[asdict(source)],revisions=[meta["revision"]],steps=nodes,output_dir=str(out),display_name=f"CINE axis {task}",seed=0,center=None,sample_index=0,provenance=[])
        events=[]
        process_sequence_job(payload,events.append,lambda:False)
        end=events[-1]
        assert end["run"]["status"]=="succeeded",end
        item=end["run"]["results"][0]
        part=Path(end["directory"])/item["output_directory"]
        values=np.load(part/"values.npy",allow_pickle=False)
        refs=[json.loads(s) for s in (part/"references.jsonl").read_text(encoding="utf-8").splitlines()]
        assert item["modules"]["stages"][4]["input"]["length"]==201
        if task=="curve":
            assert values.shape==(5,)
            assert [r["correlation"]["lag_samples"] for r in refs]==[-2,-1,0,1,2]
            pairs=np.arange(75,126)
            expected_times=[float(np.mean(actual_times[pairs+q]-actual_times[pairs])) for q in range(-2,3)]
            np.testing.assert_allclose([r["correlation"]["lag_seconds"] for r in refs],expected_times,rtol=0,atol=1e-15)
            regional_left,regional_right=left.mean(axis=(1,2)),right.mean(axis=(1,2))
            expected=np.asarray([direct(regional_left[pairs],regional_right[pairs+q]) for q in range(-2,3)])
        else:
            assert values.shape==(8,64,64)
            centers=list(range(26,176,20))
            assert [r["correlation"]["center_sample"] for r in refs]==centers
            expected=np.stack([direct(left[c-25:c+26],right[c-26:c+25]) for c in centers])
        np.testing.assert_allclose(values,expected,rtol=5e-12,atol=2e-12)
        report[task]=dict(shape=list(values.shape),finite=int(np.isfinite(values).sum()),dtype=str(values.dtype),max_absolute_error=float(np.max(np.abs(values-expected))),first_reference=refs[0])
    after=path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(dict(output=str(out),source_unchanged=True,**report),ensure_ascii=False))


if __name__=="__main__":main()
