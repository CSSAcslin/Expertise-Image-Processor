"""Read-only verification of a real Packed10 CINE and complete-source statistics.

The rectangles are test coordinates, not an accepted experimental calibration.
Independent unpackbits decoding checks signed-window results against direct g².
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys
import threading
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from eip.domain import SourceSpec
from eip.readers import open_source
from eip.modes.split_correlation.config import CorrelationConfig
from eip.modes.split_correlation.storage import read_slice
import eip.modes.split_correlation.runner as runner
from smoke_cine import reference


class Events:
    def __init__(self):
        self.final=None
        self.last=0
    def put(self,value):
        if value["type"]!="progress":
            self.final=value
            return
        now=time.monotonic()
        if now-self.last>=20 or value["completed"]==value["total"]:
            print(json.dumps({"phase":value["name"],"done":value["completed"],"total":value["total"]},ensure_ascii=False),flush=True)
            self.last=now


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    path=args.source.resolve()
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    header,shape,ticks,times,decode=reference(path)
    assert shape==(256,128),"This verification uses explicit 256×128 test rectangles"
    output=root/".test-output"/f"split-cine-{int(time.time())}"
    output.mkdir(parents=True)
    source=SourceSpec(str(path))
    reader=open_source(source)
    revision=reader.meta.revision
    reader.close()
    center=-header[6]
    cfg=CorrelationConfig.from_dict({
        "geometry":{"left":{"x":32,"y":32,"width":64,"height":64},"right":{"x":168,"y":32,"width":64,"height":64},"midline":[128,128]},
        "photometry":{"input_semantics":"sensor_codes"},"window":31,"stride":31,"lag_min":-4,"lag_max":4,
        "center":center,"start":center-100,"end":center+101,"roi":{"x":16,"y":16,"width":16,"height":16},"points":[[21,27]],"memory_mb":64,"backend":"cpu"})
    # An independent scalar sum sees the exact complete scan but shares its I/O.
    # Decoder independence is checked separately for all frames used below.
    sums=np.zeros((2,2),dtype=np.float64)
    observed=[0]
    original_open=runner.open_source
    class AuditReader:
        def __init__(self,reader):self.reader=reader
        def __getattr__(self,name):return getattr(self.reader,name)
        def read_frame(self,index):
            f=self.reader.read_frame(index)
            if index==observed[0] and observed[0]<self.reader.meta.frame_count:
                sums[:]+=np.asarray([[f.pixels[59,53],f.pixels[75,69]],[f.pixels[59,189],f.pixels[75,205]]])
                observed[0]+=1
            return f
    def audit_open(spec):
        r=original_open(spec)
        return AuditReader(r) if observed[0]==0 else r
    runner.open_source=audit_open
    def execute(task,c,identifier):
        event=Events()
        payload={"id":identifier,"source":asdict(source),"revision":revision,"config":c.to_dict(),"task":task,"output_dir":str(output),"cache_root":str(root/".eip-cache"/"split"),"lut_sha256":None,"provenance":{"source_id":"read-only-cine-verification","lineage":[]}}
        runner.split_job_entry(payload,event,threading.Event())
        assert event.final and event.final["type"]=="split_completed",event.final
        return event.final
    try:
        started=time.monotonic()
        maps=execute("run",cfg,"real_maps")
        run=maps["run"]
        assert run["statistics"]["frames"]==len(times)
        assert run["peak_frame_cache_bytes"]<=64*1024**2*.32
        means=np.load(Path(maps["directory"])/"mean.npy",allow_pickle=False)
        if observed[0]==len(times):
            np.testing.assert_allclose(means[:,[27,43],[21,37]],sums/len(times),rtol=0,atol=2e-9)
        else:
            assert observed[0]==0,"A cached run must not be reported as an audited complete scan"
        first=run["centers"]["first"]
        required=set()
        for ti in range(run["array_shape"][0]):
            c=first+ti*cfg.stride
            required.update(range(c-15-4,c+15+4+1))
        required.update(range(center-15-4,center+15+4+1))
        decoded={i:decode(i) for i in sorted(required)}
        r=open_source(source)
        for i,a in decoded.items():np.testing.assert_array_equal(r.read_frame(i).pixels,a)
        r.close()
        max_error=0
        for ti in range(run["array_shape"][0]):
            c=first+ti*cfg.stride
            left=np.stack([decoded[i][48:64,48:64] for i in range(c-15,c+16)]).astype(float)
            for qi,q in enumerate(run["lags"]):
                right=np.stack([decoded[i+q][48:64,184:200] for i in range(c-15,c+16)]).astype(float)
                expected=(left*right).mean(axis=0)/(left.mean(axis=0)*right.mean(axis=0))
                actual,count,ref,_=read_slice(maps["directory"],ti,qi)
                np.testing.assert_allclose(actual,expected,rtol=7e-8,atol=0)
                assert np.all(count==31)
                assert ref["center_time"]==times[c] and ref["source_number"]==header[6]+c
                assert abs(ref["lag_seconds"]-np.mean(times[c-15+q:c+16+q]-times[c-15:c+16]))<1e-16
                max_error=max(max_error,float(np.max(np.abs(actual-expected))))
        curves=execute("curve",cfg,"real_curves")
        curve=json.loads((Path(curves["directory"])/"curves.json").read_text(encoding="utf-8"))
        assert curves["run"]["statistics_cache"]==maps["run"]["statistics_cache"]
        left=np.stack([decoded[i][48:64,48:64] for i in range(center-15,center+16)]).astype(float)
        for qi,q in enumerate(run["lags"]):
            right=np.stack([decoded[i+q][48:64,184:200] for i in range(center-15,center+16)]).astype(float)
            expected=(left*right).mean(axis=0)/(left.mean(axis=0)*right.mean(axis=0))
            np.testing.assert_allclose(curve["curves"][0]["g2"][qi],expected[11,5],rtol=1e-11)
            np.testing.assert_allclose(curve["curves"][1]["g2"][qi],expected.mean(),rtol=1e-11)
        assert (path.stat().st_size,path.stat().st_mtime_ns)==before
        report={"status":"passed","source":str(path),"frames":len(times),"reference_decoded_frames":len(decoded),"complete_scan_audited_frames":observed[0],"window":31,"lags":run["lags"],"map_shape":run["array_shape"],"max_float32_error":max_error,"full_mean_fit_seconds":run["statistics"]["fit_seconds"],"total_seconds":time.monotonic()-started,"test_rectangles_only":True,"original_unchanged":True,"outputs":str(output)}
        (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(report,ensure_ascii=False),flush=True)
    finally:runner.open_source=original_open


if __name__=="__main__":main()
