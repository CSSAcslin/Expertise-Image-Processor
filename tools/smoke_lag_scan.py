"""Isolated CPU/CUDA Q run, trial, inspection and browser task-detail checks."""
from pathlib import Path
import argparse
import io
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect,sync_playwright

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from eip.module_registry import new_node


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend",choices=("cpu","gpu"),required=True)
    parser.add_argument("--port",type=int,default=8914)
    args=parser.parse_args()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1",args.port))
    folder=ROOT/".test-output"/f"q94-web-{args.backend}-{time.time_ns()}"
    folder.mkdir(parents=True)
    t=np.arange(3073)[:,None,None]
    grid=np.arange(144).reshape(12,12)
    values=np.concatenate((20+(grid+3*t)%3000,50+(grid+2*t+t*t)%3500),axis=2).astype(np.uint16)
    tifffile.imwrite(folder/"source.tif",values,photometric="minisblack",metadata={"axes":"TYX"})
    url=f"http://127.0.0.1:{args.port}"
    with (folder/"server.log").open("w",encoding="utf-8") as log:
        server=subprocess.Popen([sys.executable,str(ROOT/"run_web.py"),"--port",str(args.port),"--no-browser","--workspace",str(folder)],
                                cwd=ROOT,stdout=log,stderr=log,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        try:
            for _ in range(150):
                if server.poll() is not None:
                    raise RuntimeError(f"Server failed: {folder/'server.log'}")
                try:
                    if httpx.get(url+"/api/state",timeout=.5).status_code==200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError("Isolated service did not start")
            with httpx.Client(base_url=url,timeout=60) as api,sync_playwright() as pw:
                state=api.get("/api/state").json()
                api.headers["X-EIP-Token"]=state["session_token"]
                settings=api.get("/api/resources").json()["settings"]
                settings["backend"]=args.backend
                api.post("/api/resources",json=settings).raise_for_status()
                api.post("/api/sources/open",json=dict(path=str(folder/"source.tif"),interval=.01)).raise_for_status()
                source_id=api.get("/api/state").json()["sources"][0]["id"]
                nodes=[new_node(a) for a in ("alignment","point_extract","correlation_curve")]
                nodes[0]["parameters"].update(left=dict(x=0,y=0,width=12,height=12),right=dict(x=12,y=0,width=12,height=12),
                    midline=[12.,12.],dx=0,dy=0,mirror_x=False,mirror_y=False,interpolation="nearest")
                nodes[1]["parameters"].update(x=2,y=3,output="main")
                nodes[2]["parameters"].update(analysis="curve",window=31,min_samples=25,lag_min=-1500,lag_max=1500,lag_step=1,center=None)
                browser=pw.chromium.launch(channel="msedge" if os.name=="nt" else "chromium",headless=True)
                page=browser.new_page(viewport=dict(width=1600,height=1000),accept_downloads=True)
                errors,hidden_jobs=[],[]
                page.on("pageerror",lambda e:errors.append(str(e)))
                page.on("request",lambda r:hidden_jobs.append(r.url) if r.method=="POST" and r.url.endswith("/api/jobs") else None)
                page.goto(url)
                expect(page.locator("#frame-info")).to_contain_text("3073")
                def run(mode):
                    response=api.post("/api/jobs",json=dict(contract_version=3,mode=mode,source_ids=[source_id],steps=nodes,name=f"Q scan {mode}"))
                    response.raise_for_status()
                    identifier=response.json()["id"]
                    for _ in range(600):
                        job=api.get(f"/api/jobs/{identifier}").json()
                        if job["status"] in {"succeeded","failed","cancelled","completed_with_errors"}:
                            break
                        time.sleep(.05)
                    assert job["status"]=="succeeded",job
                    item=job["run"]["results"][0]
                    row=item["modules"]["execution"]["lag_output_blocks"]["3"]
                    assert row["output_samples"]==3001 and row["output_blocks"]==12
                    assert row["pair_updates"]==3001*31
                    assert row["state"]=="complete"
                    assert item["progress_axis"]=="Q" and job["progress_unit"]=="延迟点"
                    assert all(d["actual"]==args.backend for d in item["modules"]["devices"].values())
                    assert item["modules"]["reference_execution"]["lag_time_summaries"]==3001
                    assert item["modules"]["reference_execution"].get("tree_rebuilds",0)==0
                    if args.backend=="gpu":
                        timing=item["performance"]["gpu_timing"]["kernels"]["eip_g2_lag_block_float64"]
                        assert timing["calls"]==12 and timing["timed_calls"]>0
                    response=api.get(f"/api/jobs/{identifier}/download?file=array")
                    response.raise_for_status()
                    return job,np.load(io.BytesIO(response.content),allow_pickle=False)
                formal,actual=run("run")
                mapping=formal["run"]["results"][0]["mappings"][nodes[-1]["id"]]
                center=mapping["centers"]["start"]
                first=center-15
                left=values[first:first+31,3,2].astype(float)
                expected=[]
                for lag in range(-1500,1501):
                    right=values[first+lag:first+lag+31,3,14].astype(float)
                    expected.append(np.mean(left*right)/(left.mean()*right.mean()))
                np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
                trial,trial_values=run("trial")
                np.testing.assert_array_equal(actual,trial_values)
                response=api.post("/api/modules/inspect",json=dict(source_id=source_id,steps=nodes,sample_index=0))
                response.raise_for_status()
                np.testing.assert_allclose(response.json()["values"],[[expected[0]]],rtol=2e-13,atol=2e-13)
                page.locator("#toggle-tasks").click()
                task=page.locator(".job-row").filter(has=page.locator(f"[data-detail='{formal['id']}']"))
                expect(task.locator(".job-progress")).to_contain_text("延迟点")
                task.locator("[data-detail]").click()
                dialog=page.locator("#job-dialog")
                execution=dialog.locator("details").filter(has=page.locator("summary",has_text="执行优化与数据范围"))
                execution.locator("summary").click()
                expect(execution).to_contain_text("τ批量输出")
                expect(execution).to_contain_text("连续信号块")
                expect(execution).to_contain_text("τ来源时间")
                expect(execution).to_contain_text("12 批")
                expect(execution).to_contain_text("12 次GPU发射" if args.backend=="gpu" else "CPU保序累计")
                page.screenshot(path=str(folder/"q-details.png"),animations="disabled")
                page.set_viewport_size(dict(width=1024,height=768))
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                assert dialog.evaluate("e=>e.scrollWidth<=e.clientWidth+1")
                page.screenshot(path=str(folder/"q-small.png"),animations="disabled")
                assert not errors and not hidden_jobs,(errors,hidden_jobs)
                assert len(api.get("/api/jobs").json())==2
                checks=dict(status="passed",backend=args.backend,version=state["version"],formal_id=formal["id"],trial_id=trial["id"],
                            output_samples=3001,window=31,max_absolute_error=float(np.max(np.abs(actual-expected))),
                            javascript_errors=errors,hidden_browser_jobs=hidden_jobs,
                            execution=formal["run"]["results"][0]["modules"]["execution"],
                            references=formal["run"]["results"][0]["modules"]["reference_execution"])
                (folder/"checks.json").write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding="utf-8")
                browser.close()
            print(json.dumps(dict(status="passed",backend=args.backend,output=str(folder))))
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(5)


if __name__=="__main__":
    main()
