"""Bounded saved-signal comparison with complete coordinate validation."""
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import struct
import numpy as np

from eip.domain import SourceSpec
from eip.sequence_storage import DatasetReader


def coordinate(c,mapping,profile,axis,index,reference):
    if profile:
        if mapping.get("edges"):
            return (mapping["edges"][index]+mapping["edges"][index+1])/2
        return mapping["coordinates"][index] if "coordinates" in mapping else index+1
    if c["sequence_axis"] not in {"T","Q","W"}:
        raise ValueError("静态记录只能使用分量视图")
    correlation=reference.get("correlation",{})
    if c["sequence_axis"]=="W":
        if axis=="time":
            raise ValueError("窗长扫描的横轴为平均样本数 W，不能改为采集时间或 τ")
        return correlation.get("window")
    if axis=="time":
        return correlation.get("lag_seconds") if c["sequence_axis"]=="Q" else reference.get("time")
    return correlation.get("lag_samples") if c["sequence_axis"]=="Q" else index+1


def output_mapping(item,c,columns):
    maps=item.get("mappings") or {}
    mappings=[dict(v,node_id=k) for k,v in maps.items()] if isinstance(maps,dict) else maps
    matches=[m for m in mappings if m.get("axis") and m["axis"] in c["axes"]]
    if c["semantic"]=="distribution":
        matches=[m for m in matches if len(m.get("edges",[]))==columns+1]
        if not matches:
            raise ValueError("直方图缺少强度箱边界，不能比较")
    node_id=item.get("node_id")
    if not node_id and c["semantic"]=="distribution":
        nodes=item.get("modules",{}).get("nodes",[])
        node_id=next((n["id"] for n in reversed(nodes) if n["enabled"] and n["algorithm"]=="distribution" and n["parameters"]["output"]=="main"),None)
    return next((m for m in matches if m.get("node_id")==node_id),matches[-1] if matches else {})


def candidates(workbench):
    rows=[]
    with workbench.lock:
        jobs=list(workbench.jobs.values())
    for job in jobs:
        if job.get("engine")!="sequence" or job["status"] not in {"succeeded","completed_with_errors"}:
            continue
        for source,parent in enumerate(job.get("run",{}).get("results",[])):
            for side,item in [(None,parent),*((s["id"],s) for s in parent.get("side_outputs",[]))]:
                c=item.get("contract",{})
                if item.get("status")!="succeeded" or not c or "Y" in c["axes"] and "X" in c["axes"]:
                    continue
                rows.append(dict(job_id=job["id"],source=source,side_output=side,contract=c,temporary=job["mode"]=="trial",
                    name=f"{job['name']} · {parent.get('data_spec',{}).get('name',source+1)} · {item.get('name') or ('主输出' if side is None else side)}"))
    return rows


@contextmanager
def opened(workbench,value):
    entries=value.get("items")
    if not isinstance(entries,list) or not 1<=len(entries)<=32:
        raise ValueError("请选择 1–32 条曲线")
    mode=value.get("mode","sequence"); axis=value.get("x","samples")
    if mode not in {"sequence","component"} or axis not in {"samples","time"}:
        raise ValueError("曲线视图或横轴无效")
    seen=set()
    with ExitStack() as stack:
        rows=[]
        for entry in entries:
            if not isinstance(entry,dict):
                raise ValueError("曲线来源格式无效")
            job=workbench.job(entry.get("job_id"))
            if job.get("engine")!="sequence" or job["status"] not in {"succeeded","completed_with_errors"}:
                raise ValueError("仅完整模块结果可加入比较")
            _,item,path=workbench._sequence_output(job,entry)
            c=item["contract"]
            if "Y" in c["axes"] and "X" in c["axes"]:
                raise ValueError("矩阵结果请先提取点、区域统计或截面")
            profile=mode=="component" or c["sequence_axis"]=="N"
            if profile and len(c["axes"])>1:
                raise ValueError("当前截面含多个轴；请先用按轴截面提取一维数据，或沿样本轴比较指定分量")
            column=entry.get("column",0); index=entry.get("index",0)
            columns=max(1,math.prod(c["shape"]))
            if type(column) is not int or not 0<=column<columns or type(index) is not int or not 0<=index<c["length"]:
                raise ValueError("曲线分量或样本超出范围")
            key=(job["id"],entry.get("source",0),entry.get("side_output"),None if profile else column,index if profile else None)
            if key in seen:
                raise ValueError("同一曲线不能重复加入")
            seen.add(key)
            reader=DatasetReader(SourceSpec(str(path/"dataset.json"),"dataset"));stack.callback(reader.close)
            mapping=output_mapping(item,c,columns)
            if profile and mapping.get("coordinates") is not None and len(mapping["coordinates"])!=columns:
                raise ValueError("截面坐标数量与数据尺寸不符")
            count=columns if profile else c["length"]
            indices=sorted(set([*range(0,count,max(1,math.ceil(count/2000))),count-1]))
            selected=set(indices); xs=[]; digest=hashlib.sha256(); time_digest=hashlib.sha256(); known_time=True
            for i in range(count):
                ref={} if profile else reader.sample_ref(i)
                time=ref.get("correlation",{}).get("lag_seconds") if c["sequence_axis"]=="Q" else ref.get("time")
                known_time &= time is not None and math.isfinite(time)
                time_digest.update(json.dumps(time,allow_nan=False).encode()+b"\n")
                coord=coordinate(c,mapping,profile,axis,i,ref)
                if coord is None or not math.isfinite(coord):
                    raise ValueError("横轴有未知值，请选择样本横轴或具有完整坐标的结果")
                digest.update(struct.pack("<d",float(coord)))
                if i in selected:xs.append(float(coord))
            axis_kind=("B" if c["semantic"]=="distribution" else mapping.get("axis") or c["axes"] or "component") if profile else c["sequence_axis"]
            # Equal centres alone cannot establish equal histogram bins.
            extra=mapping.get("edges") if axis_kind=="B" else None
            signature=(profile,axis_kind,count,digest.hexdigest(),json.dumps(extra))
            title=(item.get("name") or job["name"]).removeprefix("双幅时序比较 · ")
            if c["axes"]=="P" and not profile:
                title=("左路" if column==0 else "右路")+" · "+title
            elif not profile and columns>1:
                title+=f" · 分量 {column+1}"
            name=entry.get("name") or title
            if not isinstance(name,str) or not 1<=len(name)<=240:
                raise ValueError("曲线名称需为1至240个字符")
            row=dict(reader=reader,item=item,contract=c,entry=entry,name=name,profile=profile,column=column,index=index,
                count=count,indices=indices,xs=xs,mapping=mapping,signature=signature,time_signature=time_digest.hexdigest(),known_time=known_time,
                    temporary=job["mode"]=="trial",run_spec=job["run"].get("spec"),provenance=dict(job_id=job["id"],source=entry.get("source",0),side_output=entry.get("side_output"),column=column,index=index,run_version=job["run"].get("app_version")))
            if rows and signature!=rows[0]["signature"]:
                raise ValueError(f"「{name}」尺寸或横轴坐标不同，不能直接叠加；不会自动插值或截短")
            rows.append(row)
        yield rows


def compare(workbench,value):
    with opened(workbench,value) as rows:
        first=rows[0]; curves=[]
        for row in rows:
            reader=row["reader"]
            data=reader.read_sample(row["index"]).reshape(-1) if row["profile"] else None
            values=[data[i] if row["profile"] else reader.read_sample(i).reshape(-1)[row["column"]] for i in row["indices"]]
            curves.append(dict(name=row["name"],values=[float(v) if np.isfinite(v) else None for v in values],
                temporary=row["temporary"],contract=row["contract"],provenance=row["provenance"]))
        warnings=[]
        if not first["profile"] and first["contract"]["sequence_axis"]=="T" and value.get("x","samples")=="samples" and len({r["time_signature"] for r in rows})>1:
            warnings.append("按编辑后样本号对齐；各组真实时间不同，未作时间重采样")
        if not first["profile"] and first["contract"]["sequence_axis"]=="Q" and value.get("x","samples")=="samples" and len({r["time_signature"] for r in rows})>1:
            warnings.append("按有符号延迟样本对齐；各组实际配对时差不同")
        if not first["profile"] and first["contract"]["sequence_axis"] in {"T","Q"} and not all(r["known_time"] for r in rows) and value.get("x","samples")=="samples":
            warnings.append("部分真实时间未知；当前按样本 / τ 坐标比较")
        if len({r["contract"]["meaning"] for r in rows})>1:
            warnings.append("各曲线数值含义不同；共用纵轴，未作归一化")
        kind=first["signature"][1]
        label={"T":"真实时间 / s" if value.get("x")=="time" else "编辑后样本号","Q":"真实配对时差 / s" if value.get("x")=="time" else "有符号 τ / 上游样本","W":"平均窗长 W / 上游样本数","R":"剖面距离 / px","B":"强度","component":"分量索引"}.get(kind,kind)
        return dict(xs=first["xs"],indices=first["indices"],total=first["count"],downsampled=len(first["indices"])<first["count"],
                    x_label=label,curves=curves,warnings=warnings,coordinate_digest=first["signature"][3],
                    kind=kind,edges=first["mapping"].get("edges") if kind=="B" else None)


def export(workbench,value):
    """Full values plus reproducible comparison recipe; never export display sampling."""
    import csv
    import io
    from pathlib import Path
    import secrets
    import shutil
    import zipfile
    from eip import __version__
    manager=workbench.results
    with manager.lock,opened(workbench,value) as rows:
        manager._expire()
        if sum(e.get("temporary",True) for e in manager.downloads.values())>=2:
            raise ValueError("已有两个待下载运行包，请下载后重试")
        folder=workbench.workspace/".eip-cache"/"downloads";folder.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(folder).free<rows[0]["count"]*(len(rows)+1)*32+32*1024**2:
            raise ValueError("比较包临时磁盘空间不足")
        target=folder/(secrets.token_urlsafe(24)+".zip")
        try:
            with zipfile.ZipFile(target,"w",allowZip64=True) as archive:
                manifest=dict(schema_version=1,app_version=__version__,alignment=value.get("x","samples"),mode=value.get("mode","sequence"),
                    items=[dict(name=r["name"],contract=r["contract"],temporary=r["temporary"],selection=r["provenance"],
                        modules=r["item"].get("upstream_modules") or r["item"].get("modules"),module=r["item"].get("module"),
                        source=r["item"].get("source"),run_spec=r["run_spec"],mappings=r["item"].get("mappings"),
                        visible=r["entry"].get("visible",True),color=r["entry"].get("color")) for r in rows],
                    coordinate_digest=rows[0]["signature"][3],full_values=True,view=value.get("view",{}))
                archive.writestr("comparison.json",json.dumps(manifest,ensure_ascii=False,indent=2))
                with archive.open("values.csv","w",force_zip64=True) as raw,io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as text:
                    writer=csv.writer(text);writer.writerow(["coordinate",*[r["name"] for r in rows]])
                    first=rows[0];m=first["mapping"];c=first["contract"]
                    profiles=[r["reader"].read_sample(r["index"]).reshape(-1) if r["profile"] else None for r in rows]
                    for i in range(first["count"]):
                        ref={} if first["profile"] else first["reader"].sample_ref(i)
                        x=coordinate(c,m,first["profile"],value.get("x","samples"),i,ref)
                        values=[profiles[k][i] if r["profile"] else r["reader"].read_sample(i).reshape(-1)[r["column"]] for k,r in enumerate(rows)]
                        writer.writerow([x,*[v.item() if np.isfinite(v) else "" for v in values]])
            return manager.file_link(rows[0]["provenance"]["job_id"],target,"comparison.zip",temporary=True)
        except Exception:
            target.unlink(missing_ok=True)
            raise
