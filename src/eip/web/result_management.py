"""Saved-run lifecycle; file operations never target raw sources."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
import zipfile
from eip.persistence import write_json


TERMINAL = {"succeeded","completed_with_errors","failed","cancelled"}


class ResultManager:
    def __init__(self,workbench):
        self.workbench = workbench
        self.lock = threading.RLock()
        self.downloads = {}
        self.active_files = {}
        self.confirmations = {}
        self.index_path = workbench.workspace/".eip-results.json"

    def remember(self):
        """Only local references; numerical data and run manifests stay authoritative."""
        with self.workbench.lock:
            rows = [{k:j[k] for k in ("id","mode","engine","contract_version","task","name","source_name","source_ids","source_count","status","completed","total","created","directory","summary","error") if k in j}
                    for j in self.workbench.jobs.values() if j["status"] in TERMINAL and j.get("directory")]
            write_json(self.index_path,dict(schema_version=1,results=rows))

    def restore_index(self):
        if not self.index_path.is_file():
            return []
        errors = []
        try:
            value = json.loads(self.index_path.read_text(encoding="utf-8"))
            if value.get("schema_version")!=1 or not isinstance(value.get("results"),list) or len(value["results"])>64:
                raise ValueError("结果索引格式无效")
            for row in value["results"]:
                try:
                    if row["id"] in self.workbench.jobs:
                        continue
                    root = self._directory(row)
                    run = json.loads((root/"run.json").read_text(encoding="utf-8"))
                    job = dict(row,run=run,payload=run.get("spec",{}),logs=[],result_available=run.get("status") in {"succeeded","completed_with_errors"})
                    if run.get("status") not in TERMINAL:
                        job.update(status="failed",error="上次运行未完成，保留临时输出供检查")
                    self.workbench.enqueue(row["id"],job,queue_job=False)
                except (KeyError,ValueError,OSError,TypeError) as error:
                    errors.append(str(error))
        except (ValueError,OSError) as error:
            errors.append(str(error))
        return errors

    def _read_run(self,path):
        path = Path(path).expanduser().resolve()
        if path.is_dir():
            path = path/"run.json"
        if path.name!="run.json" or not path.is_file():
            raise ValueError("请选择导出目录或其中的 run.json")
        if path.stat().st_size>8*1024**2:
            raise ValueError("运行记录超过 8 MiB，请选择更具体的导出目录")
        run = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(run,dict) or not isinstance(run.get("run_id"),str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",run["run_id"]):
            raise ValueError("此文件不是有效的 EIP 运行记录")
        return path,run

    def _register_run(self,path,run):
        """Recover metadata only; opening a directory never loads or copies arrays."""
        from datetime import datetime
        identifier = run["run_id"]
        sequence = run.get("contract_version")==3
        split = not sequence and run.get("mode")=="split_correlation"
        if split:
            if run.get("schema_version")!=1 or not isinstance(run.get("config"),dict) or not isinstance(run.get("source"),dict):
                raise ValueError("分幅运行记录不完整")
            payload = dict(source=run["source"],provenance=run.get("provenance",{}))
            specs = [run["source"]]
        else:
            if (not sequence and run.get("schema_version")!=1) or not isinstance(run.get("spec"),dict) or not isinstance(run.get("results"),list):
                raise ValueError("不支持此历史运行格式")
            payload = run["spec"]
            specs = payload.get("sources",[])
        good = [item for item in run.get("results",[]) if isinstance(item,dict) and item.get("status")=="succeeded"]
        status = run.get("status")
        incomplete = status not in TERMINAL or path.parent.name.endswith(".partial")
        if status not in TERMINAL or incomplete and status in {"succeeded","completed_with_errors"}:
            status = "failed"
        if not split and not good and status in {"succeeded","completed_with_errors"}:
            status = "failed"
        source_name = Path(specs[0].get("path","未知来源")).name if specs else "未知来源"
        provenance = payload.get("provenance",[])
        provenance = [provenance] if isinstance(provenance,dict) else provenance
        created = path.stat().st_mtime
        try:
            created = datetime.fromisoformat(run.get("created_at") or run.get("started_at") or "").timestamp()
        except (ValueError,TypeError):
            pass
        job = dict(id=identifier,mode="split" if split else payload.get("mode","run"),engine="sequence" if sequence else "legacy",name=run.get("display_name") or source_name,source_name=source_name,source_ids=[item.get("source_id") for item in provenance if isinstance(item,dict)],source_count=len(specs),status=status,completed=sum(item.get("completed_frames",0) for item in good),total=None,created=created,directory=str(path.parent),summary="历史分幅结果" if split else "历史模块结果" if sequence else "历史图像序列",run=run,payload=payload,logs=[],result_available=status in {"succeeded","completed_with_errors"})
        if sequence:
            job["contract_version"] = 3
        if split:
            job["task"] = run.get("task","run")
        if incomplete:
            job["error"] = "上次运行未完成或输出未发布，仅保留记录供检查与清理"
        elif run.get("error"):
            job["error"] = str(run["error"])
        with self.workbench.lock:
            existing = self.workbench.jobs.get(identifier)
            if existing:
                if Path(existing.get("directory") or "").resolve()!=path.parent:
                    raise ValueError("此运行标识已对应另一目录，请先移除原记录再打开此目录")
                return dict(id=identifier,existing=True)
            if len(self.workbench.jobs)>=64:
                raise ValueError("结果列表已达 64 项，请先清理或移除部分记录，再次打开此目录")
            self.workbench.enqueue(identifier,job,queue_job=False)
        return dict(id=identifier,existing=False)

    def restore(self,value):
        path,run = self._read_run(value)
        if not path.parent.name.endswith(".partial"):
            if run.get("contract_version")==3 and run.get("status") in {"succeeded","completed_with_errors"}:
                return self.workbench.restore_sequence(str(path))
            if run.get("mode")=="split_correlation" and run.get("status")=="succeeded":
                return self.workbench.split.restore(str(path))
        with self.lock:
            result = self._register_run(path,run)
            self.remember()
            return result

    def restore_directory(self,value):
        root = Path(value).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("请选择已有的导出文件夹")
        restored,failed,existing = [],[],0
        pending,visited,entries = [root],0,0
        reason = None
        deadline = time.monotonic()+10
        with self.lock:
            while pending:
                if visited>=2500 or entries>=20000 or time.monotonic()>deadline:
                    reason = "目录较大，本次扫描达到上限；可选择更具体的导出子目录继续找回"
                    break
                folder = pending.pop();visited += 1
                manifest = folder/"run.json"
                try:
                    if manifest.is_file():
                        if manifest.is_symlink():
                            raise ValueError("运行记录为链接，请直接选择其真实导出目录")
                        path,run = self._read_run(manifest)
                        with self.workbench.lock:
                            full = run["run_id"] not in self.workbench.jobs and len(self.workbench.jobs)>=64
                        if full:
                            reason = "结果列表已达 64 项，请先清理或移除部分记录，再次打开此目录"
                            break
                        result = self._register_run(path,run)
                        if result["existing"]:
                            existing += 1
                        else:
                            restored.append(result["id"])
                        # A run's children contain outputs, not additional run roots.
                        continue
                    with os.scandir(folder) as children:
                        for child in children:
                            entries += 1
                            if entries>=20000 or time.monotonic()>deadline:
                                reason = "目录较大，本次扫描达到上限；可选择更具体的导出子目录继续找回"
                                break
                            if not child.name.startswith(".") and child.is_dir(follow_symlinks=False) and not Path(child.path).is_junction():
                                pending.append(Path(child.path))
                    if reason:
                        break
                except (ValueError,OSError,KeyError,TypeError) as error:
                    failed.append(dict(path=str(folder),error=str(error)))
                    if len(failed)>=64:
                        reason = "未识别记录较多，请选择更具体的导出子目录"
                        break
            if restored:
                try:
                    self.remember()
                except OSError as error:
                    failed.append(dict(path=str(self.index_path),error=f"结果已找回，但索引未能保存：{error}"))
        return dict(path=str(root),restored=restored,existing=existing,failed=failed,truncated=bool(reason),limit_reason=reason)

    def open_source(self,identifier,value):
        """Read an older TIFF export on demand, preserving its recorded coordinates."""
        job = self.workbench.job(identifier)
        if job.get("engine")=="sequence" or job["mode"]=="split" or job["status"] not in {"succeeded","completed_with_errors"}:
            raise ValueError("此记录没有可打开的旧版图像序列")
        rows = job["run"]["results"]
        position = value.get("source")
        if position is None:
            position = next((i for i,item in enumerate(rows) if item.get("status")=="succeeded"),-1)
        if not isinstance(position,int) or not 0<=position<len(rows) or rows[position].get("status")!="succeeded":
            raise ValueError("请选择完整的来源输出")
        item = rows[position];name = item["output_directory"]
        if not isinstance(name,str) or name in {".",".."} or Path(name).name!=name:
            raise ValueError("历史输出目录无效")
        root = Path(job["directory"]).resolve();folder = (root/name).resolve()
        if not folder.is_relative_to(root) or not folder.is_dir():
            raise ValueError("历史输出目录缺失或已移动")
        with self.workbench.lock:
            entry = next((s for s in self.workbench.sources.values() if s["role"]=="result" and s.get("run_id")==identifier and Path(s["source"]["path"]).resolve()==folder),None)
            if entry:
                return dict(source=entry)
        return self.workbench.open(dict(path=str(folder),kind="sequence"),role="result",run_id=identifier,source_position=position,display_name=job["name"],input_id=job["source_ids"][position] if position<len(job["source_ids"]) else None,origin=item.get("origin",[0,0]),base_shape=item.get("base_shape") or item.get("data_spec",{}).get("shape",[])[:2],result_kind=item.get("result_kind","image"))

    def _directory(self,job):
        if not job.get("directory"):
            return None
        path = Path(job["directory"])
        root = path.resolve()
        identifier = job["id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}",identifier):
            raise ValueError("运行标识无效")
        names = {identifier,f"run_{identifier}",f"split_{identifier}"}
        names |= {name+".partial" for name in list(names)}
        if path.name not in names or path.is_symlink() or path.is_junction() or root==root.parent:
            raise ValueError("结果目录与运行标识不一致，禁止清理或打包")
        if not root.is_dir() or not (root/"run.json").is_file():
            raise ValueError("结果目录或运行记录缺失，请移除工作区记录")
        record = json.loads((root/"run.json").read_text(encoding="utf-8"))
        if record.get("run_id")!=identifier:
            raise ValueError("磁盘运行记录与结果不匹配")
        # A malicious/restored record cannot make an input directory a deletion target.
        specs = [record.get("source",{}),*record.get("spec",{}).get("sources",[])]
        if any(s.get("path") and Path(s["path"]).resolve().is_relative_to(root) for s in specs):
            raise ValueError("运行目录包含原始输入，禁止清理")
        return root

    def _files(self,root):
        for path in root.rglob("*"):
            if path.is_symlink() or path.is_junction() or not path.resolve().is_relative_to(root):
                raise ValueError("结果目录含链接或越界文件，禁止清理或打包")
            if path.is_file():
                yield path

    def _dependencies(self,identifier,root):
        result = []
        self._expire()
        for ticket,entry in {**self.downloads,**self.active_files}.items():
            if not entry.get("temporary",True) and entry["run_id"]==identifier:
                result.append(dict(kind="download",id=ticket,name="进行中或待开始的文件下载"))
        for source in self.workbench.sources.values():
            if source["role"]=="result" and source.get("run_id")==identifier:
                continue
            direct = root is not None and Path(source["source"]["path"]).resolve().is_relative_to(root)
            lineage = any(part.get("run_id")==identifier for part in source.get("lineage",[]))
            if direct or lineage:
                result.append(dict(kind="source",id=source["id"],name=source.get("display_name",source["meta"]["name"])))
        for job in self.workbench.jobs.values():
            if job["id"]==identifier:
                continue
            run = job.get("run",{})
            payload = job.get("payload") or run.get("spec") or dict(source=run.get("source",{}),provenance=run.get("provenance",[]))
            specs = [payload.get("source",{}),*payload.get("sources",[])]
            direct = root is not None and any(s.get("path") and Path(s["path"]).resolve().is_relative_to(root) for s in specs)
            provenance = payload.get("provenance",[])
            provenance = [provenance] if isinstance(provenance,dict) else provenance
            lineage = any(part.get("run_id")==identifier for item in provenance for part in item.get("lineage",[]))
            if direct or lineage:
                result.append(dict(kind="job",id=job["id"],name=job["name"]))
        return result

    def deletion_preview(self,identifier):
        with self.lock,self.workbench.lock:
            job = self.workbench.job(identifier)
            if job["status"] not in TERMINAL or self.workbench.active and self.workbench.active["id"]==identifier:
                raise ValueError("请等待任务进程结束后管理结果")
            error = None
            try:
                root = self._directory(job)
                count,size = 0,0
                if root:
                    for path in self._files(root):
                        count += 1
                        size += path.stat().st_size
            except (ValueError,OSError) as exception:
                root,count,size,error = None,0,0,str(exception)
            dependencies = self._dependencies(identifier,root)
            token = secrets.token_urlsafe(32)
            fingerprint = hashlib.sha256((root/"run.json").read_bytes()).hexdigest() if root else None
            self.confirmations = {k:v for k,v in self.confirmations.items() if v["expires"]>time.monotonic()}
            if len(self.confirmations)>=64:
                self.confirmations.pop(next(iter(self.confirmations)))
            self.confirmations[token] = dict(id=identifier,path=str(root) if root else None,hash=fingerprint,expires=time.monotonic()+300)
            return dict(id=identifier,name=job["name"],path=str(root) if root else job.get("directory"),
                        file_count=count,size_bytes=size,dependencies=dependencies,can_delete_files=bool(root) and not dependencies,
                        error=error,confirmation=token)

    def delete(self,identifier,value):
        mode = value.get("mode")
        if mode not in {"reference","files"}:
            raise ValueError("删除方式需为移除记录或清理运行文件")
        with self.lock,self.workbench.lock:
            confirmation = self.confirmations.pop(value.get("confirmation"),None)
            if not confirmation or confirmation["id"]!=identifier or confirmation["expires"]<time.monotonic():
                raise ValueError("删除预览已过期，请重新查看删除范围")
            job = self.workbench.job(identifier)
            if job["status"] not in TERMINAL or self.workbench.active and self.workbench.active["id"]==identifier:
                raise ValueError("任务仍在执行")
            if mode=="files":
                root = self._directory(job)
                if root is None or str(root)!=confirmation["path"] or hashlib.sha256((root/"run.json").read_bytes()).hexdigest()!=confirmation["hash"]:
                    raise ValueError("结果目录或记录已变化，请重新确认")
                blockers = self._dependencies(identifier,root)
                if blockers:
                    raise ValueError("文件仍被来源或后续任务引用："+"、".join(b["name"] for b in blockers))
                for _ in self._files(root):
                    pass
                self.workbench.preview.reset()
                self.workbench.inspection.release_idle()
                self.workbench.frames.clear()
                # Absolute resolved target and its run identity were checked above.
                shutil.rmtree(root)
            removed = [s["id"] for s in self.workbench.sources.values() if s["role"]=="result" and s.get("run_id")==identifier]
            for key in removed:
                del self.workbench.sources[key]
            del self.workbench.jobs[identifier]
            self.remember()
            return dict(ok=True,removed_sources=removed,files_deleted=mode=="files")

    def _expire(self):
        for token,entry in list(self.downloads.items()):
            if entry["expires"]<time.monotonic():
                if entry.get("temporary",True):
                    entry["path"].unlink(missing_ok=True)
                del self.downloads[token]

    def file_link(self,identifier,path,filename,*,temporary=False):
        with self.lock:
            self._expire()
            if len(self.downloads)>=16:
                raise ValueError("待下载文件过多，请完成下载后重试")
            token = secrets.token_urlsafe(32)
            self.downloads[token] = dict(path=Path(path),name=filename,temporary=temporary,run_id=identifier,expires=time.monotonic()+300)
            return dict(url=f"/api/result-downloads/{token}",filename=filename)

    def package(self,identifier):
        """ZIP64 copies in bounded chunks to disk, then a single-use download."""
        return self.package_many([identifier])

    def package_many(self,identifiers):
        if not isinstance(identifiers,list) or not 1<=len(identifiers)<=64 or any(not isinstance(i,str) for i in identifiers):
            raise ValueError("请选择 1–64 个处理结果")
        with self.lock:
            self._expire()
            if sum(e.get("temporary",True) for e in self.downloads.values())>=2:
                raise ValueError("已有两个待下载运行包，请下载后重试")
            roots = []
            for identifier in dict.fromkeys(identifiers):
                job = self.workbench.job(identifier)
                if job["status"] not in TERMINAL:
                    raise ValueError("请等待任务结束后导出")
                root = self._directory(job)
                if root is None:
                    raise ValueError("此临时试算没有磁盘输出，请正式运行后导出")
                roots.append(root)
            folder = self.workbench.workspace/".eip-cache"/"downloads"
            folder.mkdir(parents=True,exist_ok=True)
            size = sum(p.stat().st_size for root in roots for p in self._files(root))
            if shutil.disk_usage(folder).free<size+32*1024**2:
                raise ValueError("打包临时磁盘空间不足，可从结果工具分别导出数值文件")
            token = secrets.token_urlsafe(32)
            target = folder/f"{token}.zip"
            try:
                with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                    for root in roots:
                        for path in self._files(root):
                            archive.write(path,str(Path(root.name)/path.relative_to(root)))
                name = f"{roots[0].name}.zip" if len(roots)==1 else f"results_{len(roots)}.zip"
                self.downloads[token] = dict(path=target,expires=time.monotonic()+300,name=name)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return dict(url=f"/api/result-downloads/{token}",filename=name,size_bytes=target.stat().st_size)

    def take_download(self,token):
        with self.lock:
            self._expire()
            entry = self.downloads.pop(token,None)
            if not entry:
                raise ValueError("下载链接已过期或已使用，请重新导出")
            if not entry.get("temporary",True):
                self.active_files[token] = entry
            return entry

    def finish_download(self,token,entry):
        with self.lock:
            self.active_files.pop(token,None)
            if entry.get("temporary",True):
                entry["path"].unlink(missing_ok=True)

    def close(self):
        with self.lock:
            for entry in self.downloads.values():
                if entry.get("temporary",True):
                    entry["path"].unlink(missing_ok=True)
            self.downloads.clear()
