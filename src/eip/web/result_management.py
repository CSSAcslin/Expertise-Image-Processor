"""Saved-run lifecycle; file operations never target raw sources."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
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
            rows = [{k:j[k] for k in ("id","mode","engine","contract_version","task","name","source_name","source_ids","source_count","status","completed","total","created","directory","summary") if k in j}
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
            self.workbench.frames.clear()
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
        with self.lock:
            self._expire()
            if sum(e.get("temporary",True) for e in self.downloads.values())>=2:
                raise ValueError("已有两个待下载运行包，请下载后重试")
            job = self.workbench.job(identifier)
            if job["status"] not in TERMINAL:
                raise ValueError("请等待任务结束后导出")
            root = self._directory(job)
            if root is None:
                raise ValueError("此临时试算没有磁盘输出，请正式运行后导出")
            folder = self.workbench.workspace/".eip-cache"/"downloads"
            folder.mkdir(parents=True,exist_ok=True)
            size = sum(p.stat().st_size for p in self._files(root))
            if shutil.disk_usage(folder).free<size+32*1024**2:
                raise ValueError("打包临时磁盘空间不足，可从结果工具分别导出数值文件")
            token = secrets.token_urlsafe(32)
            target = folder/f"{token}.zip"
            try:
                with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_STORED,allowZip64=True) as archive:
                    for path in self._files(root):
                        archive.write(path,str(Path(root.name)/path.relative_to(root)))
                self.downloads[token] = dict(path=target,expires=time.monotonic()+300,name=f"{root.name}.zip")
            except Exception:
                target.unlink(missing_ok=True)
                raise
            return dict(url=f"/api/result-downloads/{token}",filename=f"{root.name}.zip",size_bytes=target.stat().st_size)

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
