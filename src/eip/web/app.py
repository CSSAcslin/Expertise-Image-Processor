from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import json
import secrets

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
import anyio

from eip import __version__
from eip.web.runtime import Workbench


class DownloadResponse(FileResponse):
    """Release download leases and temporary files even when a client aborts."""
    def __init__(self,path,filename,finish):
        super().__init__(path,filename=filename)
        self.finish = finish

    async def __call__(self,scope,receive,send):
        try:
            await super().__call__(scope,receive,send)
        finally:
            with anyio.CancelScope(shield=True):
                await run_in_threadpool(self.finish)


def create_app(workspace: Path | None = None, restore_session: Path | None = None):
    restored = json.loads(restore_session.read_text(encoding="utf-8")) if restore_session else None
    token = restored["state"]["session_token"] if restored else secrets.token_urlsafe(32)
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("会话快照缺少有效令牌")
    root = (workspace or Path.cwd()).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.workbench = Workbench(root, restored)
        try:
            yield
        finally:
            app.state.workbench.close()

    app = FastAPI(title="Expertise Image Processor", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "请从本机工作台发起请求"}, status_code=403)
        if request.url.path.startswith("/api/") and request.url.path != "/api/state" and not request.url.path.startswith("/api/result-downloads/"):
            if not secrets.compare_digest(request.headers.get("x-eip-token", ""), token):
                return JSONResponse({"detail": "工作台会话已过期，请刷新页面"}, status_code=403)
        if request.url.path == "/api/state" and request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "请直接打开本机工作台"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    @app.exception_handler(KeyError)
    @app.exception_handler(TypeError)
    @app.exception_handler(OSError)
    def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    def workbench():
        return app.state.workbench

    @app.get("/api/state")
    def state():
        return {**workbench().state(), "session_token": token, "version": __version__}

    @app.get("/api/resources")
    def resources():
        return workbench().resource_status()

    @app.post("/api/resources")
    def set_resources(value: dict):
        return workbench().update_resources(value)

    @app.post("/api/resources/clear")
    def clear_resources():
        return workbench().clear_caches()

    @app.get("/api/results/{identifier}/delete-preview")
    def deletion_preview(identifier: str):
        return workbench().results.deletion_preview(identifier)

    @app.post("/api/results/{identifier}/delete")
    def delete_result(identifier: str,value: dict):
        return workbench().results.delete(identifier,value)

    @app.post("/api/results/{identifier}/package")
    def package_result(identifier: str):
        return workbench().results.package(identifier)

    @app.post("/api/results/restore")
    def restore_result(value: dict):
        return workbench().results.restore(value["path"])

    @app.post("/api/results/restore-directory")
    def restore_result_directory(value: dict):
        return workbench().results.restore_directory(value["path"])

    @app.post("/api/results/{identifier}/source")
    def open_historical_source(identifier: str,value: dict):
        return workbench().results.open_source(identifier,value)

    @app.get("/api/result-downloads/{ticket}")
    def package_download(ticket: str):
        entry = workbench().results.take_download(ticket)
        return DownloadResponse(entry["path"],entry["name"],lambda:workbench().results.finish_download(ticket,entry))

    @app.get("/api/browse")
    def browse(path: str | None = None, mode: str = "data"):
        return workbench().browse(path, mode)

    @app.get("/api/path")
    def inspect_path(path: str):
        return workbench().inspect_path(path)

    @app.post("/api/sources/open")
    def open_source(value: dict):
        return workbench().open(value)

    @app.post("/api/sources/{identifier}/promote")
    def promote(identifier: str):
        return workbench().promote(identifier)

    @app.post("/api/sources/{identifier}/remove")
    def remove(identifier: str):
        workbench().remove(identifier)
        return {"ok": True}

    @app.post("/api/sources/remove-selected")
    def remove_selected_sources(value: dict):
        return workbench().remove_sources(value.get("ids"))

    @app.post("/api/results/package-selected")
    def package_selected_results(value: dict):
        return workbench().results.package_many(value.get("ids"))

    @app.post("/api/sources/{identifier}/frame")
    def frame(identifier: str, value: dict):
        return workbench().frame(identifier, value)

    @app.post("/api/frames/{identifier}/render")
    def render(identifier: str, value: dict):
        return workbench().frames.render(identifier, value.get("display"))

    @app.get("/api/frames/{identifier}/pixel")
    def pixel(identifier: str, x: int, y: int):
        return workbench().frames.pixel(identifier, x, y)

    @app.get("/api/frames/{identifier}/region")
    def region(identifier: str, x: int, y: int, width: int, height: int):
        return workbench().frames.region(identifier, x, y, width, height)

    @app.post("/api/plugins/register")
    def register(value: dict):
        return workbench().register(value["path"])

    @app.get("/api/jobs")
    def jobs():
        return workbench().job_list()

    @app.post("/api/jobs")
    def submit(value: dict):
        return workbench().submit(value)

    @app.get("/api/jobs/{identifier}")
    def job(identifier: str):
        result = workbench().job(identifier)
        run = result.get("run",{})
        payload = result.pop("payload", {}) or run.get("spec",{})
        result["parameters"] = run.get("config") or payload.get("config") or {key: payload[key] for key in ("steps", "selection", "seed", "input_kinds") if key in payload}
        source = payload.get("source") or run.get("source")
        result["source_specs"] = [source] if source else payload.get("sources", [])
        return result

    @app.post("/api/jobs/{identifier}/rename")
    def rename_job(identifier: str, value: dict):
        return workbench().rename_job(identifier,value.get("name"))

    @app.get("/api/split/modules")
    def split_modules():
        from eip.module_registry import catalog
        return catalog(workbench().algorithms.values())

    @app.get("/api/modules")
    def modules():
        from eip.module_registry import catalog
        return catalog(workbench().algorithms.values())

    @app.post("/api/modules/plan")
    def module_plan(value: dict):
        return workbench().module_plan(value)

    @app.post("/api/modules/inspect")
    def module_inspect(value: dict):
        return workbench().inspect_module(value)

    @app.post("/api/modules/inspect/cancel")
    def cancel_inspection(value: dict):
        return dict(cancelled=workbench().inspection.cancel(value.get("inspection_id")))

    @app.post("/api/modules/migrate")
    def module_migrate(value: dict):
        from eip.module_registry import migrate_workbench_nodes, normalize_nodes, is_reference
        steps = value.get("steps",[])
        # Validate the draft structure without requiring an attached source or
        # executing a plugin. Incomplete numerical connections remain editable.
        normalized,_,_ = normalize_nodes(steps,check_types=False)
        nodes,report = migrate_workbench_nodes(normalized)
        normalized,_,_ = normalize_nodes(nodes,check_types=False)
        providers = {}
        for node in normalized:
            if node["enabled"]:
                for value in node["parameters"].values():
                    if not is_reference(value):
                        continue
                    reference = value["ref"]
                    provider = providers.get(reference["node_id"])
                    if provider is None or not provider["enabled"]:
                        raise ValueError("参数引用只能指向此前已启用的模块实例")
                    port = reference["port"]
                    if port.startswith("parameter:"):
                        parameter = provider["parameters"].get(port.partition(":")[2])
                        if not is_reference(parameter) and (isinstance(parameter,bool) or not isinstance(parameter,(int,float))):
                            raise ValueError("引用的参数端口不是数值参数")
                    elif not (provider["algorithm"]=="sequence_statistics" and port in {"mean","variance","count","scale"} or provider["algorithm"]=="global_statistics" and port in {"value","count"} or provider["algorithm"]=="parameter_binding" and port=="value"):
                        raise ValueError("参数引用的结果端口不存在或不支持数值绑定")
            providers[node["id"]] = node
        return dict(nodes=normalized,report=report)

    @app.get("/api/curves/candidates")
    def curve_candidates():
        from eip.web.curve_comparison import candidates
        return candidates(workbench())

    @app.post("/api/curves/compare")
    def curve_compare(value: dict):
        from eip.web.curve_comparison import compare
        with workbench().results.lock:
            return compare(workbench(),value)

    @app.post("/api/curves/export")
    def curve_export(value: dict):
        from eip.web.curve_comparison import export
        return export(workbench(),value)

    @app.get("/api/jobs/{identifier}/sequence")
    def sequence_result(identifier: str):
        return workbench().sequence_result(identifier)

    @app.post("/api/jobs/{identifier}/sequence")
    def sequence_slice(identifier: str,value: dict):
        return workbench().sequence_result(identifier,value)

    @app.post("/api/jobs/{identifier}/sequence/promote")
    def sequence_promote(identifier: str,value: dict):
        return workbench().sequence_promote(identifier,value)

    @app.post("/api/sequence/restore")
    def restore_sequence(value: dict):
        return workbench().restore_sequence(value["path"])

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel(identifier: str):
        workbench().cancel(identifier)
        return {"ok": True}

    @app.get("/api/jobs/{identifier}/download")
    def download(identifier: str, file: str = "manifest", source: int = 0, side_output: str | None = None):
        job = workbench().job(identifier)
        if not job.get("directory"):
            raise ValueError("任务尚未生成输出")
        directory = Path(job["directory"])
        if side_output:
            if job.get("engine")!="sequence" or job["status"] not in {"succeeded","completed_with_errors"}:
                raise ValueError("截流输出尚未完整保存")
            _,result,directory = workbench()._sequence_output(job,dict(source=source,side_output=side_output))
            names = {"manifest":"delivery.json","array":"values.npy","csv":"export.csv" if "export.csv" in result.get("artifacts",[]) else "measurements.csv","image":"first.tif","png":"curve.png","svg":"curve.svg","references":"references.jsonl","export":result.get("export_file")}
            if file not in names or not names[file]:
                raise ValueError("此截流未生成所选导出文件")
            if Path(names[file]).name!=names[file] or names[file] in {".",".."}:
                raise ValueError("截流导出文件记录无效")
            path = directory/names[file]
        elif file == "manifest":
            path = directory / "run.json"
        elif file in {"csv", "image", "array", "png", "svg", "references"}:
            run = job.get("run", {})
            results = run.get("results", [])
            if source < 0 or source >= len(results) or results[source]["status"] != "succeeded":
                raise ValueError("该来源没有完成的输出")
            result = results[source]
            names = {"csv":"measurements.csv","array":"values.npy","png":"curve.png","svg":"curve.svg","references":"references.jsonl"}
            if job.get("engine")=="sequence" and file=="image":
                path = directory/result["output_directory"]/"first.tif"
            else:
                path = directory / result["output_directory"] / (names[file] if file in names else result["artifacts"][0])
        else:
            raise ValueError("未知导出类型")
        if not path.resolve().is_relative_to(Path(job["directory"]).resolve()):
            raise ValueError("导出路径超出结果目录")
        if not path.is_file():
            raise HTTPException(404, "输出文件已移走")
        return FileResponse(path, filename=f"{identifier}-{path.name}")

    @app.post("/api/jobs/{identifier}/download-link")
    def download_link(identifier: str,value: dict):
        with workbench().results.lock,workbench().lock:
            response = download(identifier,value.get("file","manifest"),value.get("source",0),value.get("side_output"))
            return workbench().results.file_link(identifier,response.path,response.filename)

    @app.post("/api/project/save")
    def save_project(value: dict):
        return workbench().save_project(value["path"], value["view"])

    @app.post("/api/split/geometry")
    def split_geometry(value: dict):
        return workbench().split.geometry(value)

    @app.post("/api/split/preflight")
    def split_preflight(value: dict):
        return workbench().split.preflight(value)

    @app.post("/api/split/jobs")
    def split_submit(value: dict):
        return workbench().split.submit(value)

    @app.post("/api/split/restore")
    def split_restore(value: dict):
        return workbench().split.restore(value["path"])

    @app.get("/api/split/jobs/{identifier}")
    def split_result(identifier: str):
        return workbench().split.result(identifier)

    @app.post("/api/split/jobs/{identifier}/slice")
    def split_slice(identifier: str, value: dict):
        return workbench().split.slice(identifier,value)

    @app.post("/api/split/jobs/{identifier}/statistics")
    def split_statistics(identifier: str, value: dict):
        return workbench().split.statistics_view(identifier,value)

    @app.post("/api/split/jobs/{identifier}/promote")
    def split_promote(identifier: str, value: dict):
        return workbench().split.promote(identifier,value["lag_index"])

    @app.get("/api/split/jobs/{identifier}/download")
    def split_download(identifier: str, file: str="manifest",time_index: int=0,lag_index: int=0):
        _,directory,run = workbench().split.completed(identifier)
        names = {"manifest":"run.json","csv":"curves.csv","png":"curves.png","svg":"curves.svg","curve_json":"curves.json","windows":"windows.csv"}
        if file == "image":
            import io
            import tifffile
            from eip.modes.split_correlation.storage import read_slice
            if run["task"] not in {"trial","run"}:
                raise ValueError("曲线 / 统计任务没有系数图")
            pixels,_,ref,_ = read_slice(directory,time_index,lag_index)
            buffer = io.BytesIO()
            tifffile.imwrite(buffer,pixels,photometric="minisblack",metadata={"axes":"YX","window_ref":ref,"run_id":identifier,"display_name":run.get("display_name",identifier),"geometry":run["geometry"],"analysis_region":run["analysis_region"],"result_semantics":run.get("result_semantics","g2"),"workflow":run["config"].get("workflow",[])})
            return Response(buffer.getvalue(),media_type="image/tiff",headers={"Content-Disposition":f'attachment; filename="{identifier}-t{time_index}-q{lag_index}.tif"'})
        if file not in names or not (directory/names[file]).is_file():
            raise ValueError("该任务没有所选导出内容")
        return FileResponse(directory/names[file],filename=f"{identifier}-{names[file]}")

    @app.post("/api/split/jobs/{identifier}/download-link")
    def split_download_link(identifier: str,value: dict):
        response = split_download(identifier,value.get("file","manifest"),value.get("time_index",0),value.get("lag_index",0))
        if isinstance(response,FileResponse):
            return workbench().results.file_link(identifier,response.path,response.filename)
        folder = workbench().workspace/".eip-cache"/"downloads"
        folder.mkdir(parents=True,exist_ok=True)
        path = folder/(secrets.token_hex(24)+".tif")
        path.write_bytes(response.body)
        return workbench().results.file_link(identifier,path,f"{identifier}-t{value.get('time_index',0)}-q{value.get('lag_index',0)}.tif",temporary=True)

    @app.post("/api/project/load")
    def load_project(value: dict):
        return workbench().load_project(value["path"],migrate=bool(value.get("migrate",False)))

    @app.post("/api/demo")
    def demo():
        from eip.web.demo import create_demo
        paths = create_demo(root / "outputs" / "demo")
        return [workbench().open({"path": str(path)}) for path in paths]

    assets = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=assets), name="static")

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    return app
