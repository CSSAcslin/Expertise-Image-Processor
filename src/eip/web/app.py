from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import json
import secrets

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from eip import __version__
from eip.web.runtime import Workbench


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
        if request.url.path.startswith("/api/") and request.url.path != "/api/state":
            if not secrets.compare_digest(request.headers.get("x-eip-token", ""), token):
                return JSONResponse({"detail": "工作台会话已过期，请刷新页面"}, status_code=403)
        if request.url.path == "/api/state" and request.headers.get("sec-fetch-site") == "cross-site":
            return JSONResponse({"detail": "请直接打开本机工作台"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; frame-ancestors 'none'"
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
        result.pop("payload", None)
        return result

    @app.post("/api/jobs/{identifier}/cancel")
    def cancel(identifier: str):
        workbench().cancel(identifier)
        return {"ok": True}

    @app.get("/api/jobs/{identifier}/download")
    def download(identifier: str, file: str = "manifest", source: int = 0):
        job = workbench().job(identifier)
        if not job.get("directory"):
            raise ValueError("任务尚未生成输出")
        directory = Path(job["directory"])
        if file == "manifest":
            path = directory / "run.json"
        elif file in {"csv", "image"}:
            run = job.get("run", {})
            results = run.get("results", [])
            if source < 0 or source >= len(results) or results[source]["status"] != "succeeded":
                raise ValueError("该来源没有完成的输出")
            result = results[source]
            path = directory / result["output_directory"] / ("measurements.csv" if file == "csv" else result["artifacts"][0])
        else:
            raise ValueError("未知导出类型")
        if not path.is_file():
            raise HTTPException(404, "输出文件已移走")
        return FileResponse(path, filename=f"{identifier}-{path.name}")

    @app.post("/api/project/save")
    def save_project(value: dict):
        return workbench().save_project(value["path"], value["view"])

    @app.post("/api/project/load")
    def load_project(value: dict):
        return workbench().load_project(value["path"])

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
