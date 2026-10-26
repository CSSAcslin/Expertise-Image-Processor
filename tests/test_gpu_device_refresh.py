"""Refresh stale device discovery without changing execution settings."""
from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient

from eip.modes.split_correlation import backend
from eip.web.app import create_app


def test_device_discovery_notices_new_optional_package(monkeypatch):
    available = False
    invalidations = []
    original = backend.importlib.util.find_spec
    monkeypatch.setattr(backend.importlib, "invalidate_caches", lambda: invalidations.append(True))
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name, *a: (SimpleNamespace() if available else None) if name == "cupy" else original(name, *a))
    runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDeviceProperties=lambda index: dict(name=b"test GPU"))
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(cuda=SimpleNamespace(runtime=runtime)))
    assert not backend.devices()["gpu"]
    available = True
    assert backend.devices() == dict(cpu=True, gpu=True, gpu_name="test GPU", reason=None)
    assert len(invalidations) == 2


def test_resource_refresh_updates_device_and_exposes_running_python(tmp_path, monkeypatch):
    detected = dict(cpu=True, gpu=False, gpu_name=None, reason="missing CuPy")
    monkeypatch.setattr("eip.web.split.devices", lambda: dict(detected))
    with TestClient(create_app(tmp_path)) as client:
        state = client.get("/api/state").json()
        client.headers["X-EIP-Token"] = state["session_token"]
        detected.update(gpu=True, gpu_name="test GPU", reason=None)
        # Ordinary status reads stay inexpensive; explicit discovery replaces
        # the startup snapshot used by both settings and legacy preflight.
        assert not client.get("/api/resources").json()["devices"]["gpu"]
        assert client.post("/api/resources/refresh", headers={"X-EIP-Token": "wrong"}, json={}).status_code == 403
        response = client.post("/api/resources/refresh", json={})
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["devices"] == detected
        assert data["runtime"]["python_executable"] == sys.executable
        assert data["runtime"]["python_version"] == sys.version.split()[0]
        assert Path(data["runtime"]["code_directory"]).name == "eip"
        assert data["settings"] == state["resources"]
        assert client.get("/api/state").json()["split_devices"] == detected
        detected.update(gpu=False, gpu_name=None, reason="CUDA device disconnected")
        assert client.post("/api/resources/refresh", json={}).json()["devices"] == detected
