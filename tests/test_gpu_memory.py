"""Driver telemetry is separate from CUDA computation and allocation budgets."""
from types import SimpleNamespace
import os
import subprocess

import pytest
from fastapi.testclient import TestClient

from eip import gpu_memory
from eip.resources import MIB
from eip.web.app import create_app


def driver(monkeypatch, gpu, processes="", error=None):
    calls = []
    monkeypatch.setattr(gpu_memory.shutil, "which", lambda name: "nvidia-smi")
    def run(args, **options):
        calls.append((args, options))
        assert options["timeout"] == 2
        if error:
            raise error
        return SimpleNamespace(stdout=gpu if args[1].startswith("--query-gpu=") else processes)
    monkeypatch.setattr(gpu_memory.subprocess, "run", run)
    return calls


def test_tracks_only_current_workbench_pids_and_reuses_samples(monkeypatch):
    calls = driver(monkeypatch, '0, GPU-a, "RTX, test", 16303, 2048, 14000, TCC\n1, GPU-b, Other, 8000, 400, 7500, N/A\n',
                   "GPU-a, 101, 100\nGPU-a, 102, 25.5\nGPU-a, 101, 100\nGPU-a, 999, 1922.5\nGPU-b, 101, 10\n")
    monitor = gpu_memory.GpuMemoryMonitor()
    first = monitor.sample([101, 102])
    assert first["available"] and first["reason"] is None
    assert first["cards"][0]["name"] == "RTX, test"
    assert first["cards"][0]["application_bytes"] == int(125.5*MIB)
    assert first["cards"][0]["used_bytes"] == 2048*MIB
    assert first["cards"][0]["free_bytes"] == 14000*MIB  # reserved memory is not guessed away
    assert first["cards"][1]["application_bytes"] == 10*MIB
    second = monitor.sample([102])
    assert second["sampled_at"] == first["sampled_at"] and len(calls) == 2
    assert second["cards"][0]["application_bytes"] == int(25.5*MIB)
    assert second["cards"][1]["application_bytes"] == 0


@pytest.mark.parametrize("rows", ["", "GPU-a, 101, N/A\n", "GPU-a, 101, 100\nGPU-a, 102, [Not Supported]\n"])
def test_wddm_unknown_is_not_zero_or_card_total(monkeypatch, rows):
    driver(monkeypatch, "0, GPU-a, RTX 5070 Ti, 16303, 2048, 14000, WDDM\n", rows)
    card = gpu_memory.GpuMemoryMonitor().sample([101, 102])["cards"][0]
    assert card["application_bytes"] is None and "Windows" in card["application_reason"]
    assert card["used_bytes"] == 2048*MIB


def test_missing_process_query_preserves_card_memory(monkeypatch):
    monitor = gpu_memory.GpuMemoryMonitor()
    def query(executable, query):
        if query.startswith("--query-compute"):
            raise subprocess.TimeoutExpired("nvidia-smi", 2)
        return [["0", "GPU-a", "RTX", "16303", "2048", "14000", "TCC"]]
    monkeypatch.setattr(gpu_memory.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(monitor, "_query", query)
    data = monitor.sample([101])
    assert data["available"] and data["cards"][0]["application_bytes"] is None
    assert data["cards"][0]["free_bytes"] == 14000*MIB


def test_unavailable_and_timeout_are_bounded_then_can_recover(monkeypatch):
    clock = [100.]
    monkeypatch.setattr(gpu_memory.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(gpu_memory.shutil, "which", lambda name: None)
    monitor = gpu_memory.GpuMemoryMonitor()
    assert not monitor.sample([])["available"]
    calls = driver(monkeypatch, "0, GPU-a, RTX, 16303, 0, 16303, TCC\n", "\n")
    assert not monitor.sample([])["available"] and not calls
    clock[0] += 2
    assert monitor.sample([])["cards"][0]["application_bytes"] == 0
    calls = driver(monkeypatch, "", error=subprocess.TimeoutExpired("nvidia-smi", 2))
    clock[0] += 2
    data = monitor.sample([])
    assert not data["available"] and "超时" in data["reason"] and len(calls) == 1


@pytest.mark.parametrize("value", ["N/A", "[Not Supported]", "-1", "nan", "inf"])
def test_unknown_or_invalid_memory_is_not_reported_as_zero(value):
    assert gpu_memory.memory_bytes(value) is None


def test_api_uses_application_pids_and_current_saved_budget(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        state = client.get("/api/state").json()
        client.headers["X-EIP-Token"] = state["session_token"]
        workbench = client.app.state.workbench
        captured = []
        def sample(pids):
            captured.append(set(pids))
            return dict(available=True, cards=[], reason=None, sampled_at=100., interval_seconds=2)
        monkeypatch.setattr(workbench.gpu_memory, "sample", sample)
        assert client.get("/api/resources/gpu", headers={"X-EIP-Token": "wrong"}).status_code == 403
        data = client.get("/api/resources/gpu").json()
        assert data["budget_mb"] == state["resources"]["gpu_mb"]
        assert os.getpid() in captured[0]
        assert workbench.preview.process.pid in captured[0]
        assert client.get("/api/state").json()["resources"] == state["resources"]
