"""Host readings use bounded counter deltas and share application PID scope."""
import os

from fastapi.testclient import TestClient
import pytest

from eip import device_telemetry as telemetry
from eip.web.app import create_app


def readings(monkeypatch):
    state = dict(now=100.0, system=(1000.0, 200.0), processes={101: (1, 10.0), 102: (2, 4.0)}, rss={101: 100, 102: 50})
    calls = []
    monkeypatch.setattr(telemetry.time, "monotonic", lambda: state["now"])
    monkeypatch.setattr(telemetry.time, "time", lambda: state["now"] + 1000)
    monkeypatch.setattr(telemetry, "system_cpu_times", lambda: calls.append("cpu") or state["system"])
    monkeypatch.setattr(telemetry, "process_cpu_times", lambda pid: state["processes"].get(pid))
    monkeypatch.setattr(telemetry, "process_memory", lambda pid: state["rss"].get(pid))
    monkeypatch.setattr(telemetry, "system_memory", lambda: dict(total_bytes=1000, available_bytes=400))
    return state, calls


def test_cpu_uses_all_processor_delta_and_readings_are_shared(monkeypatch):
    state, calls = readings(monkeypatch)
    monitor = telemetry.HostTelemetryMonitor()
    first = monitor.sample([101, 102, 101])
    assert first["cpu"]["system_percent"] is None
    assert first["cpu"]["application_percent"] is None
    assert first["memory"]["used_bytes"] == 600
    assert first["memory"]["application_bytes"] == 150  # duplicate PID counted once
    assert first["processes"] == [dict(pid=101,resident_bytes=100,cpu_percent=None),
                                  dict(pid=102,resident_bytes=50,cpu_percent=None)]
    state["now"] += 1
    assert monitor.sample([101])["sampled_at"] == first["sampled_at"]
    assert calls == ["cpu"]
    state["now"] += 1
    state["system"] = (1020.0, 205.0)
    state["processes"] = {101: (1, 12.0), 102: (2, 5.0)}
    second = monitor.sample([101, 102])
    assert second["sampled_at"] == first["sampled_at"] + 2
    assert second["cpu"]["system_percent"] == 75
    assert second["cpu"]["application_percent"] == 15
    assert second["processes"] == [dict(pid=101,resident_bytes=100,cpu_percent=10),
                                   dict(pid=102,resident_bytes=50,cpu_percent=5)]
    # Removing a completed worker filters the current set without another read.
    assert monitor.sample([101])["cpu"]["application_percent"] == 10
    assert len(calls) == 2


def test_worker_creation_pid_reuse_and_unavailable_rss_are_not_zero(monkeypatch):
    state, _ = readings(monkeypatch)
    monitor = telemetry.HostTelemetryMonitor()
    monitor.sample([101])
    changed = monitor.sample([101, 103])
    assert changed["cpu"]["application_percent"] is None
    assert changed["memory"]["application_bytes"] is None
    assert "切换" in changed["memory"]["application_reason"]
    assert changed["processes"][1] == dict(pid=103,resident_bytes=None,cpu_percent=None)
    state["now"] += 2
    state["system"] = (1020.0, 205.0)
    state["processes"][101] = (9, 100.0)  # same PID, different creation identity
    state["rss"][101] = None
    current = monitor.sample([101])
    assert current["cpu"]["system_percent"] == 75
    assert current["cpu"]["application_percent"] is None
    assert current["memory"]["application_bytes"] is None
    state["now"] += 2
    state["system"] = (1040.0, 210.0)
    state["processes"][101] = (9, 101.0)
    assert monitor.sample([101])["cpu"]["application_percent"] == 5
    assert len(monitor.previous["processes"]) == 1


@pytest.mark.parametrize("system", [None, (1000.0, 200.0), (990.0, 199.0)])
def test_missing_or_reset_cpu_counter_is_unknown(monkeypatch, system):
    state, _ = readings(monkeypatch)
    monitor = telemetry.HostTelemetryMonitor()
    monitor.sample([101])
    state["now"] += 2
    state["system"] = system
    result = monitor.sample([101])
    assert result["cpu"]["system_percent"] is None
    assert result["cpu"]["application_percent"] is None


def test_native_read_failure_preserves_other_available_readings(monkeypatch):
    readings(monkeypatch)
    def failed():
        raise OSError("permission")
    monkeypatch.setattr(telemetry, "system_cpu_times", failed)
    result = telemetry.HostTelemetryMonitor().sample([101])
    assert result["cpu"]["system_percent"] is None
    assert result["memory"]["application_bytes"] == 100


def test_process_detail_uses_cached_rss_and_unknown_for_new_worker(monkeypatch):
    readings(monkeypatch)
    rss_calls = []
    monkeypatch.setattr(telemetry,"process_memory",lambda pid:rss_calls.append(pid) or 123)
    monitor = telemetry.HostTelemetryMonitor()
    first = monitor.sample([101])
    assert first["processes"] == [dict(pid=101,resident_bytes=123,cpu_percent=None)]
    current = monitor.sample([101,103])
    assert current["sampled_at"] == first["sampled_at"]
    assert current["processes"] == [dict(pid=101,resident_bytes=123,cpu_percent=None),
                                   dict(pid=103,resident_bytes=None,cpu_percent=None)]
    assert rss_calls == [101]


def test_unified_api_reuses_pid_scope_budget_and_gpu_monitor(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        state = client.get("/api/state").json()
        client.headers["X-EIP-Token"] = state["session_token"]
        workbench = client.app.state.workbench
        captured = []
        def host(pids):
            captured.append(("host", set(pids)))
            return dict(sampled_at=100., interval_seconds=2, cpu={"system_percent": 30, "application_percent": 20},
                        memory={"application_bytes": 100},
                        processes=[dict(pid=pid,resident_bytes=100 if pid==os.getpid() else None,cpu_percent=None)
                                   for pid in pids])
        def gpu(pids):
            captured.append(("gpu", set(pids)))
            return dict(available=False, cards=[], reason="无 NVIDIA 显卡", sampled_at=101., interval_seconds=2)
        monkeypatch.setattr(workbench.host_telemetry, "sample", host)
        monkeypatch.setattr(workbench.gpu_memory, "sample", gpu)
        assert client.get("/api/resources/telemetry", headers={"X-EIP-Token": "wrong"}).status_code == 403
        result = client.get("/api/resources/telemetry").json()
        assert result["cpu"]["system_percent"] == 30
        assert result["sampled_at"] == 100 and result["gpu"]["sampled_at"] == 101
        assert result["activity"] == dict(busy=False, pending=0, job_name=None, status=None)
        budgets = result["budgets"]
        assert {key:budgets[key] for key in ("memory_mb","gpu_mb","preview_mb","cache_mb")} == {
            key:state["resources"][key] for key in ("memory_mb","gpu_mb","preview_mb","cache_mb")}
        assert budgets["processing_mb"] == workbench.resources.processing_mb
        assert budgets["job_mb"] == workbench.resources.job_mb
        assert budgets["inspection_mb"] == workbench.resources.inspection_mb
        assert result["cache"] == dict(frame_cache_bytes=0,frame_cache_count=0,
                                       frame_cache_limit_bytes=state["resources"]["preview_mb"]*1024**2//2)
        assert result["processes"] == [dict(name="工作台",pid=os.getpid(),resident_bytes=100,cpu_percent=None),
                                       dict(name="数据读取",pid=workbench.preview.process.pid,resident_bytes=None,cpu_percent=None)]
        assert captured[0][1] == captured[1][1]
        assert {os.getpid(), workbench.preview.process.pid} == captured[0][1]
        assert client.get("/api/state").json()["resources"] == state["resources"]
