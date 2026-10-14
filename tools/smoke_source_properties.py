"""Focused browser check: source properties describe data, not preview frames."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

import av
import httpx
import numpy as np
import tifffile
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eip.module_contracts import Contract


def typed_source(directory, contract):
    directory.mkdir()
    np.save(directory / "values.npy", np.arange(
        int(np.prod(contract.shape)) * contract.length, dtype=contract.dtype,
    ).reshape(contract.length, *contract.shape))
    offsets = []
    with (directory / "references.jsonl").open("wb") as references:
        for i in range(contract.length):
            offsets.append(references.tell())
            references.write((json.dumps(dict(sample_index=i, time=i*.01)) + "\n").encode())
    np.save(directory / "reference_offsets.npy", np.array(offsets, dtype=np.int64))
    manifest = directory / "dataset.json"
    manifest.write_text(json.dumps(dict(
        schema_version=1, status="succeeded", contract=contract.record(),
    )), encoding="utf-8")
    return manifest


def main():
    output = ROOT / ".test-output" / f"source-properties-{int(time.time())}"
    output.mkdir(parents=True)
    gray = np.arange(45, dtype=np.uint16).reshape(3, 3, 5)
    tifffile.imwrite(output / "sequence.tif", gray, photometric="minisblack", metadata={"axes": "TYX"})
    tifffile.imwrite(output / "single.tif", gray[0], photometric="minisblack")
    rgb = np.arange(135, dtype=np.uint8).reshape(3, 3, 5, 3)
    tifffile.imwrite(output / "color.tif", rgb, photometric="rgb", metadata={"axes": "TYXS"})
    with av.open(str(output / "video.mp4"), "w") as video:
        stream = video.add_stream("mpeg4", rate=25)
        stream.width, stream.height, stream.pix_fmt = 16, 16, "yuv420p"
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(np.full((16, 16, 3), i*80, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                video.mux(packet)
        for packet in stream.encode():
            video.mux(packet)
    fixtures = [
        ("sequence", output / "sequence.tif", "image", "TYX / gray", "T:3 × Y:3 × X:5", "uint16", "帧 / 样本数", "3"),
        ("single", output / "single.tif", "image", "TYX / gray", "T:1 × Y:3 × X:5", "uint16", "帧 / 样本数", "1"),
        ("color", output / "color.tif", "image", "TYXC / RGB", "T:3 × Y:3 × X:5 × C:3", "uint8", "帧 / 样本数", "3"),
        ("unindexed_video", output / "video.mp4", "video", "TYXC / RGB", "T:? × Y:16 × X:16 × C:3", "uint8", "帧 / 样本数", "未知"),
        ("time_signal", typed_source(output / "time_signal", Contract("", (), 4)), "dataset", "T / 不适用", "T:4", "float64", "帧 / 样本数", "4"),
        ("delay_signal", typed_source(output / "delay_signal", Contract("", (), 5, sequence_axis="Q", semantic="g2")), "dataset", "Q / 不适用", "Q:5", "float64", "延迟项数", "5"),
        ("static_slice", typed_source(output / "static_slice", Contract("YX", (3, 5), 1, sequence_axis="N")), "dataset", "YX / gray", "N:1 × Y:3 × X:5", "float64", "记录数", "1"),
    ]
    port = 8898
    url = f"http://127.0.0.1:{port}"
    with (output / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen(
            [sys.executable, str(ROOT / "run_web.py"), "--port", str(port), "--no-browser", "--workspace", str(output)],
            cwd=ROOT, stdout=log, stderr=log, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        try:
            for _ in range(100):
                try:
                    if httpx.get(url + "/api/state", timeout=.5).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(.1)
            else:
                raise AssertionError("isolated server did not start")
            with httpx.Client(base_url=url, timeout=30) as api, sync_playwright() as pw:
                api.headers["X-EIP-Token"] = api.get("/api/state").json()["session_token"]
                opened = []
                for name, path, kind, *expected in fixtures:
                    response = api.post("/api/sources/open", json=dict(path=str(path), kind=kind, interval=.01))
                    response.raise_for_status()
                    opened.append((name, response.json()["source"], expected))
                browser = pw.chromium.launch(channel="msedge" if sys.platform == "win32" else "chromium", headless=True)
                page = browser.new_page(viewport=dict(width=1600, height=1000))
                errors, calculations = [], []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.on("request", lambda r: calculations.append(r.url) if r.method == "POST" and (r.url.endswith("/api/jobs") or r.url.endswith("/api/modules/inspect") or r.url.endswith("/sequence")) else None)
                page.goto(url)
                if not page.locator("#library-sources").is_visible(): page.locator("#toggle-data-sources").click()
                report = {}
                for name, source, expected in opened:
                    page.locator(f"[data-info='{source['id']}']").click()
                    expect(page.locator("#source-dialog")).to_be_visible()
                    rows = page.locator("#source-detail .metadata-table tr").evaluate_all(
                        "rows => Object.fromEntries(rows.map(row => Array.from(row.cells, cell => cell.textContent.trim())))"
                    )
                    axes, dimensions, dtype, count_label, count = expected
                    assert rows["轴 / 颜色"] == axes, (name, rows)
                    assert rows["数据尺寸"] == dimensions, (name, rows)
                    assert rows["数值类型"] == dtype, (name, rows)
                    assert rows[count_label] == count, (name, rows)
                    report[name] = dict(axes=axes, dimensions=dimensions, dtype=dtype, count=count)
                    if name == "sequence":
                        page.screenshot(path=str(output / "sequence.png"), full_page=True)
                    if name == "time_signal":
                        page.set_viewport_size(dict(width=1100, height=760))
                        expect(page.locator("#source-dialog")).to_be_visible()
                        page.screenshot(path=str(output / "signal-compact.png"), full_page=True)
                    page.locator("#source-close").click()
                    expect(page.locator("#source-dialog")).not_to_be_visible()
                assert not api.get("/api/jobs").json(), "properties must not create processing jobs"
                assert not calculations, calculations
                assert not errors, errors
                report.update(javascript_errors=errors, calculation_requests=calculations)
                (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(json.dumps(dict(output=str(output), **report), ensure_ascii=False))
                browser.close()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


if __name__ == "__main__":
    main()
