"""Exercise real Qt interaction and spawned workers, and save UI captures."""
from __future__ import annotations

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import json
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import av
import numpy as np
import tifffile
from PySide6.QtWidgets import QApplication, QFileDialog

from eip.algorithms import load_manifest
from eip.domain import ROI, SourceSpec, StepSpec
from eip.ui.window import MainWindow
from eip.ui.fonts import configure_fonts


def pump(app, condition, label, timeout=40):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError(f"Timed out: {label}")


def make_video(path):
    stamps = [0, 40, 110, 160, 300, 350, 400, 510]
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width, stream.height = 96, 64
        stream.pix_fmt = "gray16le"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for index, pts in enumerate(stamps):
            pixels = np.full((64, 96), (index + 1) * 4000, dtype=np.uint16)
            pixels[20:44, 12 + index * 5:28 + index * 5] = 60000
            frame = av.VideoFrame.from_ndarray(pixels, format="gray16le")
            frame.pts = pts
            frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / ".test-output" / f"ui-{int(time.time())}"
    output.mkdir(parents=True)
    y, x = np.mgrid[:512, :640]
    pixels = (3000 + x / 640 * 24000 + 28000 * np.exp(-((x - 210) ** 2 + (y - 190) ** 2) / 5500) + 21000 * np.exp(-((x - 450) ** 2 + (y - 330) ** 2) / 2800)).astype(np.uint16)
    image_path = output / "scientific_uint16.tif"
    tifffile.imwrite(image_path, pixels)
    video_path = output / "variable_rate_uint16.mkv"
    make_video(video_path)
    app = QApplication([])
    configure_fonts(app)
    window = MainWindow()
    window._error = lambda message: (_ for _ in ()).throw(AssertionError(str(message)))
    window.show()
    try:
        window._request("open", SourceSpec(str(image_path)))
        pump(app, lambda: window.ready and window.frame is not None, "image open")
        np.testing.assert_array_equal(window.frame.pixels, pixels)
        window._set_roi(ROI(110, 100, 360, 280))
        window.steps = [StepSpec("intensity", {"gain": 1.3, "offset": -2000})]
        window._refresh_steps()
        window._config_changed()
        window._trial()
        pump(app, lambda: window.result_fresh, "spawned trial")
        pump(app, lambda: window.jobs.active is None, "trial exit")
        expected = np.clip(np.rint(pixels.astype(float) * 1.3 - 2000), 0, 65535).astype(np.uint16)[100:380, 110:470]
        np.testing.assert_array_equal(window.result_pixels, expected)
        app.processEvents()
        assert window.input_canvas.transform().m11() < 1, "A new result must not reset input zoom to 100%"
        assert window.run_button.isVisible()
        window.grab().save(str(output / "workbench-1440.png"))
        window.resize(1024, 720)
        app.processEvents()
        assert window.width() == 1024 and window.run_button.isVisible()
        window.input_canvas.fit_image()
        app.processEvents()
        window.grab().save(str(output / "workbench-1024.png"))
        window.resize(1440, 900)
        project_path = output / "project.eip.json"
        with patch.object(QFileDialog, "getSaveFileName", return_value=(str(project_path), "")):
            window._save_project()
        window._set_roi(None)
        with patch.object(QFileDialog, "getOpenFileName", return_value=(str(project_path), "")):
            window._open_project()
        pump(app, lambda: window.ready and window.roi is not None, "project restoration")
        assert window.roi == ROI(110, 100, 360, 280)
        assert window.steps[0].parameters["gain"] == 1.3
        window._set_roi(None)
        window._request("open", SourceSpec(str(video_path)))
        pump(app, lambda: window.ready and window.source.path == str(video_path), "video open")
        window._request("time", time=0.08)
        pump(app, lambda: window.ready and window.frame.ref.time == 0.11, "PTS seek")
        assert window.frame.pixels[0, 0] == 12000
        window._step_frame(-1)
        pump(app, lambda: window.ready and window.frame.ref.time == 0.04, "backward step")
        window._step_frame(1)
        pump(app, lambda: window.ready and window.frame.ref.time == 0.11, "forward step")
        plugin = load_manifest(str(root / "examples/plugins/invert.json"))
        window.algorithm_specs[plugin.id] = plugin
        window.steps = [StepSpec(plugin.id, {"reference": 65535}, manifest=plugin.manifest, code_hash=plugin.code_hash)]
        window._refresh_steps()
        window._config_changed()
        window._trial()
        pump(app, lambda: window.result_fresh, "external plugin trial")
        pump(app, lambda: window.jobs.active is None, "plugin worker exit")
        assert window.result_pixels[0, 0] == 53535
        window.grab().save(str(output / "video-plugin.png"))
        window.steps = [StepSpec("intensity", {"gain": 1, "offset": 0})]
        window._refresh_steps()
        window._config_changed()
        window.range_mode.setCurrentIndex(2)
        window.output_edit.setText(str(output / "runs"))
        window.batch.setChecked(True)
        for item in window.items.values():
            item.setSelected(True)
        window._run()
        pump(app, lambda: window.result_root.childCount() == 2, "batch export")
        pump(app, lambda: window.jobs.active is None, "batch exit")
        runs = list((output / "runs").glob("*/run.json"))
        assert len(runs) == 1
        record = json.loads(runs[0].read_text(encoding="utf-8"))
        assert [item["completed_frames"] for item in record["results"]] == [1, 8]
        assert record["status"] == "succeeded"
        # Selecting a result loads only the result pane, and keeps processing input.
        input_before = window.source
        result = window.result_root.child(0)
        window.tree.setCurrentItem(result)
        pump(app, lambda: "正式结果" in window.result_header.text(), "result reopen")
        assert window.source == input_before
        window._step_frame(1)
        pump(app, lambda: window.ready and window.frame.ref.time == 0.16, "step after result view")
        assert window.preview.process.is_alive()
        plugin_path = output / "failure.py"
        plugin_path.write_text("raise RuntimeError('deliberate plugin failure')\n", encoding="utf-8")
        manifest_path = output / "failure.json"
        manifest_path.write_text(json.dumps({"schema_version": 1, "id": "smoke.failure", "name": "故障验证", "entry_point": "failure.py:Broken"}), encoding="utf-8")
        bad_spec = load_manifest(str(manifest_path))  # Does not execute the module.
        window.algorithm_specs[bad_spec.id] = bad_spec
        window.steps = [StepSpec(bad_spec.id, manifest=bad_spec.manifest, code_hash=bad_spec.code_hash)]
        window._refresh_steps()
        window._config_changed()
        window._trial()
        row = window.task_table.rowCount() - 1
        pump(app, lambda: window.task_table.item(row, 2).text() == "失败", "plugin failure isolation")
        pump(app, lambda: window.jobs.active is None, "failed worker exit")
        assert window.preview.process.is_alive() and window.ready
        slow_path = output / "slow.py"
        slow_path.write_text("import time\nclass Slow:\n    def prepare(self, parameters, context): pass\n    def process(self, pixels, context):\n        time.sleep(0.15)\n        context.check_cancelled()\n        return pixels.copy()\n    def close(self): pass\n", encoding="utf-8")
        slow_manifest = output / "slow.json"
        slow_manifest.write_text(json.dumps({"schema_version": 1, "id": "smoke.slow", "name": "取消验证", "entry_point": "slow.py:Slow"}), encoding="utf-8")
        slow_spec = load_manifest(str(slow_manifest))
        window.algorithm_specs[slow_spec.id] = slow_spec
        window.steps = [StepSpec(slow_spec.id, manifest=slow_spec.manifest, code_hash=slow_spec.code_hash)]
        window.batch.setChecked(False)
        window._refresh_steps()
        window._config_changed()
        window._run()
        row = window.task_table.rowCount() - 1
        identifier = next(key for key, value in window.job_rows.items() if value == row)
        pump(app, lambda: "1 帧" in window.task_table.item(row, 3).text(), "first frame before cancellation")
        window.jobs.cancel(identifier)
        pump(app, lambda: window.task_table.item(row, 2).text() == "已取消", "worker cancellation")
        pump(app, lambda: window.jobs.active is None, "cancelled worker exit")
        assert (output / "runs" / f"{identifier}.partial").is_dir()
        assert not (output / "runs" / identifier).exists()
        print(json.dumps({"status": "passed", "checks": ["uint16 read", "worker trial", "ROI mapping", "project restore", "VFR seek/step", "custom plugin", "batch export", "result/input isolation", "plugin failure isolation", "worker cancellation"], "screenshots": [str(output / "workbench-1440.png"), str(output / "workbench-1024.png"), str(output / "video-plugin.png")]}, ensure_ascii=False))
    finally:
        window.jobs.close()
        window.preview.close()
        window.hide()
    return 0


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    raise SystemExit(main())
