"""One bounded, replaceable worker for interactive module inspection.

It owns a reader and uses the same SequenceEngine as saved runs. It never joins
the job queue, exports a dataset, or occupies the original-data preview worker.
"""
from __future__ import annotations

from dataclasses import replace
import multiprocessing
import tempfile
import threading
import time
import uuid

import numpy as np


class InspectionSuperseded(ValueError):
    pass


def inspection_entry(commands, results, folder):
    from eip.domain import source_from_dict
    from eip.readers import open_source
    from eip.sequence_engine import SequenceEngine

    reader, key = None, None
    try:
        while True:
            command = commands.get()
            identifier = command["id"]
            try:
                source = source_from_dict(command["source"])
                if key != source.key:
                    if reader:
                        reader.close()
                    reader, key = None, None
                    reader = open_source(source)
                    key = source.key
                if reader.meta.revision != command["revision"]:
                    raise ValueError("来源已改变，请重新导入")
                with tempfile.TemporaryDirectory(prefix="stage-", dir=folder) as workdir:
                    engine = SequenceEngine(
                        reader, command["nodes"], workdir=workdir, single=True,
                        input_kind=command.get("input_kind", "image"),
                        dependencies=command.get("dependencies", []),
                    )
                    try:
                        side_id, index = command.get("side_output"), command["index"]
                        side = next((s for s in engine.side_outputs() if s["id"]==side_id), None)
                        if side_id and side is None:
                            raise ValueError("此阶段没有指定的旁路输出")
                        c = side["contract"] if side else engine.output
                        if not 0 <= index < c.length:
                            raise ValueError("检查样本号超出编辑后的输出范围")
                        pixels = np.asarray(engine.get_side_output(side_id,index) if side else engine.get(len(engine.nodes),index))
                        if pixels.nbytes > 128 * 1024**2:
                            raise ValueError("单个检查样本超过 128 MiB，请先缩小空间范围")
                        if c.dtype == "dynamic":
                            c = replace(c, dtype=str(pixels.dtype))
                        reference = engine.side_ref(side_id,index) if side else engine.ref(len(engine.nodes),index)
                        event = dict(id=identifier, contract=c.record(), reference=reference,
                                     index=index, inspection=True, temporary=True, mappings=engine.mappings,
                                     side_output=side_id, node_id=side["node_id"] if side else next((
                                         n["id"] for n in reversed(engine.nodes)
                                         if n["algorithm"]=="distribution" and n["parameters"]["output"]=="main"
                                     ),None))
                        if c.spatial:
                            channel = command.get("channel", 0)
                            if c.paired:
                                if not 0 <= channel < pixels.shape[0]:
                                    raise ValueError("检查幅面超出范围")
                                pixels = pixels[channel]
                            elif channel != 0:
                                raise ValueError("此输入没有双路幅面")
                            event.update(kind="image", pixels=pixels,
                                         result_kind="coefficient" if c.semantic in {"g2", "processed_g2"}
                                         else "label" if c.semantic == "label" else "image")
                        else:
                            columns = max(1, pixels.size)
                            selected = np.unique(np.linspace(0, columns-1, min(columns, 4096), dtype=int))
                            values = pixels.reshape(-1)[selected]
                            event.update(kind="signals", profile=True, indices=[index],
                                         column_indices=selected.tolist(), references=[reference],
                                         values=[[float(v) if np.isfinite(v) else None for v in values]],
                                         downsampled=len(selected) < columns, total=c.length)
                    finally:
                        engine.close()
                results.send(event)
            except Exception as error:
                results.send(dict(id=identifier, kind="failed", message=str(error)))
    finally:
        if reader:
            reader.close()


class InspectionBroker:
    """Latest explicit inspection wins; at most one process and request survive."""

    def __init__(self):
        self.context = multiprocessing.get_context("spawn")
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.closed = False

    def _start(self):
        self.folder = tempfile.TemporaryDirectory(prefix="eip-inspection-")
        self.commands = self.context.Queue(maxsize=1)
        self.results, writer = self.context.Pipe(duplex=False)
        self.process = self.context.Process(target=inspection_entry,
                                           args=(self.commands, writer, self.folder.name),
                                           name="EIP-module-inspection")
        self.process.start()
        # Only the child keeps a write handle. Termination then unblocks recv,
        # including cancellation during transfer of a large sample.
        writer.close()

    def _dispose(self):
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(2)
        self.commands.cancel_join_thread()
        self.commands.close()
        self.results.close()
        self.process.close()
        self.process = None
        self.folder.cleanup()

    def request(self, source, *, inspection_id=None, timeout=120, **values):
        identifier = uuid.uuid4().hex
        with self.lock:
            if self.closed:
                raise ValueError("工作台正在关闭")
            if self.active or self.process is not None and not self.process.is_alive():
                self._dispose()
            if self.process is None:
                self._start()
            self.active = identifier
            self.request_id = inspection_id
            results = self.results
            self.commands.put(dict(id=identifier, source=source, **values), timeout=2)
        deadline = time.monotonic() + timeout
        try:
            while True:
                with self.lock:
                    if self.active != identifier:
                        raise InspectionSuperseded("阶段检查已取消或被新的检查替代")
                    if not self.process.is_alive():
                        raise ValueError("阶段检查进程已退出，请重新检查")
                try:
                    ready = results.poll(.2)
                    if ready:
                        result = results.recv()
                except (OSError, ValueError, EOFError) as error:
                    raise InspectionSuperseded("阶段检查已取消或被新的检查替代") from error
                if not ready:
                    if time.monotonic() >= deadline:
                        self.cancel(identifier)
                        raise ValueError("阶段检查超过两分钟，请缩小前序范围或使用已保存结果")
                    continue
                if result["id"] != identifier:
                    continue
                if result["kind"] == "failed":
                    raise ValueError(result["message"])
                return result
        finally:
            with self.lock:
                if self.active == identifier:
                    self.active = None

    def cancel(self, identifier=None):
        if identifier is not None and not isinstance(identifier, str):
            raise ValueError("检查标识无效")
        with self.lock:
            if self.active and (identifier is None or identifier in {self.active, self.request_id}):
                self.active = None
                self._dispose()
                return True
        return False

    def close(self):
        with self.lock:
            self.closed = True
            self.active = None
            self._dispose()
