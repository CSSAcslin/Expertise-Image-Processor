from collections import deque
import json
import multiprocessing
import queue
import time

from PySide6.QtCore import QObject, QTimer, Signal

from eip.preview_worker import preview_entry
from eip.processing import job_entry


class PreviewService(QObject):
    received = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = multiprocessing.get_context("spawn")
        self.commands = self.context.Queue(maxsize=2)
        self.results = self.context.Queue(maxsize=2)
        self.process = self.context.Process(target=preview_entry, args=(self.commands, self.results), name="EIP-preview")
        self.process.start()
        self.pending = deque()
        self.in_flight = None
        self.latest_id = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(20)

    @property
    def busy(self):
        return self.in_flight is not None or bool(self.pending)

    def submit(self, action: str, source: dict, **values) -> int:
        if not self.process.is_alive():
            raise RuntimeError("预览进程已停止，请重新启动应用")
        self.latest_id += 1
        command = {"id": self.latest_id, "action": action, "source": source, **values}
        if action != "open":
            self.pending = deque(item for item in self.pending if item["action"] == "open")
        if len(self.pending) >= 32:
            raise ValueError("打开请求过多，请等待当前数据读取完成")
        self.pending.append(command)
        self.dispatch()
        return self.latest_id

    def dispatch(self):
        if self.in_flight is None and self.pending:
            command = self.pending.popleft()
            self.in_flight = command["id"]
            self.commands.put_nowait(command)

    def poll(self):
        try:
            event = self.results.get_nowait()
        except queue.Empty:
            if self.in_flight is not None and not self.process.is_alive():
                self.received.emit({"id": self.in_flight, "type": "failed", "message": "预览进程意外退出，请重新启动应用"})
                self.in_flight = None
                self.pending.clear()
            return
        self.in_flight = None
        self.received.emit(event)
        self.dispatch()

    def close(self):
        self.timer.stop()
        self.pending.clear()
        if self.process.is_alive():
            try:
                self.commands.put_nowait({"action": "stop"})
            except queue.Full:
                pass
            self.process.join(0.4)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(0.4)
        self.commands.cancel_join_thread()
        self.results.cancel_join_thread()
        self.commands.close()
        self.results.close()


class JobService(QObject):
    event = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = multiprocessing.get_context("spawn")
        self.pending = deque()
        self.active = None
        self.process = None
        self.results = None
        self.cancel_event = None
        self.saw_terminal = False
        self.dead_since = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(30)

    def submit(self, payload: dict):
        if len(self.pending) >= 32:
            raise ValueError("任务队列已满，请等待或取消任务")
        payload = json.loads(json.dumps(payload, allow_nan=False))
        self.pending.append(payload)
        self.event.emit({"id": payload["id"], "type": "queued", "payload": payload})
        self.start_next()

    def start_next(self):
        if self.active is not None or not self.pending:
            return
        self.active = self.pending.popleft()
        self.results = self.context.Queue(maxsize=64)
        self.cancel_event = self.context.Event()
        self.process = self.context.Process(target=job_entry, args=(self.active, self.results, self.cancel_event), name="EIP-compute")
        self.saw_terminal = False
        self.dead_since = None
        self.process.start()
        self.event.emit({"id": self.active["id"], "type": "running"})

    def cancel(self, identifier: str | None = None):
        identifier = identifier or (self.active["id"] if self.active else None)
        if self.active and self.active["id"] == identifier:
            self.cancel_event.set()
            self.event.emit({"id": identifier, "type": "cancelling"})
        else:
            for item in list(self.pending):
                if item["id"] == identifier:
                    self.pending.remove(item)
                    self.event.emit({"id": identifier, "type": "cancelled"})

    def poll(self):
        if self.active is None:
            return
        for _ in range(128):
            try:
                event = self.results.get_nowait()
            except queue.Empty:
                break
            event["id"] = self.active["id"]
            if event["type"] in {"completed", "cancelled", "failed", "trial"}:
                self.saw_terminal = True
            if event["type"] != "worker_done":
                self.event.emit(event)
        if not self.process.is_alive():
            self.dead_since = self.dead_since or time.monotonic()
            if time.monotonic() - self.dead_since < 0.15:
                return
            if not self.saw_terminal:
                self.event.emit({"id": self.active["id"], "type": "failed", "message": "计算进程意外退出；临时输出未标记为成功"})
            self.process.join()
            self.results.cancel_join_thread()
            self.results.close()
            self.active = None
            self.process = None
            self.start_next()

    def close(self):
        self.timer.stop()
        self.pending.clear()
        if self.process:
            self.cancel_event.set()
            self.process.join(0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(0.4)
            self.results.cancel_join_thread()
            self.results.close()
