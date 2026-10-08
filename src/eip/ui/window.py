from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import json
import uuid

import numpy as np
import tifffile
from PySide6.QtCore import QItemSelectionModel, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QCheckBox, QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from eip.algorithms import BUILTINS, load_manifest, validate_parameters
from eip.display import DisplaySettings, display_range, labels_to_display, to_display
from eip.domain import FrameData, FrameRef, ROI, Selection, SourceSpec, StepSpec
from eip.persistence import read_project, write_json
from eip.ui.canvas import ImageCanvas
from eip.ui.services import JobService, PreviewService


STYLE = """
QMainWindow, QWidget { background: #202731; color: #e4eaf1; font-size: 12px; }
QToolBar { border: none; spacing: 7px; padding: 6px; background: #28323e; }
QDockWidget::title { background: #2c3744; padding: 7px; font-weight: 600; }
QTreeWidget, QListWidget, QTableWidget, QTextEdit { background: #171e27; border: 1px solid #3a4756; selection-background-color: #254d70; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #151e28; border: 1px solid #465365; padding: 4px; min-height: 20px; }
QPushButton { background: #334254; border: 1px solid #526276; border-radius: 3px; padding: 6px 10px; }
QPushButton:hover { background: #435b72; }
QPushButton:disabled { color: #7e8997; background: #28323e; }
QPushButton#primary { background: #247cab; border: 1px solid #3895c5; font-weight: 600; }
QLabel#caption { background: #283442; padding: 6px; }
QHeaderView::section { background: #2c3948; padding: 5px; border: none; }
QStatusBar { background: #18202a; }
QSplitter::handle { background: #364354; }
QToolTip { color: #ecf4fc; background: #344355; border: 1px solid #63758a; }
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Expertise Image Processor · 视频预览与数据处理")
        self.resize(1440, 900)
        self.setMinimumSize(1000, 700)
        self.setStyleSheet(STYLE)
        self.sources = {}
        self.items = {}
        self.source = None
        self.meta = None
        self.frame = None
        self.roi = None
        self.result_pixels = None
        self.result_kind = "image"
        self.result_origin = (0, 0)
        self.result_fresh = False
        self.result_snapshot = None
        self.config_revision = 0
        self.request_id = None
        self.ready = False
        self.result_requests = {}
        self.job_rows = {}
        self.job_specs = {}
        self.algorithm_specs = {spec.id: spec for spec in BUILTINS}
        self.steps = [StepSpec("intensity", BUILTINS[0].defaults())]
        self.syncing = False
        self.timeline_updating = False
        self.project_path = None
        self._build_toolbar()
        self._build_preview()
        self._build_sources()
        self._build_processing()
        self._build_tasks()
        self.preview = PreviewService(self)
        self.preview.received.connect(self._preview_event)
        self.jobs = JobService(self)
        self.jobs.event.connect(self._job_event)
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        self._refresh_steps()
        self.statusBar().showMessage("打开图像、视频或图像序列，然后配置处理。显示调整不会改变算法输入。")

    def _action(self, toolbar, text, callback, shortcut=None):
        action = QAction(text, self)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(shortcut)
        toolbar.addAction(action)
        return action

    def _build_toolbar(self):
        bar = self.addToolBar("工作台")
        bar.setMovable(False)
        self._action(bar, "打开文件", self._open_files, "Ctrl+O")
        self._action(bar, "图像序列", self._open_sequence)
        bar.addSeparator()
        self._action(bar, "打开项目", self._open_project)
        self._action(bar, "保存项目", self._save_project, "Ctrl+S")
        bar.addSeparator()
        self._action(bar, "适应窗口", lambda: self.input_canvas.fit_image(), "F")
        self.roi_action = QAction("矩形 ROI", self)
        self.roi_action.setCheckable(True)
        self.roi_action.toggled.connect(lambda value: self.input_canvas.set_roi_mode(value))
        bar.addAction(self.roi_action)
        self._action(bar, "清除 ROI", lambda: self._set_roi(None))
        bar.addSeparator()
        self._action(bar, "注册算法", self._register_plugin)
        self._action(bar, "输出目录", lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_edit.text())))

    def _build_preview(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        view_bar = QHBoxLayout()
        view_bar.addWidget(QLabel("预览"))
        self.view_mode = QComboBox()
        self.view_mode.addItems(["输入与结果并排", "仅输入", "仅结果"])
        self.view_mode.currentIndexChanged.connect(self._view_mode_changed)
        view_bar.addWidget(self.view_mode)
        self.sync_views = QCheckBox("同步视图")
        self.sync_views.setChecked(True)
        view_bar.addWidget(self.sync_views)
        view_bar.addStretch()
        layout.addLayout(view_bar)
        self.view_splitter = QSplitter()
        self.input_canvas = ImageCanvas()
        self.result_canvas = ImageCanvas()
        self.input_header = QLabel("输入 · 尚未打开数据")
        self.result_header = QLabel("结果 · 配置处理后点击“精确试算”")
        for label, canvas in [(self.input_header, self.input_canvas), (self.result_header, self.result_canvas)]:
            panel = QWidget()
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(0, 0, 0, 0)
            label.setObjectName("caption")
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            panel_layout.addWidget(label)
            panel_layout.addWidget(canvas, 1)
            self.view_splitter.addWidget(panel)
        self.view_splitter.setSizes([500, 500])
        layout.addWidget(self.view_splitter, 1)
        self.input_canvas.roi_changed.connect(self._set_roi)
        self.input_canvas.pixel_moved.connect(lambda x, y: self._pixel_value(x, y, False))
        self.result_canvas.pixel_moved.connect(lambda x, y: self._pixel_value(x, y, True))
        self.input_canvas.view_changed.connect(lambda: self.result_canvas.sync_from(self.input_canvas) if self.sync_views.isChecked() else None)
        self.result_canvas.view_changed.connect(lambda: self.input_canvas.sync_from(self.result_canvas) if self.sync_views.isChecked() else None)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.sliderPressed.connect(self._pause)
        self.timeline.sliderReleased.connect(self._seek_slider)
        self.timeline.valueChanged.connect(self._slider_changed)
        layout.addWidget(self.timeline)
        controls = QHBoxLayout()
        for text, callback in [("首帧", self._first_frame), ("上一帧", lambda: self._step_frame(-1))]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self._toggle_play)
        controls.addWidget(self.play_button)
        button = QPushButton("下一帧")
        button.clicked.connect(lambda: self._step_frame(1))
        controls.addWidget(button)
        self.speed = QComboBox()
        self.speed.addItems(["0.25×", "0.5×", "1×", "2×", "4×"])
        self.speed.setCurrentIndex(2)
        self.speed.currentIndexChanged.connect(self._update_speed)
        controls.addWidget(self.speed)
        self.position_label = QLabel("帧 —   时间 —")
        self.position_label.setWordWrap(True)
        self.position_label.setMinimumWidth(100)
        self.position_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        controls.addWidget(self.position_label, 1)
        layout.addLayout(controls)
        display_controls = QHBoxLayout()
        display_controls.addWidget(QLabel("显示范围"))
        self.display_low = QDoubleSpinBox()
        self.display_high = QDoubleSpinBox()
        for spin in [self.display_low, self.display_high]:
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(3)
            spin.setMaximumWidth(110)
            spin.editingFinished.connect(self._render)
            display_controls.addWidget(spin)
        self.display_high.setValue(65535)
        auto = QPushButton("自动")
        auto.clicked.connect(self._auto_display)
        display_controls.addWidget(auto)
        display_controls.addWidget(QLabel("伽马"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.05, 10)
        self.gamma.setValue(1)
        self.gamma.setSingleStep(0.1)
        self.gamma.setMaximumWidth(70)
        self.gamma.valueChanged.connect(self._render)
        display_controls.addWidget(self.gamma)
        self.colormap = QComboBox()
        self.colormap.addItems(["灰度 / RGB", "热度伪彩"])
        self.colormap.currentIndexChanged.connect(self._render)
        display_controls.addWidget(self.colormap)
        self.pixel_label = QLabel("像素原值：—")
        self.pixel_label.setWordWrap(True)
        layout.addLayout(display_controls)
        layout.addWidget(self.pixel_label)
        self.setCentralWidget(central)

    def _dock(self, title, widget, area):
        dock = QDockWidget(title, self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _build_sources(self):
        panel = QWidget()
        panel.setMinimumWidth(160)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.input_root = QTreeWidgetItem(self.tree, ["输入数据"])
        self.result_root = QTreeWidgetItem(self.tree, ["处理结果"])
        self.input_root.setExpanded(True)
        self.result_root.setExpanded(True)
        self.tree.currentItemChanged.connect(self._tree_changed)
        layout.addWidget(self.tree, 1)
        self.properties = QLabel("支持图像、序列与常规视频。\n序列目录可指定时间间隔。")
        self.properties.setWordWrap(True)
        self.properties.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.properties)
        self._dock("数据源与结果", panel, Qt.DockWidgetArea.LeftDockWidgetArea)

    def _build_processing(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.algorithm_combo = QComboBox()
        for spec in BUILTINS:
            self.algorithm_combo.addItem(spec.name, spec.id)
        layout.addWidget(self.algorithm_combo)
        buttons = QHBoxLayout()
        for text, callback in [("添加", self._add_step), ("移除", self._remove_step), ("↑", lambda: self._move_step(-1)), ("↓", lambda: self._move_step(1))]:
            button = QPushButton(text)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.step_list = QListWidget()
        self.step_list.setMaximumHeight(155)
        self.step_list.currentRowChanged.connect(self._build_parameters)
        self.step_list.itemChanged.connect(self._step_checked)
        layout.addWidget(self.step_list)
        self.algorithm_description = QLabel()
        self.algorithm_description.setWordWrap(True)
        layout.addWidget(self.algorithm_description)
        self.parameter_panel = QWidget()
        self.parameter_form = QFormLayout(self.parameter_panel)
        self.parameter_form.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.parameter_panel)
        self.roi_label = QLabel("范围：全图；ROI 为全帧处理后的输出裁剪")
        self.roi_label.setWordWrap(True)
        layout.addWidget(self.roi_label)
        self.range_mode = QComboBox()
        self.range_mode.addItems(["当前帧", "指定范围", "全部数据"])
        self.range_mode.setCurrentIndex(2)
        self.range_mode.currentIndexChanged.connect(self._range_changed)
        layout.addWidget(self.range_mode)
        self.range_form = QFormLayout()
        self.range_start = QDoubleSpinBox()
        self.range_end = QDoubleSpinBox()
        for spin in [self.range_start, self.range_end]:
            spin.setRange(0, 1e9)
            spin.valueChanged.connect(self._config_changed)
        self.range_form.addRow("起点", self.range_start)
        self.range_form.addRow("终点（不含）", self.range_end)
        layout.addLayout(self.range_form)
        self.batch = QCheckBox("批量：处理选中的输入数据")
        self.batch.setToolTip("在左侧按 Ctrl / Shift 选择多个输入；批量不使用当前帧模式。")
        layout.addWidget(self.batch)
        layout.addWidget(QLabel("输出目录（新建独立运行目录）"))
        out = QHBoxLayout()
        self.output_edit = QLineEdit(str(Path.cwd() / "outputs"))
        out.addWidget(self.output_edit)
        browse = QPushButton("…")
        browse.setMaximumWidth(32)
        browse.clicked.connect(self._choose_output)
        out.addWidget(browse)
        layout.addLayout(out)
        outer = QWidget()
        outer.setMinimumWidth(270)
        outer.setMaximumWidth(390)
        footer = QVBoxLayout(outer)
        footer.setContentsMargins(5, 5, 5, 5)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(panel)
        footer.addWidget(scroll, 1)
        layout = footer
        self.trial_button = QPushButton("精确试算 · 当前帧")
        self.trial_button.clicked.connect(self._trial)
        self.run_button = QPushButton("运行处理")
        self.run_button.setObjectName("primary")
        self.run_button.clicked.connect(self._run)
        self.export_button = QPushButton("保存试算结果")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_trial)
        layout.addWidget(self.trial_button)
        layout.addWidget(self.run_button)
        layout.addWidget(self.export_button)
        note = QLabel("内置算法：CPU · 插件执行范围：逐帧\n模型运行接口已预留，后续接入推理运行时。")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._dock("数据处理", outer, Qt.DockWidgetArea.RightDockWidgetArea)
        self._range_changed()

    def _build_tasks(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("后台任务"))
        cancel = QPushButton("取消选中 / 当前任务")
        cancel.clicked.connect(self._cancel_job)
        bar.addWidget(cancel)
        self.logs_button = QPushButton("显示日志")
        self.logs_button.setCheckable(True)
        self.logs_button.toggled.connect(lambda shown: self.log_view.setVisible(shown))
        bar.addWidget(self.logs_button)
        bar.addStretch()
        layout.addLayout(bar)
        self.task_table = QTableWidget(0, 4)
        self.task_table.setHorizontalHeaderLabels(["任务", "来源", "状态", "进度"])
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.task_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.task_table.verticalHeader().hide()
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_table.setColumnWidth(0, 190)
        self.task_table.setColumnWidth(1, 220)
        self.task_table.setColumnWidth(2, 120)
        layout.addWidget(self.task_table)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(140)
        self.log_view.document().setMaximumBlockCount(2000)
        self.log_view.hide()
        layout.addWidget(self.log_view)
        self.task_dock = self._dock("任务与日志", panel, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.resizeDocks([self.task_dock], [170], Qt.Orientation.Vertical)

    def _error(self, message):
        self.log_view.append(str(message))
        self.statusBar().showMessage(str(message))
        QMessageBox.warning(self, "操作未完成", str(message))

    def _open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "打开图像或视频", "", "数据 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mpg *.mpeg);;所有文件 (*)")
        for path in files[:32]:
            self._request("open", SourceSpec(str(Path(path).resolve())))
        if len(files) > 32:
            self._error("一次最多打开 32 个文件，其余文件可分次添加")

    def _open_sequence(self):
        path = QFileDialog.getExistingDirectory(self, "选择图像序列目录")
        if not path:
            return
        interval, accepted = QInputDialog.getDouble(self, "序列时间间隔", "相邻帧间隔（秒，用户指定）：\n文件采用自然排序，请确认命名对应实际顺序。", 1 / 30, 0.000001, 100000, 6)
        if accepted:
            self._request("open", SourceSpec(str(Path(path).resolve()), "sequence", interval))

    def _request(self, action, source=None, **values):
        source = source or self.source
        if source is None:
            return
        try:
            self.ready = False
            self.request_id = self.preview.submit(action, asdict(source), **values)
            self.statusBar().showMessage("正在读取数据…")
        except Exception as error:
            self._error(error)

    def _tree_changed(self, item, previous):
        if self.syncing or item is None:
            return
        value = item.data(0, Qt.ItemDataRole.UserRole)
        if not value:
            return
        self._pause()
        source = SourceSpec(**value["source"])
        if value.get("result"):
            try:
                identifier = self.preview.submit("open", asdict(source))
                self.result_requests[identifier] = value
            except Exception as error:
                self._error(error)
        else:
            self._request("open", source)

    def _preview_event(self, event):
        result_request = self.result_requests.pop(event["id"], None)
        if result_request is not None:
            if event["type"] == "frame":
                self.result_pixels = event["pixels"]
                self.result_kind = result_request.get("result_kind", "image")
                self.result_origin = tuple(result_request.get("origin", [0, 0])) if self.source and result_request.get("input_key") == self.source.key else (0, 0)
                self.result_fresh = False
                self.export_button.setEnabled(False)
                self.result_header.setText(f"正式结果 · {Path(result_request['source']['path']).name} · 原输入保持不变")
                self._render()
            else:
                self._error(event.get("message", "结果无法读取"))
            return
        if event["type"] == "frame" and event["action"] == "open":
            source = SourceSpec(**event["source"])
            self.sources[source.key] = {"source": source, "meta": event["meta"]}
            if source.key not in self.items:
                item = QTreeWidgetItem(self.input_root, [event["meta"]["name"]])
                item.setData(0, Qt.ItemDataRole.UserRole, {"source": event["source"]})
                self.items[source.key] = item
        if event["id"] != self.request_id:
            return
        if event["type"] == "eof":
            self._pause()
            self.ready = True
            self.statusBar().showMessage("已到视频末尾")
            return
        if event["type"] == "failed":
            self._pause()
            self.ready = False
            self._error(event["message"])
            return
        source = SourceSpec(**event["source"])
        changed = self.source is None or self.source.key != source.key or self.meta["revision"] != event["meta"]["revision"]
        self.source = source
        self.meta = event["meta"]
        self.frame = FrameData(event["pixels"], FrameRef(**event["frame"]))
        self.ready = True
        if changed:
            self.roi = None
            self.input_canvas.set_roi(None)
            self.result_canvas.set_roi(None)
            self.result_pixels = None
            self.result_canvas.clear_image()
            self._set_range_limits()
            self._auto_display(render=False)
        self._config_changed()
        self.input_header.setText(f"输入 · {self.meta['name']} · {self.meta['dtype']} · {self.meta['axes']}")
        self._render(fit=changed)
        self._update_position()
        self.properties.setText(f"{self.meta['name']}\n尺寸：{' × '.join(map(str, self.meta['shape']))}\n类型：{self.meta['dtype']}\n轴：{self.meta['axes']}\n帧数：{self.meta['frame_count'] or '未知'}\n颜色：{self.meta['color']}\n时间来源：{self.meta['timestamp_source']}\n" + "\n".join(self.meta["notes"]))
        self.syncing = True
        if source.key in self.items:
            self.tree.setCurrentItem(self.items[source.key], 0, QItemSelectionModel.SelectionFlag.NoUpdate)
        self.syncing = False
        if hasattr(self, "_pending_project_view"):
            value = self._pending_project_view
            del self._pending_project_view
            if value.get("display"):
                display = DisplaySettings(**value["display"])
                if display.low is not None and display.high is not None:
                    self.display_low.setValue(display.low)
                    self.display_high.setValue(display.high)
                self.gamma.setValue(display.gamma)
                self.colormap.setCurrentIndex(int(display.colormap == "heat"))
            if value.get("roi"):
                restored_roi = ROI(**value["roi"])
                try:
                    restored_roi.validate(self.frame.pixels.shape)
                    self._set_roi(restored_roi)
                except ValueError as error:
                    self._error(error)
            self.range_mode.setCurrentIndex(value.get("range_mode", 2))
            self.range_start.setValue(value.get("range_start", self.range_start.value()))
            self.range_end.setValue(value.get("range_end", self.range_end.value()))
            self._render()
        self.statusBar().showMessage("读取完成；精确试算使用原始精度，显示调整只作用于画面。")

    def _display_settings(self):
        return DisplaySettings(self.display_low.value(), self.display_high.value(), self.gamma.value(), "heat" if self.colormap.currentIndex() else "gray")

    def _render(self, *args, fit=False):
        if self.frame is None:
            return
        try:
            settings = self._display_settings()
            self.input_canvas.set_image(to_display(self.frame.pixels, settings), fit=fit)
            if self.result_pixels is not None:
                rendered_result = labels_to_display(self.result_pixels) if self.result_kind == "label" else to_display(self.result_pixels, settings)
                self.result_canvas.set_image(rendered_result, self.result_origin, self.frame.pixels.shape[:2], fit=fit)
                if self.sync_views.isChecked():
                    self.result_canvas.sync_from(self.input_canvas)
        except ValueError as error:
            self.statusBar().showMessage(str(error))

    def _auto_display(self, *args, render=True):
        if self.frame is not None:
            low, high = display_range(self.frame.pixels)
            self.display_low.setValue(low)
            self.display_high.setValue(high)
            if render:
                self._render()

    def _view_mode_changed(self, index):
        self.view_splitter.widget(0).setVisible(index != 2)
        self.view_splitter.widget(1).setVisible(index != 1)

    def _pixel_value(self, x, y, result):
        pixels = self.result_pixels if result else self.frame.pixels if self.frame else None
        if result:
            x -= self.result_origin[0]
            y -= self.result_origin[1]
        if pixels is not None and 0 <= y < pixels.shape[0] and 0 <= x < pixels.shape[1]:
            value = pixels[y, x]
            text = np.array2string(value, precision=4) if isinstance(value, np.ndarray) else str(value)
            self.pixel_label.setText(f"{'结果' if result else '输入'}像素原值  x={x}, y={y}：{text}")

    def _set_roi(self, roi):
        self.roi = roi
        self.input_canvas.set_roi(roi)
        self.result_canvas.set_roi(roi)
        self.roi_label.setText(f"ROI：x={roi.x}, y={roi.y}, {roi.width} × {roi.height}\n全帧处理后裁剪输出" if roi else "范围：全图；ROI 为全帧处理后的输出裁剪")
        self._config_changed()

    def _config_changed(self, *args):
        self.config_revision += 1
        self.result_fresh = False
        self.export_button.setEnabled(False)
        if self.result_pixels is not None:
            self.result_header.setText("结果 · 配置或位置已变化，显示的是上次结果；请重新试算")

    def _set_range_limits(self):
        video = self.meta["kind"] == "video"
        for spin in [self.range_start, self.range_end]:
            spin.setDecimals(3 if video else 0)
            spin.setRange(0 if video else 1, self.meta["duration"] or 1e9 if video else (self.meta["frame_count"] or 1) + 1)
        self.range_start.setValue(0 if video else 1)
        self.range_end.setValue(self.meta["duration"] or 1 if video else (self.meta["frame_count"] or 1) + 1)
        self.range_form.labelForField(self.range_start).setText("起始时间（秒）" if video else "起始帧（从 1 起）")
        self.range_form.labelForField(self.range_end).setText("结束时间（不含）" if video else "结束帧（不含）")
        self._set_roi(None)

    def _range_changed(self, *args):
        enabled = self.range_mode.currentIndex() == 1
        self.range_start.setEnabled(enabled)
        self.range_end.setEnabled(enabled)
        self._config_changed()

    def _update_position(self):
        if self.frame is None:
            return
        ref = self.frame.ref
        count = self.meta["frame_count"]
        ordinal = str(ref.index + 1) if ref.index is not None else "未知索引"
        stamp = f"{ref.time:.6f} s" if ref.time is not None else "无时间信息"
        self.position_label.setText(f"帧 {ordinal} / {count or '未知'}   ·   {stamp}")
        self.timeline_updating = True
        if self.meta["kind"] == "video":
            duration = self.meta["duration"]
            self.timeline.setRange(0, 10000)
            self.timeline.setEnabled(bool(duration))
            if duration:
                self.timeline.setValue(min(10000, max(0, int((ref.time or 0) / duration * 10000))))
        else:
            self.timeline.setRange(0, max(0, (count or 1) - 1))
            self.timeline.setEnabled(bool(count and count > 1))
            self.timeline.setValue(ref.index or 0)
        self.timeline_updating = False

    def _slider_changed(self, value):
        if not self.timeline_updating and not self.timeline.isSliderDown():
            self._seek_slider()

    def _seek_slider(self):
        if self.source is None:
            return
        self._pause()
        if self.meta["kind"] == "video":
            self._request("time", time=self.timeline.value() / 10000 * self.meta["duration"])
        else:
            self._request("index", index=self.timeline.value())

    def _first_frame(self):
        self._pause()
        if self.source:
            self._request("time", time=0) if self.meta["kind"] == "video" else self._request("index", index=0)

    def _step_frame(self, direction, playback=False):
        if not playback:
            self._pause()
        if not self.ready or self.frame is None:
            return
        if self.meta["kind"] == "video":
            self._request("next" if direction > 0 else "previous", time=self.frame.ref.time or 0)
        else:
            index = self.frame.ref.index + direction
            if 0 <= index < self.meta["frame_count"]:
                self._request("index", index=index)
            else:
                self._pause()

    def _toggle_play(self):
        if self.play_timer.isActive():
            self._pause()
        elif self.ready and self.meta and self.meta["kind"] != "image":
            self._update_speed()
            self.play_timer.start()
            self.play_button.setText("暂停")

    def _pause(self):
        self.play_timer.stop()
        self.play_button.setText("播放")

    def _update_speed(self, *args):
        multiplier = [0.25, 0.5, 1, 2, 4][self.speed.currentIndex()]
        rate = (self.meta.get("rate") or 30) if self.meta else 30
        self.play_timer.setInterval(max(5, int(1000 / rate / multiplier)))

    def _play_tick(self):
        if not self.preview.busy:
            self._step_frame(1, playback=True)

    def _refresh_steps(self, selected=None):
        selected = self.step_list.currentRow() if selected is None else selected
        self.step_list.blockSignals(True)
        self.step_list.clear()
        for step in self.steps:
            spec = self.algorithm_specs[step.algorithm]
            item = QListWidgetItem(spec.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
            self.step_list.addItem(item)
        self.step_list.blockSignals(False)
        self.step_list.setCurrentRow(max(0, min(selected, len(self.steps) - 1)))
        self._build_parameters(self.step_list.currentRow())

    def _add_step(self):
        spec = self.algorithm_specs[self.algorithm_combo.currentData()]
        self.steps.append(StepSpec(spec.id, spec.defaults(), manifest=spec.manifest, code_hash=spec.code_hash, version=spec.version))
        self._refresh_steps(len(self.steps) - 1)
        self._config_changed()

    def _remove_step(self):
        index = self.step_list.currentRow()
        if index >= 0:
            self.steps.pop(index)
            self._refresh_steps(index)
            self._config_changed()

    def _move_step(self, direction):
        index = self.step_list.currentRow()
        target = index + direction
        if 0 <= index < len(self.steps) and 0 <= target < len(self.steps):
            self.steps[index], self.steps[target] = self.steps[target], self.steps[index]
            self._refresh_steps(target)
            self._config_changed()

    def _step_checked(self, item):
        index = self.step_list.row(item)
        self.steps[index] = replace(self.steps[index], enabled=item.checkState() == Qt.CheckState.Checked)
        self._config_changed()

    def _build_parameters(self, index):
        while self.parameter_form.rowCount():
            self.parameter_form.removeRow(0)
        if index < 0 or index >= len(self.steps):
            self.algorithm_description.setText("空处理链会复制输入数据。")
            return
        step = self.steps[index]
        spec = self.algorithm_specs[step.algorithm]
        self.algorithm_description.setText(spec.description)
        for parameter in spec.parameters:
            value = step.parameters.get(parameter.key, parameter.default)
            if parameter.kind in {"int", "float"}:
                control = QSpinBox() if parameter.kind == "int" else QDoubleSpinBox()
                lower, upper = parameter.minimum if parameter.minimum is not None else -1e9, parameter.maximum if parameter.maximum is not None else 1e9
                control.setRange(int(lower) if parameter.kind == "int" else lower, int(upper) if parameter.kind == "int" else upper)
                if parameter.kind == "float":
                    control.setDecimals(6)
                control.setValue(value)
                control.valueChanged.connect(lambda new, key=parameter.key, row=index: self._parameter_changed(row, key, new))
            elif parameter.kind == "bool":
                control = QCheckBox()
                control.setChecked(value)
                control.toggled.connect(lambda new, key=parameter.key, row=index: self._parameter_changed(row, key, new))
            else:
                control = QLineEdit(value)
                control.textChanged.connect(lambda new, key=parameter.key, row=index: self._parameter_changed(row, key, new))
            self.parameter_form.addRow(parameter.label + (f" ({parameter.unit})" if parameter.unit else ""), control)

    def _parameter_changed(self, index, key, value):
        if 0 <= index < len(self.steps):
            parameters = dict(self.steps[index].parameters)
            parameters[key] = value
            self.steps[index] = replace(self.steps[index], parameters=parameters)
            self._config_changed()

    def _register_plugin(self):
        path, _ = QFileDialog.getOpenFileName(self, "注册本地算法描述", "", "算法描述 (*.json)")
        if not path:
            return
        try:
            spec = load_manifest(path)
            if spec.id in {value.id for value in BUILTINS}:
                raise ValueError("外部插件不能替换内置算法标识")
            self.algorithm_specs[spec.id] = spec
            existing = self.algorithm_combo.findData(spec.id)
            if existing < 0:
                self.algorithm_combo.addItem(spec.name, spec.id)
                existing = self.algorithm_combo.count() - 1
            self.algorithm_combo.setCurrentIndex(existing)
            # Refresh hashes of already configured instances after explicit registration.
            self.steps = [replace(step, code_hash=spec.code_hash, version=spec.version) if step.algorithm == spec.id else step for step in self.steps]
            self._refresh_steps()
            self._config_changed()
            self.statusBar().showMessage(f"已注册 {spec.name}；Python 插件仅在计算进程执行。")
        except Exception as error:
            self._error(error)

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _selection(self, trial=False):
        if trial or self.range_mode.currentIndex() == 0:
            return Selection("current", current=self.frame.ref, roi=self.roi)
        if self.range_mode.currentIndex() == 1:
            offset = 0 if self.meta["kind"] == "video" else 1
            return Selection("range", self.range_start.value() - offset, self.range_end.value() - offset, roi=self.roi)
        return Selection("all", roi=self.roi)

    def _job_payload(self, trial=False):
        if not self.ready or self.frame is None:
            raise ValueError("请先打开并定位有效数据帧")
        selection = self._selection(trial)
        sources = [self.sources[self.source.key]]
        if not trial and self.batch.isChecked():
            keys = []
            for item in self.tree.selectedItems():
                value = item.data(0, Qt.ItemDataRole.UserRole)
                if value and not value.get("result"):
                    key = SourceSpec(**value["source"]).key
                    if key not in keys:
                        keys.append(key)
            if not keys:
                raise ValueError("请在左侧选中至少一个输入数据源")
            sources = [self.sources[key] for key in keys]
            if selection.mode == "current":
                raise ValueError("批处理请使用全部数据或指定范围")
            if selection.mode == "range" and len({entry["meta"]["kind"] == "video" for entry in sources}) != 1:
                raise ValueError("视频时间范围和序列帧范围不能混用于同一次批处理")
        return {"id": uuid.uuid4().hex[:12], "mode": "trial" if trial else "run", "sources": [asdict(entry["source"]) for entry in sources], "revisions": [entry["meta"]["revision"] for entry in sources], "steps": [asdict(step) for step in self.steps], "selection": asdict(selection), "output_dir": str(Path(self.output_edit.text()).resolve()), "seed": 0, "config_revision": self.config_revision}

    def _trial(self):
        try:
            if self.jobs.active and self.jobs.active["mode"] == "trial":
                self.jobs.cancel()
            for pending in list(self.jobs.pending):
                if pending["mode"] == "trial":
                    self.jobs.cancel(pending["id"])
            self.jobs.submit(self._job_payload(trial=True))
        except Exception as error:
            self._error(error)

    def _run(self):
        try:
            self.jobs.submit(self._job_payload())
        except Exception as error:
            self._error(error)

    def _cancel_job(self):
        row = self.task_table.currentRow()
        identifier = next((key for key, value in self.job_rows.items() if value == row), None)
        self.jobs.cancel(identifier)

    def _job_event(self, event):
        identifier = event["id"]
        kind = event["type"]
        if kind == "queued":
            payload = event["payload"]
            self.job_specs[identifier] = payload
            row = self.task_table.rowCount()
            self.task_table.insertRow(row)
            self.job_rows[identifier] = row
            texts = [f"{'试算' if payload['mode'] == 'trial' else '处理'} · {identifier[:8]}", Path(payload["sources"][0]["path"]).name + (f" 等 {len(payload['sources'])} 个" if len(payload['sources']) > 1 else ""), "等待", "—"]
            for column, text in enumerate(texts):
                self.task_table.setItem(row, column, QTableWidgetItem(text))
            return
        row = self.job_rows.get(identifier)
        if row is None:
            return
        states = {"running": "运行中", "cancelling": "取消中", "cancelled": "已取消", "failed": "失败", "trial": "试算完成", "completed": "完成"}
        if kind in states:
            self.task_table.item(row, 2).setText(states[kind])
        if kind == "progress":
            text = f"{event['completed']} / {event['total']} 帧" if event["total"] else f"已处理 {event['completed']} 帧"
            self.task_table.item(row, 3).setText(f"数据 {event['source'] + 1} · {text}")
        elif kind == "log":
            self.log_view.append(f"[{identifier[:8]}] {event['message']}")
        elif kind == "failed":
            self.log_view.append(event.get("details", event["message"]))
            self.task_table.item(row, 2).setToolTip(event["message"])
            self.statusBar().showMessage(f"任务失败：{event['message']}")
        elif kind == "trial":
            if event["revision"] != self.config_revision:
                self.task_table.item(row, 2).setText("完成（配置已变化）")
                return
            self.result_pixels = event["pixels"]
            self.result_kind = event["kind"]
            self.result_origin = tuple(event["origin"])
            self.result_snapshot = self.job_specs[identifier]
            self.result_fresh = True
            self.export_button.setEnabled(True)
            ref = event["frame"]
            position = f"{ref['time']:.6f} s" if ref["time"] is not None else f"帧 {ref['index'] + 1}"
            self.result_header.setText(f"精确试算 · {position} · {event['kind']} · {self.result_pixels.dtype}")
            self.task_table.item(row, 3).setText("1 / 1 帧")
            self._render()
        elif kind == "completed":
            run = event["run"]
            if run["status"] != "succeeded":
                self.task_table.item(row, 2).setText("部分失败" if run["status"] == "completed_with_errors" else "失败")
            self.task_table.item(row, 3).setText(f"{sum(item['status'] == 'succeeded' for item in run['results'])} / {len(run['results'])} 个成功")
            self.task_table.item(row, 2).setToolTip(event["directory"])
            for item in run["results"]:
                if item["status"] != "succeeded":
                    continue
                output_path = Path(event["directory"]) / item["output_directory"]
                result_source = SourceSpec(str(output_path), "sequence")
                node = QTreeWidgetItem(self.result_root, [f"{Path(item['source']['path']).name} · {identifier[:8]}"])
                node.setData(0, Qt.ItemDataRole.UserRole, {"source": asdict(result_source), "result": True, "input_key": SourceSpec(**item["source"]).key, "origin": item["origin"], "result_kind": item["result_kind"]})
            self.statusBar().showMessage(f"处理结束：{event['directory']}")

    def _export_trial(self):
        if not self.result_fresh:
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存精确试算结果", str(Path(self.output_edit.text()) / "trial.tif"), "TIFF 图像 (*.tif)")
        if not path:
            return
        try:
            target = Path(path).resolve()
            if any(target == Path(entry["source"].path).resolve() or (Path(entry["source"].path).is_dir() and target.is_relative_to(Path(entry["source"].path).resolve())) for entry in self.sources.values()):
                raise ValueError("输出不能覆盖已打开的原始数据")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            tifffile.imwrite(temporary, self.result_pixels, photometric="rgb" if self.result_pixels.ndim == 3 else "minisblack", metadata={"origin_xy": self.result_origin})
            temporary.replace(target)
            write_json(target.with_suffix(target.suffix + ".json"), {"schema_version": 1, "kind": "exact_trial", "spec": self.result_snapshot, "origin_xy": self.result_origin, "dtype": str(self.result_pixels.dtype)})
            self.statusBar().showMessage(f"已保存结果及参数记录：{target}")
        except Exception as error:
            self._error(error)

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存项目", str(self.project_path or Path.cwd() / "project.eip.json"), "项目文件 (*.eip.json);;JSON (*.json)")
        if not path:
            return
        try:
            value = {"schema_version": 1, "sources": [asdict(entry["source"]) for entry in self.sources.values()], "steps": [asdict(step) for step in self.steps], "active_source": asdict(self.source) if self.source else None, "roi": asdict(self.roi) if self.roi else None, "display": asdict(self._display_settings()), "range_mode": self.range_mode.currentIndex(), "range_start": self.range_start.value(), "range_end": self.range_end.value(), "output_dir": self.output_edit.text(), "results": [self.result_root.child(i).data(0, Qt.ItemDataRole.UserRole) for i in range(self.result_root.childCount())]}
            write_json(Path(path), value)
            self.project_path = Path(path)
            self.statusBar().showMessage(f"项目已保存：{path}")
        except Exception as error:
            self._error(error)

    def _open_project(self):
        if self.jobs.active or self.jobs.pending:
            self._error("任务运行期间暂不切换项目；可等待或取消任务后再打开")
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开项目", "", "项目文件 (*.eip.json *.json)")
        if not path:
            return
        try:
            value = read_project(Path(path))
            steps = [StepSpec(**item) for item in value["steps"]]
            for step in steps:
                if step.manifest:
                    spec = load_manifest(step.manifest)
                    if spec.code_hash != step.code_hash:
                        raise ValueError("项目引用的插件已变化，请先重新注册或恢复原插件")
                    self.algorithm_specs[spec.id] = spec
                    if self.algorithm_combo.findData(spec.id) < 0:
                        self.algorithm_combo.addItem(spec.name, spec.id)
                elif step.algorithm not in self.algorithm_specs:
                    raise ValueError("项目包含未知算法")
                validate_parameters(self.algorithm_specs[step.algorithm], step.parameters)
            sources = [SourceSpec(**item) for item in value["sources"]]
            if len(sources) > 32:
                raise ValueError("当前项目加载器一次支持最多 32 个来源")
            missing = [source.path for source in sources if not Path(source.path).exists()]
            if missing:
                raise ValueError("项目来源缺失，请恢复文件位置或修正项目路径：\n" + "\n".join(missing))
            self._pause()
            self.source = self.meta = self.frame = None
            self.ready = False
            self.result_pixels = None
            self.input_canvas.clear_image()
            self.result_canvas.clear_image()
            self.sources.clear()
            self.items.clear()
            self.input_root.takeChildren()
            self.result_root.takeChildren()
            self.steps = steps
            self._refresh_steps()
            self.output_edit.setText(value.get("output_dir", str(Path.cwd() / "outputs")))
            for result in value.get("results", []):
                node = QTreeWidgetItem(self.result_root, [Path(result["source"]["path"]).name])
                node.setData(0, Qt.ItemDataRole.UserRole, result)
            active = value.get("active_source")
            sources.sort(key=lambda source: asdict(source) == active)
            for source in sources:
                self._request("open", source)
            self.project_path = Path(path)
            self._pending_project_view = value
        except Exception as error:
            self._error(error)

    def closeEvent(self, event):
        if self.jobs.active or self.jobs.pending:
            answer = QMessageBox.question(self, "关闭工作台", "仍有后台任务。关闭将停止任务，未完成输出保留为 .partial。继续关闭？", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self._pause()
        self.jobs.close()
        self.preview.close()
        event.accept()
