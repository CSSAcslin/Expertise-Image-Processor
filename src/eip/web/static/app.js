import { api, session, download } from "./api.js";
import { Viewer } from "./viewer.js";
import { FileBrowser } from "./file-browser.js";
import { validROI, moveROI } from "./geometry.js";

const $ = (id) => document.getElementById(id);
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const icons = {
  plus: "M12 5v14M5 12h14",
  close: "m6 6 12 12M6 18 18 6",
  folder: "M3 7V5h7l2 2h9v13H3Z",
  image: "M3 3h18v18H3ZM3 17l6-7 5 5 3-3 4 5M15 7h.01",
  play: "m8 5 11 7-11 7Z",
  pause: "M8 5v14M16 5v14",
  previous: "M5 5v14m14-14L8 12l11 7Z",
  next: "M19 5v14M5 5l11 7-11 7Z",
  hand: "M8 12V5a2 2 0 0 1 4 0v6-7a2 2 0 0 1 4 0v8-6a2 2 0 0 1 4 0v10c0 5-3 7-7 7-3 0-4-2-6-4l-4-5a2 2 0 0 1 3-2l2 2",
  crop: "M6 3v15h15M3 6h15v15",
  layers: "m12 3 10 6-10 6L2 9Zm-10 12 10 6 10-6M2 15l10 6 10-6",
  sliders: "M4 5h16M4 12h16M4 19h16M8 3v4M16 10v4M10 17v4",
  spark: "m12 2 3 7 7 3-7 3-3 7-3-7-7-3 7-3Z",
  activity: "M2 12h5l3-8 4 16 3-8h5",
  chevron: "m6 9 6 6 6-6",
  up: "m5 12 7-7 7 7M12 5v15",
  down: "m5 12 7 7 7-7M12 5v15",
  video: "M3 5h13v14H3Zm13 5 5-4v12l-5-4",
  file: "M5 3h9l5 5v13H5ZM14 3v6h5",
};
function icon(name) {
  return `<svg class="svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${icons[name] || icons.file}"/></svg>`;
}
function wireIcons(root = document) {
  root
    .querySelectorAll("[data-icon]")
    .forEach((el) => (el.innerHTML = icon(el.dataset.icon)));
}
wireIcons();
let toastTimer;
function toast(message, error = false) {
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(
    () => ($("toast").hidden = true),
    error ? 7000 : 3500,
  );
}
function safe(fn) {
  return async (...args) => {
    try {
      await fn(...args);
    } catch (error) {
      stopPlay();
      toast(error.message, true);
    }
  };
}

let state,
  sources = [],
  algorithms = [],
  steps = [],
  active = null,
  input = null,
  result = null,
  formalResult = null,
  resultSignature = null,
  displayRevision = 0,
  sourceGeneration = 0,
  display = { low: null, high: null, gamma: 1, colormap: "gray" },
  resultDisplay = { low: null, high: null, gamma: 1, colormap: "gray" },
  playbackGeneration = 0,
  requestBusy = false,
  pendingSeek = null,
  playing = false,
  fetchedJobs = new Set(),
  jobs = [];
let sourceViews = new Map(),
  roiUndo = [],
  roiRedo = [],
  committedROI = null,
  pixelGeneration = 0;
let previewChain = Promise.resolve();
let selectionRequest = 0;
function frameRequest(id, value, generation = sourceGeneration) {
  const task = previewChain
    .catch(() => {})
    .then(() =>
      generation === sourceGeneration
        ? api(`sources/${id}/frame`, value)
        : null,
    );
  previewChain = task;
  return task;
}
const scene = {
  width: 640,
  height: 480,
  zoom: 1,
  panX: 0,
  panY: 0,
  tool: "pan",
  roi: null,
};
let pixelTimer;
const callbacks = {
  redraw,
  fit,
  roi: () => {
    rememberROI();
    changed();
    updateROI();
  },
  roiPreview: () => {
    updateROI();
    $("stale").hidden = !result || resultSignature === signature();
  },
  pixel: (snapshot, x, y) => {
    const generation = ++pixelGeneration;
    clearTimeout(pixelTimer);
    $("pixel-info").textContent = `x ${x} · y ${y}`;
    const origin = snapshot.origin;
    if (
      x < origin[0] ||
      y < origin[1] ||
      x >= origin[0] + snapshot.shape[1] ||
      y >= origin[1] + snapshot.shape[0]
    )
      return;
    pixelTimer = setTimeout(async () => {
      try {
        const data = await api(`frames/${snapshot.token}/pixel?x=${x}&y=${y}`);
        if (generation !== pixelGeneration) return;
        $("pixel-info").textContent =
          `x ${x} · y ${y}  |  ${Array.isArray(data.value) ? data.value.join(", ") : data.value}  ${data.dtype}`;
      } catch {
        if (generation === pixelGeneration)
          $("pixel-info").textContent =
            `x ${x} · y ${y} | 原值缓存已过期，重新定位可恢复`;
      }
    }, 130);
  },
};
const originalViewer = new Viewer($("input-canvas"), scene, callbacks),
  resultViewer = new Viewer($("result-canvas"), scene, callbacks);
function redraw() {
  originalViewer.draw();
  resultViewer.draw();
  $("zoom-value").textContent =
    `${Math.round(visibleViewer().scale() * 100) || 100}%`;
}
function visibleViewer() {
  return $("viewers").dataset.mode === "result" ? resultViewer : originalViewer;
}
function fit() {
  scene.zoom = 1;
  scene.panX = scene.panY = 0;
  redraw();
}
function signature() {
  return JSON.stringify({
    active: active?.id,
    ref: input?.ref,
    steps,
    roi: scene.roi,
  });
}
function changed() {
  $("stale").hidden = !result || resultSignature === signature();
  updateSummary();
  updateROIStatistics();
  scheduleDraft();
}
function updateROI() {
  const q = scene.roi;
  $("roi-info").textContent = q
    ? `x ${q.x} · y ${q.y}  /  ${q.width} × ${q.height} px`
    : "全幅图像";
  for (const key of ["x", "y", "width", "height"])
    $("roi-" + key).value = q?.[key] ?? "";
  $("roi-undo").disabled = !roiUndo.length;
  $("roi-redo").disabled = !roiRedo.length;
  $("roi-editor").hidden = !q;
  $("clear-roi").disabled = !q;
}
let roiStatisticsKey = "",
  roiStatisticsTimer;
function updateROIStatistics() {
  if (!input) return;
  const token = input.token,
    roi = scene.roi,
    key = JSON.stringify({ token, roi });
  if (key === roiStatisticsKey) return;
  roiStatisticsKey = key;
  clearTimeout(roiStatisticsTimer);
  const show = (stats) => {
    if (key !== roiStatisticsKey) return;
    const f = (v) => (v === null ? "无有限值" : Number(v).toPrecision(5));
    $("roi-statistics").textContent =
      `原值均值 ${f(stats.mean)} · 范围 ${f(stats.min)} ～ ${f(stats.max)}${stats.invalid_count ? " · 非有限样本 " + stats.invalid_count : ""}`;
  };
  if (!roi) {
    show(input.statistics);
    return;
  }
  $("roi-statistics").textContent = "正在读取选区原值统计…";
  roiStatisticsTimer = setTimeout(async () => {
    try {
      const data = await api(
        `frames/${token}/region?x=${roi.x}&y=${roi.y}&width=${roi.width}&height=${roi.height}`,
      );
      show(data.statistics);
    } catch {
      if (key === roiStatisticsKey)
        $("roi-statistics").textContent = "原值缓存已过期，重新定位可恢复";
    }
  }, 200);
}
function rememberROI() {
  if (JSON.stringify(scene.roi) !== JSON.stringify(committedROI)) {
    roiUndo.push(structuredClone(committedROI));
    roiUndo = roiUndo.slice(-32);
    roiRedo = [];
    committedROI = structuredClone(scene.roi);
  }
}
function commitROI(roi) {
  if (roi && !validROI(roi, scene.width, scene.height))
    throw Error("ROI 必须是图像范围内的整数像素矩形");
  scene.roi = roi;
  rememberROI();
  updateROI();
  changed();
  redraw();
}
function stashView() {
  if (active && input)
    sourceViews.set(active.id, {
      roi: structuredClone(scene.roi),
      display: { ...display },
      ref: input.ref,
      zoom: scene.zoom,
      panX: scene.panX,
      panY: scene.panY,
    });
}
function updateSummary() {
  const count = steps.filter((s) => s.enabled).length;
  $("run-summary").textContent = active
    ? `${$("batch").checked ? sources.filter((s) => s.role === "input").length : 1} 个来源 · ${count} 个处理步骤${scene.roi ? " · ROI" : ""}`
    : "打开数据以开始处理";
  $("trial").disabled = $("run").disabled = !active || !input;
  for (const id of ["previous", "next", "play", "seek", "seek-value"])
    $(id).disabled = !active || !input;
  if (active && input && active.meta.kind !== "video") {
    $("play").disabled = (active.meta.frame_count || 1) <= 1;
    $("previous").disabled = (input.ref.index ?? 0) <= 0;
    $("next").disabled =
      (input.ref.index ?? 0) >= (active.meta.frame_count || 1) - 1;
  }
  $("scrubber").disabled =
    !active ||
    !input ||
    (active.meta.kind === "video" && !active.meta.duration);
  $("promote-result").disabled = !formalResult;
  $("flow-summary").textContent = active
    ? `${active.lineage?.length ? "阶段 " + (active.lineage.length + 1) + " · " : ""}${active.display_name || active.meta.name} → ${count} 个处理步骤 → ${formalResult ? "已保存输出" : result ? "当前帧试算" : "待处理"}`
    : "导入来源 → 配置算法 → 检查结果";
}
function updateSources() {
  const inputs = sources.filter((s) => s.role === "input"),
    results = sources.filter((s) => s.role === "result");
  $("source-count").textContent = inputs.length;
  $("result-count").textContent = results.length;
  const card = (s) =>
    `<div class="source-card ${s.id === active?.id || s.id === formalResult?.id ? "selected" : ""}" tabindex="0" role="button" data-source="${s.id}" aria-label="预览 ${escape(s.display_name || s.meta.name)}"><img src="${s.thumbnail}" alt=""><div class="source-copy"><strong title="${escape(s.source.path)}">${escape(s.display_name || s.meta.name)}</strong><span class="source-kind">${s.role === "result" ? "处理结果" : s.lineage?.length ? "派生输入 · 阶段 " + (s.lineage.length + 1) : s.meta.kind === "video" ? "视频" : s.meta.kind === "sequence" ? "图像序列" : "图像"}</span><small>${s.meta.shape[1]} × ${s.meta.shape[0]} · ${escape(s.meta.dtype)}</small><button class="text-button source-info" data-info="${s.id}">属性 / 来源</button></div><button class="icon-button source-remove" data-remove="${s.id}" aria-label="移除 ${escape(s.display_name || s.meta.name)}">${icon("close")}</button></div>`;
  $("source-list").innerHTML = inputs.length
    ? inputs.map(card).join("")
    : '<div class="library-empty">导入图像、视频<br>或图像序列目录</div>';
  $("result-list").innerHTML = results.map(card).join("");
  document.querySelectorAll("[data-source]").forEach((el) => {
    el.onclick = safe(() => selectSource(el.dataset.source));
    el.onkeydown = (e) => {
      if (e.target === el && e.key === "Enter") el.click();
    };
  });
  document.querySelectorAll("[data-info]").forEach(
    (b) =>
      (b.onclick = (event) => {
        event.stopPropagation();
        showSourceDetails(b.dataset.info);
      }),
  );
  document.querySelectorAll("[data-remove]").forEach(
    (b) =>
      (b.onclick = (event) => {
        event.stopPropagation();
        safe(() => removeSource(b.dataset.remove))();
      }),
  );
  updateSummary();
}
function showSourceDetails(id) {
  const entry = sources.find((s) => s.id === id);
  if (!entry) return;
  const labels = {
    stream_pts: "视频真实时间戳",
    user: "用户指定间隔",
    run_record: "上游运行时间记录",
    unavailable: "未知 / 未指定",
  };
  const rows = [
    ["来源", entry.display_name || entry.meta.name],
    ["路径", entry.source.path],
    ["类型", entry.meta.kind],
    ["数据尺寸", entry.meta.shape.join(" × ")],
    ["像素类型", entry.meta.dtype],
    ["轴 / 颜色", entry.meta.axes + " / " + entry.meta.color],
    ["帧数", entry.meta.frame_count ?? "未知"],
    [
      "时长",
      entry.meta.duration === null ? "未知" : entry.meta.duration + " s",
    ],
    [
      "时间来源",
      labels[entry.meta.timestamp_source] || entry.meta.timestamp_source,
    ],
  ];
  $("source-detail").innerHTML =
    `<table class="metadata-table">${rows.map(([k, v]) => `<tr><td>${escape(k)}</td><td>${escape(v)}</td></tr>`).join("")}</table><p class="small-copy">${escape((entry.meta.notes || []).join("；"))}</p>${(entry.lineage || []).map((p, i) => `<div class="lineage-item"><strong>上游阶段 ${i + 1} · ${escape(p.run_id)}</strong><br>裁剪原点：${p.crop_origin.join(", ")} px<br>来源：${escape(p.parent_source.path)}<br>运行记录：${escape(p.manifest)}</div>`).join("")}`;
  $("source-dialog").showModal();
}
async function removeSource(id) {
  await api(`sources/${id}/remove`, {});
  sources = sources.filter((s) => s.id !== id);
  sourceViews.delete(id);
  if (formalResult?.id === id) {
    formalResult = result = null;
    resultViewer.clear();
    setMode("input");
  }
  if (active?.id === id) {
    stopPlay();
    active = input = result = formalResult = null;
    sourceGeneration++;
    originalViewer.clear();
    resultViewer.clear();
    scene.roi = null;
    roiUndo = roiRedo = [];
    committedROI = null;
    updateROI();
    const next = sources.find((s) => s.role === "input");
    if (next) await selectSource(next.id);
    else {
      $("welcome").hidden = false;
      $("preview-title").textContent = "数据预览";
      $("frame-info").textContent = "尚未打开数据";
    }
  }
  updateSources();
  scheduleDraft();
  toast("已移除来源引用，原始文件和输出保持完整");
}
async function promoteResult() {
  if (!formalResult) return;
  const entry = formalResult;
  const opened = await api(`sources/${entry.id}/promote`, {});
  sources = sources.filter((s) => s.id !== opened.source.id);
  sources.push(opened.source);
  steps = [];
  renderSteps();
  $("batch").checked = false;
  $("scope").value = "all";
  $("range-fields").hidden = true;
  resultDisplay = { low: null, high: null, gamma: 1, colormap: "gray" };
  await selectSource(opened.source.id);
  updateSources();
  toast("已创建下一阶段输入，处理链已清空；请添加新的算法步骤");
}
function setMode(mode) {
  $("viewers").dataset.mode = mode;
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.classList.toggle("selected", b.dataset.view === mode));
  requestAnimationFrame(redraw);
}
function drawHistogram(snapshot) {
  const c = $("histogram"),
    ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const max = Math.max(...snapshot.histogram, 1);
  ctx.fillStyle = "#5d9e91";
  snapshot.histogram.forEach((n, i) => {
    const h = (n / max) * 38;
    ctx.fillRect(i * 2.5, c.height - h, 2, h);
  });
  const bounds = snapshot.histogram_range || [
    snapshot.statistics.min,
    snapshot.statistics.max,
  ];
  $("histogram-min").textContent = Number(bounds[0]).toPrecision(4);
  $("histogram-max").textContent = Number(bounds[1]).toPrecision(4);
  $("histogram").title =
    `有限样本直方图，${snapshot.histogram_samples ?? "采样"} 个样本`;
}
async function setInput(snapshot, reset = false) {
  pixelGeneration++;
  input = snapshot;
  scene.width = snapshot.base_shape[1];
  scene.height = snapshot.base_shape[0];
  if (reset) {
    scene.roi = null;
    fit();
    updateROI();
  }
  await originalViewer.set(snapshot);
  if (input !== snapshot) return;
  $("welcome").hidden = true;
  $("preview-title").textContent = active.display_name || active.meta.name;
  $("dtype-badge").textContent = snapshot.dtype;
  $("shape-badge").textContent = `${snapshot.shape[1]} × ${snapshot.shape[0]}`;
  const ref = snapshot.ref;
  $("frame-info").textContent =
    `${ref.index === null ? "帧号未知" : `第 ${ref.index + 1} 帧`}  |  ${ref.time === null ? "时间未知" : `${ref.time.toFixed(4)} s`}${snapshot.downsampled ? "  |  显示降采样" : ""}`;
  $("seek-value").value =
    active.meta.kind === "video"
      ? (ref.time ?? 0).toFixed(4)
      : (ref.index ?? 0) + 1;
  $("scrubber").value =
    active.meta.kind === "video" ? (ref.time ?? 0) : (ref.index ?? 0);
  fillDisplayControls(
    $("display-target").value === "input" ? snapshot : result,
    $("display-target").value === "input" ? display : resultDisplay,
  );
  changed();
}
function configureTimeline(entry) {
  const video = entry.meta.kind === "video";
  $("scrubber").max = video
    ? entry.meta.duration || 1
    : Math.max(0, (entry.meta.frame_count || 1) - 1);
  $("scrubber").step = video ? "0.001" : "1";
  $("scrubber").disabled = video && !entry.meta.duration;
  $("seek-value").step = video ? "0.001" : "1";
  $("seek-value").min = video ? "0" : "1";
  $("seek-value").max = video
    ? (entry.meta.duration ?? "")
    : (entry.meta.frame_count ?? "");
  $("seek-unit").textContent = video ? "s" : "帧";
  $("time-start").textContent = video ? "0 s" : "1";
  $("time-end").textContent = video
    ? entry.meta.duration
      ? `${entry.meta.duration.toFixed(2)} s`
      : "时长未知"
    : `${entry.meta.frame_count || "?"} 帧`;
  $("timeline-note").textContent = video
    ? "真实时间戳 · 帧号仅在可确定时显示"
    : entry.meta.timestamp_source === "unavailable"
      ? "序列时间未知"
      : entry.meta.timestamp_source === "run_record"
        ? "时间沿用上游记录"
        : "用户指定序列间隔";
  $("range-unit").textContent = video
    ? "单位：秒；终点不包含"
    : "单位：从 1 开始的帧号；终点不包含";
  $("range-start").min = video ? 0 : 1;
  $("range-start").step = $("range-end").step = video ? "any" : "1";
  $("range-start").value = video ? 0 : 1;
  $("range-end").value = video
    ? entry.meta.duration || 1
    : (entry.meta.frame_count || 1) + 1;
}
async function selectSource(id, request = ++selectionRequest) {
  stopPlay();
  pendingSeek = null;
  const entry = sources.find((s) => s.id === id);
  if (!entry) return;
  if (entry.role === "result") {
    const matching = sources.find((s) => s.id === entry.input_id);
    if (matching && (active?.id !== matching.id || !input))
      await selectSource(matching.id, request);
    if (request !== selectionRequest) return;
    formalResult = entry;
    await syncFormalResult();
    if (request !== selectionRequest) return;
    updateSources();
    setMode("compare");
    toast("结果按来源帧/时间同步显示");
    return;
  }
  stashView();
  active = entry;
  const view = sourceViews.get(id);
  display = view?.display || {
    low: null,
    high: null,
    gamma: 1,
    colormap: "gray",
  };
  const generation = ++sourceGeneration;
  input = null;
  originalViewer.clear();
  $("frame-info").textContent = "正在打开数据…";
  result = formalResult = null;
  resultSignature = null;
  resultViewer.clear();
  $("result-meta").textContent = "";
  $("result-label").textContent = "处理结果";
  $("stale").hidden = true;
  configureTimeline(entry);
  updateSources();
  setMode("input");
  const snapshot = await frameRequest(id, {
    action: "open",
    display,
  });
  if (
    active?.id !== id ||
    generation !== sourceGeneration ||
    request !== selectionRequest
  )
    return;
  await setInput(snapshot, true);
  if (generation !== sourceGeneration || request !== selectionRequest) return;
  roiUndo = roiRedo = [];
  committedROI = null;
  if (view?.roi && validROI(view.roi, scene.width, scene.height)) {
    scene.roi = structuredClone(view.roi);
    committedROI = structuredClone(view.roi);
  }
  if (view?.zoom) {
    scene.zoom = view.zoom;
    scene.panX = view.panX;
    scene.panY = view.panY;
  }
  if (view?.ref && (view.ref.index !== 0 || view.ref.time > 0))
    await seekFrame(
      entry.meta.kind === "video"
        ? { action: "time", time: view.ref.time ?? 0 }
        : { action: "index", index: view.ref.index ?? 0 },
    );
  updateROI();
  redraw();
  changed();
  if (entry.meta.notes?.length) toast(entry.meta.notes.join("；"));
}
async function seekFrame(value) {
  if (!active || !input) return;
  if (requestBusy) {
    pendingSeek = value;
    return;
  }
  requestBusy = true;
  const id = active.id,
    generation = sourceGeneration;
  $("timeline-note").textContent = "正在解码…";
  try {
    const snapshot = await frameRequest(id, { ...value, display }, generation);
    if (active?.id !== id || generation !== sourceGeneration) return;
    if (snapshot.eof) {
      stopPlay();
      return;
    }
    await setInput(snapshot);
    if (generation !== sourceGeneration) return;
    if (formalResult) await syncFormalResult();
  } finally {
    requestBusy = false;
    $("timeline-note").textContent =
      active?.meta.kind === "video"
        ? "真实时间戳 · 帧号仅在可确定时显示"
        : "按帧索引定位";
    if (pendingSeek) {
      const next = pendingSeek;
      pendingSeek = null;
      await seekFrame(next);
    }
  }
}
async function syncFormalResult() {
  if (!formalResult || !input) return;
  const entry = formalResult,
    snapshot = await frameRequest(entry.id, {
      action: "match",
      reference: input.ref,
      display: resultDisplay,
    });
  if (formalResult !== entry) return;
  $("stale").hidden = true;
  resultSignature = null;
  if (snapshot.unavailable) {
    result = null;
    resultViewer.clear();
    $("result-meta").textContent = "";
    if ($("display-target").value === "result")
      fillDisplayControls(null, resultDisplay);
    $("result-label").textContent = "该输入帧没有对应输出";
  } else {
    result = snapshot;
    await paintResult(snapshot);
    $("result-label").textContent =
      `正式结果 · 输出第 ${snapshot.ref.index + 1} 帧`;
  }
}
function stopPlay() {
  playing = false;
  playbackGeneration++;
  $("play").innerHTML = icon("play");
  $("play").setAttribute("aria-label", "播放");
}
function step(direction) {
  if (!active || !input) return;
  if (active.meta.kind !== "video") {
    const next = (input.ref.index ?? 0) + direction;
    if (next < 0 || next >= (active.meta.frame_count || 1)) {
      if (direction > 0) stopPlay();
      return;
    }
  }
  return seekFrame(
    active.meta.kind === "video"
      ? {
          action: direction > 0 ? "next" : "previous",
          time: input.ref.time ?? 0,
        }
      : {
          action: "index",
          index: Math.max(
            0,
            Math.min(
              (active.meta.frame_count || 1) - 1,
              (input.ref.index ?? 0) + direction,
            ),
          ),
        },
  );
}
async function play() {
  if (playing) {
    stopPlay();
    return;
  }
  if (!active || !input) return;
  playing = true;
  const generation = ++playbackGeneration;
  $("play").innerHTML = icon("pause");
  $("play").setAttribute("aria-label", "暂停");
  while (playing && generation === playbackGeneration) {
    const before = input.token;
    await step(1);
    if (generation !== playbackGeneration) break;
    if (input.token === before) {
      stopPlay();
      break;
    }
    await new Promise((resolve) =>
      setTimeout(resolve, Math.max(25, 1000 / (active.meta.rate || 10))),
    );
  }
}
function updateAlgorithms() {
  const selected = $("algorithm").value;
  $("algorithm").innerHTML = algorithms
    .map((a) => `<option value="${escape(a.id)}">${escape(a.name)}</option>`)
    .join("");
  if (algorithms.some((a) => a.id === selected))
    $("algorithm").value = selected;
}
function renderSteps() {
  $("step-count").textContent = steps.length;
  $("steps").innerHTML = steps.length
    ? steps
        .map((s, i) => {
          const spec = algorithms.find((a) => a.id === s.algorithm);
          return `<div class="step-card ${s.enabled ? "" : "disabled"}"><div class="step-header"><span class="step-number">${String(i + 1).padStart(2, "0")}</span><span class="step-name">${escape(spec?.name || s.algorithm)}</span><input type="checkbox" data-enable="${i}" ${s.enabled ? "checked" : ""} aria-label="启用 ${escape(spec?.name || s.algorithm)}"><button class="icon-button" data-up="${i}" title="上移" aria-label="上移步骤">${icon("up")}</button><button class="icon-button" data-down="${i}" title="下移" aria-label="下移步骤">${icon("down")}</button><button class="icon-button" data-delete="${i}" title="删除" aria-label="删除步骤">${icon("close")}</button></div><p class="step-description">${escape(spec?.description)}</p><div class="step-parameters">${(spec?.parameters || []).map((p) => `<label>${escape(p.label)}${p.unit ? " · " + escape(p.unit) : ""}<input data-param="${i}" data-key="${escape(p.key)}" data-kind="${p.kind}" type="${p.kind === "bool" ? "checkbox" : p.kind === "str" ? "text" : "number"}" ${p.kind === "bool" ? (s.parameters[p.key] ? "checked" : "") : `value="${escape(s.parameters[p.key])}"`} ${p.minimum === null ? "" : `min="${p.minimum}"`} ${p.maximum === null ? "" : `max="${p.maximum}"`} step="${p.kind === "int" ? "1" : "any"}" title="${p.minimum ?? "无下限"} ～ ${p.maximum ?? "无上限"}"></label>`).join("")}</div></div>`;
        })
        .join("")
    : '<div class="empty-steps">添加算法步骤，或直接试算以验证输入和输出。</div>';
  $("steps")
    .querySelectorAll("[data-param]")
    .forEach(
      (el) =>
        (el.onchange = () => {
          if (!el.reportValidity()) return;
          steps[+el.dataset.param].parameters[el.dataset.key] =
            el.dataset.kind === "bool"
              ? el.checked
              : el.dataset.kind === "str"
                ? el.value
                : Number(el.value);
          changed();
        }),
    );
  $("steps")
    .querySelectorAll("[data-enable]")
    .forEach(
      (el) =>
        (el.onchange = () => {
          steps[+el.dataset.enable].enabled = el.checked;
          renderSteps();
          changed();
        }),
    );
  $("steps")
    .querySelectorAll("[data-delete]")
    .forEach(
      (el) =>
        (el.onclick = () => {
          steps.splice(+el.dataset.delete, 1);
          renderSteps();
          changed();
        }),
    );
  for (const [key, delta] of [
    ["up", -1],
    ["down", 1],
  ])
    $("steps")
      .querySelectorAll(`[data-${key}]`)
      .forEach(
        (el) =>
          (el.onclick = () => {
            const i = +el.dataset[key],
              j = i + delta;
            if (j < 0 || j >= steps.length) return;
            [steps[i], steps[j]] = [steps[j], steps[i]];
            renderSteps();
            changed();
          }),
      );
  updateSummary();
}
function addStep() {
  const spec = algorithms.find((a) => a.id === $("algorithm").value);
  if (steps.length >= 32) throw Error("最多支持 32 个步骤");
  steps.push({
    algorithm: spec.id,
    parameters: Object.fromEntries(
      spec.parameters.map((p) => [p.key, p.default]),
    ),
    enabled: true,
    manifest: spec.manifest,
    code_hash: spec.code_hash,
    version: spec.version,
  });
  renderSteps();
  changed();
}
async function submit(mode) {
  if (!active || !input) return;
  for (const el of $("steps").querySelectorAll("input"))
    if (!el.reportValidity()) return;
  const current = mode === "trial",
    scope = current ? "current" : $("scope").value,
    video = active.meta.kind === "video";
  const selection = {
    mode: scope,
    start:
      scope === "range" ? Number($("range-start").value) - (video ? 0 : 1) : 0,
    end:
      scope === "range" ? Number($("range-end").value) - (video ? 0 : 1) : null,
    current: input.ref,
    roi: scene.roi,
  };
  const currentSignature = signature(),
    response = await api("jobs", {
      mode,
      source_ids:
        !current && $("batch").checked
          ? sources.filter((s) => s.role === "input").map((s) => s.id)
          : [active.id],
      steps,
      selection,
      output_dir: $("output-dir").value,
      config_revision: currentSignature,
      display,
      result_display: resultDisplay,
    });
  if (current) trialSignatures.set(response.id, currentSignature);
  toast(current ? "试算已加入后台任务" : "处理任务已加入队列");
  await pollJobs();
}
const trialSignatures = new Map();
const statusText = {
  queued: "排队中",
  running: "计算中",
  cancelling: "正在取消",
  succeeded: "已完成",
  failed: "失败",
  cancelled: "已取消",
  completed_with_errors: "部分失败",
};
async function pollJobs(initial = false) {
  const list = await api("jobs");
  jobs = list;
  $("job-count").textContent = list.length;
  $("queue-note").textContent = list.some((j) => j.status === "running")
    ? "计算进行中 · 预览可继续"
    : "后台计算，预览可继续";
  $("job-list").innerHTML = list.length
    ? list
        .map(
          (j) =>
            `<div class="job-row"><span class="job-kind">${j.mode === "trial" ? "帧试算" : "正式运行"}</span><span class="job-name" title="${escape(j.name)}">${escape(j.name)}${j.source_count > 1 ? " · " + j.source_count + " 个来源" : ""}</span><span class="job-status ${j.status}">${statusText[j.status] || escape(j.status)}</span><span class="job-progress">${j.completed}${j.total ? " / " + j.total : " 帧"}</span>${["running", "queued", "cancelling"].includes(j.status) ? `<button data-cancel="${j.id}" class="quiet">取消</button>` : ""}<button data-detail="${j.id}" class="quiet">详情</button></div>`,
        )
        .join("")
    : '<div class="task-empty">试算和正式运行的进度、结果及日志会显示在这里。</div>';
  $("job-list")
    .querySelectorAll("[data-cancel]")
    .forEach(
      (b) =>
        (b.onclick = safe(async () => {
          await api(`jobs/${b.dataset.cancel}/cancel`, {});
          await pollJobs();
        })),
    );
  $("job-list")
    .querySelectorAll("[data-detail]")
    .forEach((b) => (b.onclick = safe(() => jobDetail(b.dataset.detail))));
  for (const j of list) {
    if (
      ["succeeded", "failed", "cancelled", "completed_with_errors"].includes(
        j.status,
      ) &&
      !fetchedJobs.has(j.id)
    ) {
      fetchedJobs.add(j.id);
      if (initial) continue;
      const full = await api(`jobs/${j.id}`);
      if (j.mode === "trial" && full.result) {
        const expected = trialSignatures.get(j.id);
        if (expected === signature()) {
          formalResult = null;
          result = full.result;
          resultSignature = expected;
          await paintResult(result);
          $("result-label").textContent = "当前帧精确试算";
          $("stale").hidden = true;
          setMode("compare");
          toast(
            `试算完成 · ${result.dtype} · 均值 ${Number(result.statistics.mean).toPrecision(5)}`,
          );
        } else toast("试算已完成，当前参数或帧已改变；结果保留在任务详情。");
      } else if (j.mode === "run" && full.directory) {
        const data = await api("state");
        sources = data.sources;
        updateSources();
        toast(
          j.status === "succeeded"
            ? "处理完成，结果已加入数据列表"
            : `任务结束：${statusText[j.status]}`,
          j.status === "failed",
        );
      } else if (full.error) toast(full.error, true);
    }
  }
  if (initial) {
    sources = (await api("state")).sources;
    updateSources();
  }
}
async function jobDetail(id) {
  const job = await api(`jobs/${id}`);
  $("job-dialog").showModal();
  const parts = [
    `<p>${escape(job.name)} · ${statusText[job.status]} · ${job.completed} 帧</p>`,
  ];
  if (job.error) parts.push(`<pre>${escape(job.error)}</pre>`);
  if (job.result)
    parts.push(
      `<p>结果：${job.result.dtype} · ${job.result.shape[1]} × ${job.result.shape[0]}<br>最小值 ${job.result.statistics.min} · 最大值 ${job.result.statistics.max} · 均值 ${job.result.statistics.mean}</p><button id="show-trial" class="quiet">查看该试算结果</button>`,
    );
  if (job.directory) parts.push(`<pre>${escape(job.directory)}</pre>`);
  if (job.run)
    parts.push(
      ...job.run.results.map(
        (r, i) =>
          `<p>来源 ${i + 1}：${escape(r.data_spec?.name || r.source.path)}<br>${statusText[r.status] || r.status} · ${r.completed_frames} 帧${r.error ? " · " + escape(r.error) : ""}</p>`,
      ),
    );
  if (job.logs.length) parts.push(`<pre>${escape(job.logs.join("\n"))}</pre>`);
  if (job.directory)
    parts.push(
      `<div class="downloads"><button data-download="manifest" class="quiet">运行记录 JSON</button>${(job.run?.results || []).map((r, i) => (r.status === "succeeded" ? `<button data-download="csv" data-index="${i}" class="quiet">来源 ${i + 1} CSV</button><button data-download="image" data-index="${i}" class="quiet">来源 ${i + 1} 首帧 TIFF</button>` : "")).join("")}</div>`,
    );
  $("job-detail").innerHTML = parts.join("");
  $("job-detail")
    .querySelectorAll("[data-download]")
    .forEach(
      (b) =>
        (b.onclick = safe(() =>
          download(id, b.dataset.download, Number(b.dataset.index || 0)),
        )),
    );
  if ($("show-trial"))
    $("show-trial").onclick = safe(async () => {
      if (active?.id !== job.source_ids[0])
        await selectSource(job.source_ids[0]);
      formalResult = null;
      await seekFrame(
        active.meta.kind === "video"
          ? { action: "time", time: job.result.ref.time }
          : { action: "index", index: job.result.ref.index },
      );
      result = job.result;
      resultSignature = trialSignatures.get(id);
      await paintResult(result);
      $("result-label").textContent = "历史试算结果";
      $("stale").hidden = resultSignature === signature();
      setMode("compare");
      $("job-dialog").close();
    });
}

const fileBrowser = new FileBrowser({
  icon,
  submit: () => submitDialog(),
  report: toast,
});
async function openDialog(mode) {
  await fileBrowser.open(
    mode,
    mode === "output" ? $("output-dir").value : state.workspace,
    state.workspace,
  );
}
async function submitDialog() {
  const button = $("dialog-submit");
  button.disabled = true;
  try {
    if (fileBrowser.mode === "data") {
      const requests = fileBrowser.requests(),
        report = [];
      let last;
      for (let i = 0; i < requests.length; i++) {
        const value = requests[i];
        $("dialog-submit").textContent =
          `正在导入 ${i + 1} / ${requests.length}`;
        try {
          const opened = await api("sources/open", value);
          sources = sources.filter((s) => s.id !== opened.source.id);
          sources.push(opened.source);
          last = opened;
          report.push({ path: value.path, ok: true });
        } catch (error) {
          report.push({ path: value.path, ok: false, error: error.message });
        }
      }
      updateSources();
      if (last) await selectSource(last.source.id);
      fileBrowser.showReport(report);
      const failures = report.filter((item) => !item.ok);
      if (failures.length) {
        fileBrowser.selected = failures.map((item) => item.path);
        fileBrowser.render();
        toast(
          `已导入 ${report.length - failures.length} 项，失败 ${failures.length} 项；可检查原因后重试。`,
          true,
        );
        return;
      }
      toast(`已导入 ${report.length} 个来源`);
    } else if (fileBrowser.mode === "plugin") {
      if (!fileBrowser.selected.length) throw Error("请选择算法 JSON 文件");
      const spec = await api("plugins/register", {
        path: fileBrowser.selected[0],
      });
      algorithms = algorithms.filter((a) => a.id !== spec.id);
      algorithms.push(spec);
      updateAlgorithms();
      $("algorithm").value = spec.id;
      toast("已注册 " + spec.name);
    } else if (fileBrowser.mode === "output") {
      $("output-dir").value = fileBrowser.path;
    } else if (fileBrowser.mode === "save") {
      const filename = $("filename").value.trim();
      if (!filename || /[\\/]/.test(filename))
        throw Error("请输入文件名，不包含目录");
      const data = await api("project/save", {
        path: fileBrowser.path + "/" + filename,
        view: projectView(),
      });
      state.project_path = data.path;
      $("project-title").textContent = filename;
      toast("项目已保存");
    } else if (fileBrowser.mode === "load") {
      if (!fileBrowser.selected.length) throw Error("请选择项目 JSON 文件");
      const data = await api("project/load", { path: fileBrowser.selected[0] });
      stopPlay();
      pendingSeek = null;
      sourceGeneration++;
      selectionRequest++;
      active = input = result = formalResult = null;
      state = { ...state, ...data.state };
      sources = state.sources;
      algorithms = state.algorithms;
      steps = data.view.steps;
      sourceViews = new Map(Object.entries(data.view.web_views || {}));
      display = data.view.web_display || {
        low: null,
        high: null,
        gamma: 1,
        colormap: "gray",
      };
      resultDisplay = data.view.web_result_display || {
        low: null,
        high: null,
        gamma: 1,
        colormap: "gray",
      };
      $("display-gamma").value = display.gamma;
      $("colormap").value = display.colormap;
      $("output-dir").value = data.view.output_dir || state.output_dir;
      $("project-title").textContent = fileBrowser.selected[0]
        .split(/[\\/]/)
        .pop();
      updateSources();
      updateAlgorithms();
      renderSteps();
      const first =
        sources.find(
          (s) => s.id === data.view.active_source && s.role === "input",
        ) || sources.find((s) => s.role === "input");
      if (first) {
        if (!sourceViews.has(first.id))
          sourceViews.set(first.id, {
            display: data.view.web_display || display,
            roi: data.view.web_roi || null,
          });
        await selectSource(first.id);
      } else {
        active = input = result = formalResult = null;
        originalViewer.clear();
        resultViewer.clear();
        $("welcome").hidden = false;
        updateSummary();
      }
      if (first && data.view.web_roi) {
        const q = data.view.web_roi;
        if (validROI(q, scene.width, scene.height)) {
          scene.roi = q;
          committedROI = structuredClone(q);
          updateROI();
          redraw();
        }
      }
      restoreProjectOptions(data.view);
      changed();
      toast("项目已恢复");
    }
    $("file-dialog").close();
  } finally {
    fileBrowser.render();
  }
}
function projectView() {
  stashView();
  return {
    steps,
    output_dir: $("output-dir").value,
    active_source: active?.id,
    web_display: display,
    web_result_display: resultDisplay,
    web_roi: scene.roi,
    web_views: Object.fromEntries(sourceViews),
    web_scope: $("scope").value,
    web_batch: $("batch").checked,
    web_range: [$("range-start").value, $("range-end").value],
    web_layout: {
      library: document
        .querySelector(".workspace")
        .classList.contains("library-collapsed"),
      processing: document
        .querySelector(".workspace")
        .classList.contains("processing-collapsed"),
    },
  };
}
function restoreProjectOptions(view) {
  if (["current", "all", "range"].includes(view.web_scope))
    $("scope").value = view.web_scope;
  $("range-fields").hidden = $("scope").value !== "range";
  if (view.web_range) {
    $("range-start").value = view.web_range[0];
    $("range-end").value = view.web_range[1];
  }
  $("batch").checked = !!view.web_batch;
  if (view.web_layout) {
    const w = document.querySelector(".workspace");
    w.classList.toggle("library-collapsed", !!view.web_layout.library);
    w.classList.toggle("processing-collapsed", !!view.web_layout.processing);
    updateLayout();
  }
  updateSummary();
}
let draftTimer,
  restoringDraft = false,
  focusPrevious = null;
function draftKey() {
  return state ? "eip-workbench-view-3:" + state.workspace : null;
}
function scheduleDraft() {
  if (!state || restoringDraft) return;
  clearTimeout(draftTimer);
  draftTimer = setTimeout(saveDraft, 350);
}
function saveDraft() {
  if (!state || restoringDraft) return;
  try {
    localStorage.setItem(draftKey(), JSON.stringify(projectView()));
  } catch {}
}
function updateLayout() {
  const workspace = document.querySelector(".workspace");
  for (const [id, name] of [
    ["toggle-library", "library-collapsed"],
    ["toggle-processing", "processing-collapsed"],
  ]) {
    const collapsed = workspace.classList.contains(name);
    $(id).setAttribute("aria-pressed", String(collapsed));
    $(id).classList.toggle("selected", collapsed);
  }
  const focused =
    workspace.classList.contains("library-collapsed") &&
    workspace.classList.contains("processing-collapsed");
  $("focus-preview").setAttribute("aria-pressed", String(focused));
  $("focus-preview").classList.toggle("selected", focused);
  requestAnimationFrame(redraw);
  scheduleDraft();
}
for (const [id, name] of [
  ["toggle-library", "library-collapsed"],
  ["toggle-processing", "processing-collapsed"],
])
  $(id).onclick = () => {
    document.querySelector(".workspace").classList.toggle(name);
    focusPrevious = null;
    updateLayout();
  };
$("focus-preview").onclick = () => {
  const workspace = document.querySelector(".workspace");
  if (
    workspace.classList.contains("library-collapsed") &&
    workspace.classList.contains("processing-collapsed")
  ) {
    workspace.classList.toggle(
      "library-collapsed",
      focusPrevious?.library || false,
    );
    workspace.classList.toggle(
      "processing-collapsed",
      focusPrevious?.processing || false,
    );
    focusPrevious = null;
  } else {
    focusPrevious = {
      library: workspace.classList.contains("library-collapsed"),
      processing: workspace.classList.contains("processing-collapsed"),
    };
    workspace.classList.add("library-collapsed", "processing-collapsed");
  }
  updateLayout();
};

for (const id of ["import", "import-small", "welcome-import"])
  $(id).onclick = safe(() => openDialog("data"));
$("load-project").onclick = safe(() => openDialog("load"));
$("save-project").onclick = safe(() => openDialog("save"));
$("plugin").onclick = safe(() => openDialog("plugin"));
$("choose-output").onclick = safe(() => openDialog("output"));
$("dialog-close").onclick = () => $("file-dialog").close();
$("job-close").onclick = () => $("job-dialog").close();
$("source-close").onclick = () => $("source-dialog").close();
$("promote-result").onclick = safe(promoteResult);
$("dialog-submit").onclick = safe(submitDialog);
$("demo").onclick = safe(async () => {
  const b = $("demo");
  b.disabled = true;
  b.textContent = "正在生成示例…";
  try {
    const data = await api("demo", {});
    for (const opened of data) {
      sources = sources.filter((s) => s.id !== opened.source.id);
      sources.push(opened.source);
    }
    updateSources();
    await selectSource(data[data.length - 1].source.id);
    toast("已打开实际生成的示例文件，属于模拟数据");
  } finally {
    b.disabled = false;
    b.textContent = "体验示例数据 →";
  }
});
$("add-step").onclick = safe(addStep);
$("fit").onclick = fit;
$("actual").onclick = () => {
  const canvas = visibleViewer().canvas;
  scene.zoom =
    1 /
    Math.min(
      (canvas.clientWidth - 40) / scene.width,
      (canvas.clientHeight - 50) / scene.height,
    );
  scene.panX = scene.panY = 0;
  redraw();
};
for (const tool of ["pan", "roi"])
  $("tool-" + tool).onclick = () => {
    scene.tool = tool;
    $("tool-pan").classList.toggle("selected", tool === "pan");
    $("tool-roi").classList.toggle("selected", tool === "roi");
  };
$("clear-roi").onclick = () => commitROI(null);
$("roi-full").onclick = () => {
  if (input)
    commitROI({ x: 0, y: 0, width: scene.width, height: scene.height });
};
for (const key of ["x", "y", "width", "height"])
  $("roi-" + key).onchange = safe(() => {
    const roi = Object.fromEntries(
      ["x", "y", "width", "height"].map((k) => [
        k,
        Number($("roi-" + k).value),
      ]),
    );
    try {
      commitROI(roi);
    } catch (error) {
      updateROI();
      throw error;
    }
  });
$("roi-undo").onclick = () => {
  if (!roiUndo.length) return;
  roiRedo.push(structuredClone(scene.roi));
  scene.roi = roiUndo.pop();
  committedROI = structuredClone(scene.roi);
  updateROI();
  changed();
  redraw();
};
$("roi-redo").onclick = () => {
  if (!roiRedo.length) return;
  roiUndo.push(structuredClone(scene.roi));
  scene.roi = roiRedo.pop();
  committedROI = structuredClone(scene.roi);
  updateROI();
  changed();
  redraw();
};
document
  .querySelectorAll("[data-view]")
  .forEach((b) => (b.onclick = () => setMode(b.dataset.view)));
$("scope").onchange = () => {
  $("range-fields").hidden = $("scope").value !== "range";
  updateSummary();
  scheduleDraft();
};
$("batch").onchange = () => {
  updateSummary();
  scheduleDraft();
};
for (const id of ["range-start", "range-end"]) $(id).onchange = scheduleDraft;
$("trial").onclick = safe(() => submit("trial"));
$("run").onclick = safe(() => submit("run"));
$("previous").onclick = safe(() => step(-1));
$("next").onclick = safe(() => step(1));
$("play").onclick = safe(play);
$("scrubber").oninput = safe(() => {
  stopPlay();
  return seekFrame(
    active?.meta.kind === "video"
      ? { action: "time", time: Number($("scrubber").value) }
      : { action: "index", index: Number($("scrubber").value) },
  );
});
$("seek").onclick = safe(() => {
  if (!$("seek-value").reportValidity()) return;
  stopPlay();
  return seekFrame(
    active?.meta.kind === "video"
      ? { action: "time", time: Number($("seek-value").value) }
      : { action: "index", index: Number($("seek-value").value) - 1 },
  );
});
function fillDisplayControls(snapshot, settings) {
  const label = snapshot?.kind === "label";
  $("display-low").value = snapshot?.display.low ?? settings.low ?? "";
  $("display-high").value = snapshot?.display.high ?? settings.high ?? "";
  $("display-gamma").value = settings.gamma;
  $("colormap").value = settings.colormap;
  for (const id of [
    "display-low",
    "display-high",
    "display-gamma",
    "colormap",
    "auto-display",
  ])
    $(id).disabled = !snapshot || label;
  $("copy-display").disabled =
    !input ||
    !result ||
    label ||
    input.kind === "label" ||
    result.kind === "label";
  $("display-status").textContent = label
    ? "标签使用离散色表；原始标签编号保持不变"
    : "输入与结果分别设置，仅改变显示";
  if (snapshot) drawHistogram(snapshot);
  else {
    const c = $("histogram");
    c.getContext("2d").clearRect(0, 0, c.width, c.height);
    $("histogram-min").textContent = $("histogram-max").textContent = "";
  }
}
async function paintResult(snapshot) {
  await resultViewer.set(snapshot);
  if (result?.token !== snapshot.token) return;
  $("result-meta").textContent =
    `${snapshot.dtype} · ${snapshot.shape[1]} × ${snapshot.shape[0]}`;
  fillDisplayControls(
    $("display-target").value === "result" ? snapshot : input,
    $("display-target").value === "result" ? resultDisplay : display,
  );
  updateSummary();
}
$("display-target").onchange = () =>
  fillDisplayControls(
    $("display-target").value === "result" ? result : input,
    $("display-target").value === "result" ? resultDisplay : display,
  );
async function renderDisplay(
  auto = false,
  supplied = null,
  windowChanged = false,
) {
  const target = $("display-target").value,
    snapshot = target === "result" ? result : input;
  if (!snapshot) return;
  const generation = ++displayRevision,
    token = snapshot.token;
  const settings = supplied || {
    low: auto
      ? null
      : windowChanged
        ? Number($("display-low").value)
        : (target === "result" ? resultDisplay : display).low,
    high: auto
      ? null
      : windowChanged
        ? Number($("display-high").value)
        : (target === "result" ? resultDisplay : display).high,
    gamma: Number($("display-gamma").value),
    colormap: $("colormap").value,
  };
  const rendered = await api(`frames/${token}/render`, { display: settings });
  if (
    generation !== displayRevision ||
    (target === "result" ? result : input)?.token !== token
  )
    return;
  if (target === "result") {
    resultDisplay = settings;
    result = rendered;
    await paintResult(result);
  } else {
    display = settings;
    await setInput(rendered);
  }
  scheduleDraft();
}
$("copy-display").onclick = safe(async () => {
  const from = $("display-target").value,
    snapshot = from === "result" ? result : input;
  if (!snapshot) return;
  const settings = { ...snapshot.display };
  $("display-target").value = from === "input" ? "result" : "input";
  await renderDisplay(false, settings);
});
for (const id of ["display-low", "display-high"])
  $(id).onchange = safe(() => renderDisplay(false, null, true));
for (const id of ["display-gamma", "colormap"])
  $(id).onchange = safe(() => renderDisplay());
$("auto-display").onclick = safe(() => renderDisplay(true));
$("toggle-tasks").onclick = () =>
  document.querySelector(".workspace").classList.toggle("task-collapse");
document.addEventListener("keydown", (e) => {
  if (
    ["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) ||
    document.querySelector("dialog[open]")
  )
    return;
  if (
    scene.tool === "roi" &&
    scene.roi &&
    ["ArrowRight", "ArrowLeft", "ArrowUp", "ArrowDown"].includes(e.key)
  ) {
    e.preventDefault();
    const delta = e.shiftKey ? 10 : 1;
    commitROI(
      moveROI(
        scene.roi,
        e.key === "ArrowRight" ? delta : e.key === "ArrowLeft" ? -delta : 0,
        e.key === "ArrowDown" ? delta : e.key === "ArrowUp" ? -delta : 0,
        scene.width,
        scene.height,
      ),
    );
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    $(e.shiftKey ? "roi-redo" : "roi-undo").click();
    return;
  }
  if (e.code === "Space") {
    e.preventDefault();
    safe(play)();
  } else if (e.key === "ArrowRight") safe(() => step(1))();
  else if (e.key === "ArrowLeft") safe(() => step(-1))();
  else if (e.key.toLowerCase() === "f") fit();
});
window.addEventListener("pagehide", () => {
  saveDraft();
  stopPlay();
});
async function bootstrap() {
  state = await api("state");
  session(state.session_token);
  sources = state.sources;
  algorithms = state.algorithms;
  $("version").textContent = "v" + state.version;
  $("connection").textContent = "本机已连接";
  $("workspace-path").textContent = state.workspace;
  $("workspace-path").title = state.workspace;
  if (state.project_path)
    $("project-title").textContent = state.project_path.split(/[\\/]/).pop();
  $("output-dir").value = state.output_dir;
  updateSources();
  updateAlgorithms();
  renderSteps();
  restoringDraft = true;
  let draft = null;
  try {
    draft = JSON.parse(localStorage.getItem(draftKey()) || "null");
  } catch {}
  if (draft) {
    if (
      Array.isArray(draft.steps) &&
      draft.steps.every((step) =>
        algorithms.some(
          (a) =>
            a.id === step.algorithm &&
            a.code_hash === step.code_hash &&
            a.version === step.version,
        ),
      )
    )
      steps = draft.steps;
    else if (draft.steps?.length)
      toast("草稿中的算法版本已变化，请重新添加处理步骤", true);
    sourceViews = new Map(Object.entries(draft.web_views || {}));
    resultDisplay = draft.web_result_display || resultDisplay;
    $("output-dir").value = draft.output_dir || state.output_dir;
    restoreProjectOptions(draft);
    renderSteps();
  }
  const initial =
    sources.find((s) => s.role === "input" && s.id === draft?.active_source) ||
    sources.find((s) => s.role === "input");
  if (initial) await selectSource(initial.id);
  if (draft) restoreProjectOptions(draft);
  restoringDraft = false;
  await pollJobs(true);
  let polling = false;
  setInterval(async () => {
    if (polling) return;
    polling = true;
    try {
      await pollJobs();
      $("connection").textContent = "本机已连接";
    } catch {
      $("connection").textContent = "连接中断";
    } finally {
      polling = false;
    }
  }, 900);
}
safe(bootstrap)();
