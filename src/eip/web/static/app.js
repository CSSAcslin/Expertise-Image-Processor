import { api, session, download } from "./api.js";
import { Viewer } from "./viewer.js";

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
  requestBusy = false,
  pendingSeek = null,
  playing = false,
  fetchedJobs = new Set(),
  jobs = [];
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
    changed();
    updateROI();
  },
  pixel: (snapshot, x, y) => {
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
        $("pixel-info").textContent =
          `x ${x} · y ${y}  |  ${Array.isArray(data.value) ? data.value.join(", ") : data.value}  ${data.dtype}`;
      } catch {}
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
}
function updateROI() {
  const q = scene.roi;
  $("roi-info").textContent = q
    ? `x ${q.x} · y ${q.y}  /  ${q.width} × ${q.height} px`
    : "全幅图像";
}
function updateSummary() {
  const count = steps.filter((s) => s.enabled).length;
  $("run-summary").textContent = active
    ? `${$("batch").checked ? sources.filter((s) => s.role === "input").length : 1} 个来源 · ${count} 个处理步骤${scene.roi ? " · ROI" : ""}`
    : "打开数据以开始处理";
  $("trial").disabled = $("run").disabled = !active || !input;
  for (const id of ["previous", "next", "play", "seek", "seek-value"])
    $(id).disabled = !active || !input;
  $("scrubber").disabled =
    !active ||
    !input ||
    (active.meta.kind === "video" && !active.meta.duration);
}
function updateSources() {
  const inputs = sources.filter((s) => s.role === "input"),
    results = sources.filter((s) => s.role === "result");
  $("source-count").textContent = inputs.length;
  $("result-count").textContent = results.length;
  const card = (s) =>
    `<div class="source-card ${s.id === active?.id || s.id === formalResult?.id ? "selected" : ""}" tabindex="0" role="button" data-source="${s.id}" aria-label="预览 ${escape(s.display_name || s.meta.name)}"><img src="${s.thumbnail}" alt=""><div class="source-copy"><strong title="${escape(s.source.path)}">${escape(s.display_name || s.meta.name)}</strong><span class="source-kind">${s.role === "result" ? "处理结果" : s.meta.kind === "video" ? "视频" : s.meta.kind === "sequence" ? "图像序列" : "图像"}</span><small>${s.meta.shape[1]} × ${s.meta.shape[0]} · ${escape(s.meta.dtype)}</small></div></div>`;
  $("source-list").innerHTML = inputs.length
    ? inputs.map(card).join("")
    : '<div class="library-empty">导入图像、视频<br>或图像序列目录</div>';
  $("result-list").innerHTML = results.map(card).join("");
  document.querySelectorAll("[data-source]").forEach((el) => {
    el.onclick = safe(() => selectSource(el.dataset.source));
    el.onkeydown = (e) => {
      if (e.key === "Enter") el.click();
    };
  });
  updateSummary();
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
}
async function setInput(snapshot, reset = false) {
  input = snapshot;
  scene.width = snapshot.base_shape[1];
  scene.height = snapshot.base_shape[0];
  if (reset) {
    scene.roi = null;
    fit();
    updateROI();
  }
  await originalViewer.set(snapshot);
  $("welcome").hidden = true;
  $("preview-title").textContent = active.meta.name;
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
  $("display-low").value = snapshot.display.low;
  $("display-high").value = snapshot.display.high;
  drawHistogram(snapshot);
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
      : entry.meta.timestamp_source;
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
async function selectSource(id) {
  stopPlay();
  pendingSeek = null;
  const entry = sources.find((s) => s.id === id);
  if (!entry) return;
  if (entry.role === "result") {
    const matching = sources.find((s) => s.id === entry.input_id);
    if (matching && active?.id !== matching.id) await selectSource(matching.id);
    formalResult = entry;
    await syncFormalResult();
    updateSources();
    setMode("compare");
    toast("结果按来源帧/时间同步显示");
    return;
  }
  active = entry;
  const generation = ++sourceGeneration;
  input = null;
  originalViewer.clear();
  $("frame-info").textContent = "正在打开数据…";
  result = formalResult = null;
  resultSignature = null;
  resultViewer.clear();
  configureTimeline(entry);
  updateSources();
  setMode("input");
  const snapshot = await api(`sources/${id}/frame`, {
    action: "open",
    display,
  });
  if (active.id !== id || generation !== sourceGeneration) return;
  await setInput(snapshot, true);
  if (entry.meta.notes?.length) toast(entry.meta.notes.join("；"));
}
async function seekFrame(value) {
  if (!active || !input) return;
  if (requestBusy) {
    pendingSeek = value;
    return;
  }
  requestBusy = true;
  const id = active.id;
  $("timeline-note").textContent = "正在解码…";
  try {
    const snapshot = await api(`sources/${id}/frame`, { ...value, display });
    if (active?.id !== id) return;
    if (snapshot.eof) {
      stopPlay();
      return;
    }
    await setInput(snapshot);
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
    snapshot = await api(`sources/${entry.id}/frame`, {
      action: "match",
      reference: input.ref,
      display,
    });
  if (formalResult !== entry) return;
  $("stale").hidden = true;
  resultSignature = null;
  if (snapshot.unavailable) {
    result = null;
    resultViewer.clear();
    $("result-label").textContent = "该输入帧没有对应输出";
  } else {
    result = snapshot;
    await resultViewer.set(snapshot);
    $("result-label").textContent =
      `正式结果 · 输出第 ${snapshot.ref.index + 1} 帧`;
  }
}
function stopPlay() {
  playing = false;
  $("play").innerHTML = icon("play");
  $("play").setAttribute("aria-label", "播放");
}
function step(direction) {
  if (!active || !input) return;
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
  $("play").innerHTML = icon("pause");
  $("play").setAttribute("aria-label", "暂停");
  while (playing) {
    const before = input.token;
    await step(1);
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
async function pollJobs() {
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
      const full = await api(`jobs/${j.id}`);
      if (j.mode === "trial" && full.result) {
        const expected = trialSignatures.get(j.id);
        if (expected === signature()) {
          formalResult = null;
          result = full.result;
          resultSignature = expected;
          await resultViewer.set(result);
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
      await resultViewer.set(result);
      $("result-label").textContent = "历史试算结果";
      $("stale").hidden = resultSignature === signature();
      setMode("compare");
      $("job-dialog").close();
    });
}

let fileMode = "data",
  filePath = "",
  selectedFiles = [];
async function browse(path) {
  const data = await api(
    `browse?path=${encodeURIComponent(path || state.workspace)}&mode=${fileMode === "data" ? "data" : "json"}`,
  );
  filePath = data.path;
  selectedFiles = [];
  $("browse-path").value = data.path;
  $("selection-info").textContent = "当前目录：" + data.path;
  $("drives").innerHTML = data.drives
    .map(
      (d) =>
        `<button data-drive="${escape(d)}" class="quiet">${escape(d)}</button>`,
    )
    .join("");
  $("drives")
    .querySelectorAll("button")
    .forEach((b) => (b.onclick = safe(() => browse(b.dataset.drive))));
  $("file-list").innerHTML =
    data.entries
      .map(
        (e) =>
          `<div class="file-row" data-path="${escape(e.path)}" data-directory="${e.directory}" role="button" tabindex="0">${icon(e.directory ? "folder" : fileMode === "data" ? "image" : "file")}<span>${escape(e.name)}</span><small>${e.directory ? "目录" : (e.size / 1024 / 1024).toFixed(2) + " MB"}</small></div>`,
      )
      .join("") || '<div class="library-empty">这个目录没有可用文件</div>';
  $("file-list")
    .querySelectorAll(".file-row")
    .forEach((el) => {
      el.onclick = (event) => {
        if (el.dataset.directory === "true") {
          safe(() => browse(el.dataset.path))();
          return;
        }
        if (fileMode === "data" && (event.ctrlKey || event.metaKey)) {
          selectedFiles = selectedFiles.includes(el.dataset.path)
            ? selectedFiles.filter((p) => p !== el.dataset.path)
            : [...selectedFiles, el.dataset.path];
        } else selectedFiles = [el.dataset.path];
        $("file-list")
          .querySelectorAll(".file-row")
          .forEach((r) =>
            r.classList.toggle(
              "selected",
              selectedFiles.includes(r.dataset.path),
            ),
          );
        $("selection-info").textContent = selectedFiles.length
          ? `已选择 ${selectedFiles.length} 个文件：${selectedFiles[0]}`
          : "尚未选择文件";
      };
      el.ondblclick = safe(() => submitDialog());
      el.onkeydown = (e) => {
        if (e.key === "Enter") el.click();
      };
    });
  $("browse-up").onclick = safe(() => browse(data.parent));
  if (data.truncated) toast("目录项超过 1000，当前显示前 1000 项");
}
async function openDialog(mode) {
  fileMode = mode;
  const names = {
    data: "打开本地数据",
    plugin: "接入 Python 算法",
    load: "打开项目",
    save: "保存项目",
    output: "选择输出目录",
  };
  $("dialog-title").textContent = names[mode];
  $("dialog-submit").textContent = {
    data: "打开数据 / 当前目录",
    plugin: "注册算法",
    load: "打开项目",
    save: "保存项目",
    output: "使用当前目录",
  }[mode];
  $("interval-label").hidden = mode !== "data";
  $("filename-label").hidden = mode !== "save";
  $("dialog-note").textContent =
    mode === "plugin"
      ? "选择可信的算法 JSON 描述文件；代码将在试算或运行时执行。"
      : mode === "data"
        ? "单击目录进入，Ctrl + 单击可多选文件；无文件选中时，当前目录作为图像序列。"
        : "路径位于本机；项目记录引用来源及参数，不复制原始数据。";
  try {
    await browse(mode === "output" ? $("output-dir").value : state.workspace);
  } catch {
    await browse(state.workspace);
  }
  $("file-dialog").showModal();
}
async function navigate() {
  const path = $("browse-path").value.trim();
  try {
    await browse(path);
  } catch (error) {
    if (fileMode === "data" || fileMode === "plugin" || fileMode === "load") {
      selectedFiles = [path];
      $("selection-info").textContent = path;
    } else throw error;
  }
}
async function submitDialog() {
  const button = $("dialog-submit");
  button.disabled = true;
  try {
    if (fileMode === "data") {
      const paths = selectedFiles.length ? selectedFiles : [filePath],
        interval = $("interval").value ? Number($("interval").value) : null;
      let last;
      for (const path of paths) {
        const opened = await api("sources/open", {
          path,
          kind: selectedFiles.length ? "auto" : "sequence",
          interval,
        });
        sources = sources.filter((s) => s.id !== opened.source.id);
        sources.push(opened.source);
        last = opened;
      }
      updateSources();
      if (last) await selectSource(last.source.id);
    } else if (fileMode === "plugin") {
      if (!selectedFiles.length) throw Error("请选择算法 JSON 文件");
      const spec = await api("plugins/register", { path: selectedFiles[0] });
      algorithms = algorithms.filter((a) => a.id !== spec.id);
      algorithms.push(spec);
      updateAlgorithms();
      $("algorithm").value = spec.id;
      toast("已注册 " + spec.name);
    } else if (fileMode === "output") {
      $("output-dir").value = filePath;
    } else if (fileMode === "save") {
      const filename = $("filename").value.trim();
      if (!filename || /[\\/]/.test(filename))
        throw Error("请输入文件名，不包含目录");
      const data = await api("project/save", {
        path: filePath + "/" + filename,
        view: projectView(),
      });
      state.project_path = data.path;
      $("project-title").textContent = filename;
      toast("项目已保存");
    } else if (fileMode === "load") {
      if (!selectedFiles.length) throw Error("请选择项目 JSON 文件");
      const data = await api("project/load", { path: selectedFiles[0] });
      state = { ...state, ...data.state };
      sources = state.sources;
      algorithms = state.algorithms;
      steps = data.view.steps;
      display = data.view.web_display || {
        low: null,
        high: null,
        gamma: 1,
        colormap: "gray",
      };
      $("display-gamma").value = display.gamma;
      $("colormap").value = display.colormap;
      $("output-dir").value = data.view.output_dir || state.output_dir;
      $("project-title").textContent = selectedFiles[0].split(/[\\/]/).pop();
      updateSources();
      updateAlgorithms();
      renderSteps();
      const first =
        sources.find(
          (s) => s.id === data.view.active_source && s.role === "input",
        ) || sources.find((s) => s.role === "input");
      if (first) await selectSource(first.id);
      else {
        active = input = result = formalResult = null;
        originalViewer.clear();
        resultViewer.clear();
        $("welcome").hidden = false;
        updateSummary();
      }
      if (data.view.web_roi) {
        const q = data.view.web_roi;
        if (
          q.x >= 0 &&
          q.y >= 0 &&
          q.width > 0 &&
          q.height > 0 &&
          q.x + q.width <= scene.width &&
          q.y + q.height <= scene.height
        ) {
          scene.roi = q;
          updateROI();
          redraw();
        }
      }
      restoreProjectOptions(data.view);
      toast("项目已恢复");
    }
    $("file-dialog").close();
  } finally {
    button.disabled = false;
  }
}
function projectView() {
  return {
    steps,
    output_dir: $("output-dir").value,
    active_source: active?.id,
    web_display: display,
    web_roi: scene.roi,
    web_scope: $("scope").value,
    web_batch: $("batch").checked,
    web_range: [$("range-start").value, $("range-end").value],
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
  updateSummary();
}

for (const id of ["import", "import-small", "welcome-import"])
  $(id).onclick = safe(() => openDialog("data"));
$("load-project").onclick = safe(() => openDialog("load"));
$("save-project").onclick = safe(() => openDialog("save"));
$("plugin").onclick = safe(() => openDialog("plugin"));
$("choose-output").onclick = safe(() => openDialog("output"));
$("dialog-close").onclick = () => $("file-dialog").close();
$("job-close").onclick = () => $("job-dialog").close();
$("browse-go").onclick = safe(navigate);
$("browse-path").onkeydown = (e) => {
  if (e.key === "Enter") safe(navigate)();
};
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
$("clear-roi").onclick = () => {
  scene.roi = null;
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
};
$("batch").onchange = updateSummary;
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
  stopPlay();
  return seekFrame(
    active?.meta.kind === "video"
      ? { action: "time", time: Number($("seek-value").value) }
      : { action: "index", index: Number($("seek-value").value) - 1 },
  );
});
async function renderDisplay(auto = false) {
  if (!input) return;
  const generation = ++displayRevision,
    token = input.token,
    resultToken = result?.token;
  display = {
    low: auto ? null : Number($("display-low").value),
    high: auto ? null : Number($("display-high").value),
    gamma: Number($("display-gamma").value),
    colormap: $("colormap").value,
  };
  const settings = { ...display };
  const snapshot = await api(`frames/${token}/render`, { display: settings });
  if (generation !== displayRevision || input?.token !== token) return;
  await setInput(snapshot);
  if (resultToken) {
    const rendered = await api(`frames/${resultToken}/render`, {
      display: settings,
    });
    if (generation !== displayRevision || result?.token !== resultToken) return;
    result = rendered;
    await resultViewer.set(result);
  }
}
for (const id of ["display-low", "display-high", "display-gamma", "colormap"])
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
  if (e.code === "Space") {
    e.preventDefault();
    safe(play)();
  } else if (e.key === "ArrowRight") safe(() => step(1))();
  else if (e.key === "ArrowLeft") safe(() => step(-1))();
  else if (e.key.toLowerCase() === "f") fit();
});
window.addEventListener("pagehide", stopPlay);
async function bootstrap() {
  state = await api("state");
  session(state.session_token);
  sources = state.sources;
  algorithms = state.algorithms;
  $("version").textContent = "v" + state.version;
  $("connection").textContent = "本机已连接";
  $("workspace-path").textContent = state.workspace;
  $("workspace-path").title = state.workspace;
  $("output-dir").value = state.output_dir;
  updateSources();
  updateAlgorithms();
  renderSteps();
  if (sources.some((s) => s.role === "input"))
    await selectSource(sources.find((s) => s.role === "input").id);
  await pollJobs();
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
