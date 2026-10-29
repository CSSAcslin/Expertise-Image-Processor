import { api } from "./api.js";
import { svgIcon } from "./icons.js";

const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const percent = value => Number.isFinite(value) ? `${value.toFixed(1)}%` : "未知";
const ratio = (value, total) => Number.isFinite(value) && total > 0 ? value / total * 100 : null;
const size = value => !Number.isFinite(value) ? "未知" : value >= 1073741824 ? `${(value / 1073741824).toFixed(1)} GiB` : `${Math.round(value / 1048576)} MiB`;
const mib = value => Number.isFinite(value) ? `${(value / 1048576).toFixed(1)} MiB` : "未知";
const budget = value => Number.isFinite(value) ? `${value} MiB` : "未知";

export function setupDeviceMonitor(host) {
  const state = host.querySelector("[data-device-state]"), cards = host.querySelector("[data-device-cards]");
  const pause = host.querySelector("[data-device-pause]"), gpuSelect = host.querySelector("[data-device-gpu]");
  const histories = new Map();
  let timer, epoch = 0, paused = false, busy = false, lastData = null, selectedGpu = null, wasActive = null;
  const active = () => !host.classList.contains("is-collapsed") && !document.hidden && !paused;

  function gap() {
    for (const history of histories.values()) {
      const last = history.at(-1);
      if (last && last.value != null) history.push({time: last.time, value: null});
      if (history.length > 61) history.shift();
    }
  }

  function remember(key, time, value) {
    let history = histories.get(key) || [];
    if (history.at(-1)?.time > time) history = [];
    history = history.filter(point => point.time >= time - 120);
    const previous = history.at(-1);
    if (previous?.time !== time) {
      if (previous && time - previous.time > 5) history.push({time: previous.time, value: null});
      history.push({time, value: Number.isFinite(value) ? Math.min(100, Math.max(0, value)) : null});
    } else if (previous.value != null && !Number.isFinite(value)) {
      history.push({time, value: null});
    } else if (previous.value == null && Number.isFinite(value)) {
      // A paused interval remains a break even when the server still has its last cached sample.
      history.push({time, value: Math.min(100, Math.max(0, value))});
    }
    histories.set(key, history.slice(-61));
    while (histories.size > 256) histories.delete(histories.keys().next().value);
  }

  function chart(series, now, label) {
    const paths = series.map(([key, application]) => {
      let d = "", connected = false, last = null;
      for (const point of histories.get(key) || []) {
        if (point.value == null || point.time < now - 120) { connected = false; continue; }
        const x = Math.max(0, Math.min(150, 150 - (now - point.time) * 1.25)), y = 33 - point.value * .30;
        d += `${connected ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
        connected = true; last = {x, y};
      }
      const color = application ? "app" : "system";
      return `<path data-series="${escape(key)}" class="device-${color}-line" d="${d}"/>${last ? `<circle class="device-${color}-point" cx="${last.x}" cy="${last.y}"/>` : ""}`;
    }).join("");
    return `<svg class="device-chart" viewBox="0 0 150 36" preserveAspectRatio="none" role="img" aria-label="${escape(label)}；固定 0 至 100%；最近两分钟；未知读数不连接"><path class="device-chart-grid" d="M0 3H150M0 18H150M0 33H150"/>${paths}</svg>`;
  }

  function metric(key, label, value, note, title, series, now) {
    return `<article class="device-metric" data-device-metric="${key}" title="${escape(title)}"><div class="device-metric-heading"><strong>${label}</strong><span>0–100%</span></div><div class="device-metric-value">${escape(value)}</div>${chart(series, now, label)}<div class="device-metric-note">${escape(note)}</div></article>`;
  }

  function record(data) {
    const time = data.sampled_at, cpu = data.cpu || {}, memory = data.memory || {}, gpu = data.gpu || {};
    remember("cpu-system", time, cpu.system_percent); remember("cpu-app", time, cpu.application_percent);
    remember("memory-system", time, ratio(memory.used_bytes, memory.total_bytes)); remember("memory-app", time, ratio(memory.application_bytes, memory.total_bytes));
    const present = new Set();
    for (const card of (gpu.cards || []).slice(0, 64)) {
      const key = `gpu-${card.uuid}`, gpuTime = gpu.sampled_at ?? time;
      present.add(key);
      remember(`${key}-util`, gpuTime, card.utilization_percent);
      remember(`${key}-memory`, gpuTime, ratio(card.used_bytes, card.total_bytes));
      remember(`${key}-app`, gpuTime, ratio(card.application_bytes, card.total_bytes));
    }
    for (const [key, history] of histories) {
      const missing = key.startsWith("gpu-") && ![...present].some(prefix => key.startsWith(prefix + "-"));
      if ((!gpu.available || missing) && key.startsWith("gpu-") && history.at(-1)?.value != null) {
        history.push({time, value: null});
        if (history.length > 61) history.shift();
      }
    }
  }

  function render(data) {
    const cpu = data.cpu || {}, memory = data.memory || {}, gpu = data.gpu || {}, now = data.sampled_at;
    const available = (gpu.cards || []).slice(0, 64);
    if (!available.some(card => card.uuid === selectedGpu)) selectedGpu = available[0]?.uuid ?? null;
    // Preserve focus and selection while regular samples arrive.
    const options = available.map(card => `<option value="${escape(card.uuid)}">GPU ${escape(card.index)} · ${escape(card.name)}</option>`).join("") || `<option value="">GPU 不可用</option>`;
    if (gpuSelect.dataset.options !== options) { gpuSelect.innerHTML = options; gpuSelect.dataset.options = options; }
    gpuSelect.value = selectedGpu ?? ""; gpuSelect.disabled = available.length <= 1;
    const card = available.find(item => item.uuid === selectedGpu), key = `gpu-${card?.uuid}`;
    gpuSelect.title = card ? `GPU ${card.index} · ${card.name}` : gpu.reason || "未检测到 GPU";
    const cpuTitle = `CPU 整机利用率；本程序包含工作台、数据读取、阶段检查及处理进程（不含浏览器）。两者均按 ${cpu.logical_processors ?? "全部"} 个逻辑处理器归一为 0–100%。${[cpu.reason, cpu.system_reason, cpu.application_reason].filter(Boolean).join("；")}`;
    const memoryTitle = `整机物理内存占用；本程序为各工作进程驻留内存之和，可能包含共享页。空闲 ${size(memory.available_bytes)}。${memory.application_reason || ""}`;
    const gpuTitle = card ? `GPU ${card.index} · ${card.name}；整张显卡的计算利用率，包含其他应用。${card.utilization_reason || ""}` : gpu.reason || "GPU 利用率不可用";
    const vramTitle = card ? `整卡显存分配占用；本程序由驱动上报，包含 CUDA 上下文等开销。空闲 ${size(card.free_bytes)}。${card.application_reason || ""}` : gpu.reason || "显存不可用";
    cards.innerHTML = metric("cpu", "CPU", percent(cpu.system_percent), `本程序 ${percent(cpu.application_percent)}`, cpuTitle, [["cpu-system", false], ["cpu-app", true]], now)
      + metric("memory", "内存", `${size(memory.used_bytes)} / ${size(memory.total_bytes)}`, `本程序 ${size(memory.application_bytes)}`, memoryTitle, [["memory-system", false], ["memory-app", true]], now)
      + metric("gpu", "GPU", percent(card?.utilization_percent), card ? "整卡计算利用率" : "当前不可用", gpuTitle, [[`${key}-util`, false]], now)
      + metric("vram", "显存", card ? `${size(card.used_bytes)} / ${size(card.total_bytes)}` : "未知", `本程序 ${size(card?.application_bytes)}`, vramTitle, [[`${key}-memory`, false], [`${key}-app`, true]], now);
    const activity = data.activity || {}, activityLabel = activity.busy ? `${activity.job_name || "处理 / 阶段检查"}${activity.pending ? ` · 排队 ${activity.pending}` : ""}` : "空闲";
    host.querySelector("[data-device-activity]").textContent = activityLabel;
    host.querySelector("[data-device-activity]").title = activityLabel;
    host.title = `CPU ${percent(cpu.system_percent)} · 内存 ${percent(ratio(memory.used_bytes, memory.total_bytes))} · GPU ${percent(card?.utilization_percent)} · 显存 ${percent(ratio(card?.used_bytes, card?.total_bytes))}`;
    const b = data.budgets || {}, cache = data.cache || {}, processes = data.processes || [];
    const count = Number.isFinite(cache.frame_cache_count) ? `${cache.frame_cache_count} 帧` : "帧数未知";
    const readout = host.querySelector("#resource-readout");
    const text = `当前进程内存 ${mib(memory.application_bytes)} · 可用系统内存 ${mib(memory.available_bytes)}\n处理预算 ${budget(b.processing_mb)}（任务 ${budget(b.job_mb)} / 检查 ${budget(b.inspection_mb)}）\n原值缓存 ${mib(cache.frame_cache_bytes)} / ${mib(cache.frame_cache_limit_bytes)}（${count}）\n${processes.map(p => `${p.name} ${mib(p.resident_bytes)}`).join(" · ") || "进程明细读取中…"}`;
    if (readout.textContent !== text) readout.textContent = text;
    host.querySelector("[data-device-cache-summary]").textContent = `原值缓存 ${mib(cache.frame_cache_bytes)} · ${count}`;
  }

  function schedule(delay = 2000) {
    clearTimeout(timer);
    if (active()) timer = setTimeout(poll, delay);
  }
  async function poll() {
    if (!active()) return;
    if (busy) { schedule(200); return; }
    busy = true;
    const generation = epoch;
    try {
      const data = await api("resources/telemetry");
      if (generation === epoch && active()) {
        lastData = data; record(data); render(data); host.classList.remove("stale");
        state.textContent = "实时";
        state.title = `最近采样 ${new Date(data.sampled_at * 1000).toLocaleTimeString()}；GPU 利用率为整卡读数`;
      }
    } catch (error) {
      if (generation === epoch && active()) {
        host.classList.add("stale"); gap(); state.textContent = "更新失败"; state.title = `${error.message}；保留的曲线及数值为上次读数`;
      }
    } finally {
      busy = false;
      if (generation === epoch) schedule();
    }
  }
  function sync(force = false) {
    const running = active();
    if (!force && running === wasActive) return;
    wasActive = running; clearTimeout(timer); epoch++;
    if (running) { state.textContent = "读取中…"; poll(); }
    else { gap(); state.textContent = paused ? "已暂停" : host.classList.contains("is-collapsed") ? "已折叠" : "已挂起"; }
  }
  gpuSelect.onchange = () => { selectedGpu = gpuSelect.value; if (lastData) render(lastData); };
  pause.onclick = () => {
    paused = !paused; pause.innerHTML = svgIcon(paused ? "play" : "pause");
    const label = paused ? "继续设备监测" : "暂停设备监测"; pause.setAttribute("aria-label", label); pause.title = label;
    sync();
  };
  document.addEventListener("device-monitor-layout", () => sync());
  document.addEventListener("visibilitychange", () => sync());
  window.addEventListener("pagehide", () => { clearTimeout(timer); epoch++; wasActive = false; gap(); });
  window.addEventListener("pageshow", () => sync());
  render({sampled_at: Date.now() / 1000, cpu: {}, memory: {}, gpu: {cards: []}});
  sync();
  return {refresh: () => sync(true)};
}
