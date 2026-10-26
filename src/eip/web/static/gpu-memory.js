import { api } from "./api.js";
import { svgIcon } from "./icons.js";

const escape = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const mib = value => value == null ? "未知" : Math.round(value / 1048576).toLocaleString();

export function setupGpuMemory(dialog, anchor) {
  const host = document.createElement("section");
  host.className = "gpu-memory-monitor";
  host.setAttribute("aria-label", "显存动态跟踪");
  host.innerHTML = `<div class="gpu-memory-heading"><h3>${svgIcon("activity")}显存</h3><span data-gpu-state role="status">读取中…</span><button data-gpu-pause class="icon-button" type="button" aria-label="暂停显存刷新" title="暂停显存刷新">${svgIcon("pause")}</button></div><div data-gpu-cards></div><div class="gpu-memory-footer"><span>近 2 分钟 · 每 2 秒</span><span data-gpu-budget title="已保存的统一显存预算，与实际占用分开统计"></span></div>`;
  anchor.after(host);
  const state = host.querySelector("[data-gpu-state]"), cards = host.querySelector("[data-gpu-cards]"), pause = host.querySelector("[data-gpu-pause]");
  const histories = new Map();
  let timer, epoch = 0, paused = false, busy = false, lastTime = null;
  const active = () => dialog.open && !document.hidden && !paused;
  function markGap(time) {
    for (const [id, history] of histories) {
      const last = history.at(-1);
      if (last && last.used != null) histories.set(id, [...history, {time: time ?? last.time, used: null, total: null}].slice(-61));
    }
  }

  function chart(card, now) {
    let history = histories.get(card.uuid) || [];
    if (history.at(-1)?.time > now) history = [];
    history = history.filter(point => point.time > now - 120);
    if (history.at(-1)?.time !== now) history.push({time: now, used: card.used_bytes, total: card.total_bytes});
    history = history.slice(-61);
    histories.set(card.uuid, history);
    while (histories.size > 64) histories.delete(histories.keys().next().value);
    let path = "", connected = false, last = null;
    for (const point of history) {
      if (point.used == null || !point.total) { connected = false; continue; }
      const x = Math.max(0, Math.min(300, 300 - (now - point.time) * 2.5));
      const y = 32 - Math.max(0, Math.min(1, point.used / point.total)) * 28;
      path += `${connected ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)} `;
      connected = true; last = {x, y};
    }
    return `<svg class="gpu-memory-chart" viewBox="0 0 300 36" preserveAspectRatio="none" role="img" aria-label="最近两分钟整卡显存占用比例；缺失读数不连接"><path class="gpu-memory-grid" d="M0 4H300M0 18H300M0 32H300"/><path class="gpu-memory-line" d="${path}"/>${last ? `<circle class="gpu-memory-point" cx="${last.x}" cy="${last.y}" r="2"/>` : ""}</svg>`;
  }

  function render(data) {
    host.classList.remove("stale");
    lastTime = data.sampled_at;
    const label = data.available ? "实时" : "不可用";
    if (state.textContent !== label) state.textContent = label;
    state.title = lastTime ? `最近采样 ${new Date(lastTime * 1000).toLocaleTimeString()}` : "";
    host.querySelector("[data-gpu-budget]").textContent = `已保存预算 ${data.budget_mb} MiB`;
    if (!data.available) {
      markGap(data.sampled_at);
      cards.innerHTML = `<p class="gpu-memory-empty">${escape(data.reason || "无法读取显存")}</p>`;
      return;
    }
    cards.innerHTML = data.cards.map(card => {
      const percentage = card.total_bytes && card.used_bytes != null ? card.used_bytes / card.total_bytes * 100 : null;
      const app = card.application_bytes == null ? "未知" : `${mib(card.application_bytes)} MiB`;
      return `<div class="gpu-memory-device ${percentage >= 90 ? "warning" : ""}"><div class="gpu-memory-values"><span class="gpu-memory-name" title="驱动编号 GPU ${card.index} · ${escape(card.name)}">GPU ${card.index} · ${escape(card.name)}</span><strong>${mib(card.used_bytes)} / ${mib(card.total_bytes)} MiB${percentage == null ? "" : ` · ${percentage.toFixed(0)}%`}</strong></div>${chart(card, data.sampled_at)}<div class="gpu-memory-metrics"><span>空闲 ${mib(card.free_bytes)} MiB</span><span title="${escape(card.application_reason || "本工作台及其读取、检查和处理进程的驱动上报占用；包含上下文等开销")}">本程序 ${app}${card.application_reason ? svgIcon("info") : ""}</span></div></div>`;
    }).join("");
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
      const data = await api("resources/gpu");
      if (generation === epoch && active()) render(data);
    } catch (error) {
      if (generation === epoch && active()) {
        host.classList.add("stale");
        markGap();
        state.textContent = "更新失败";
        state.title = `${error.message}${lastTime ? "；图中为上次读数" : ""}`;
      }
    } finally {
      busy = false;
      if (generation === epoch) schedule();
    }
  }
  function sync() {
    clearTimeout(timer); epoch++;
    if (active()) { state.textContent = "读取中…"; poll(); }
    else { markGap(); if (paused) state.textContent = "已暂停"; }
  }
  pause.onclick = () => {
    paused = !paused;
    pause.innerHTML = svgIcon(paused ? "play" : "pause");
    const label = paused ? "继续显存刷新" : "暂停显存刷新";
    pause.setAttribute("aria-label", label); pause.title = label;
    sync();
  };
  new MutationObserver(sync).observe(dialog, {attributes: true, attributeFilter: ["open"]});
  document.addEventListener("visibilitychange", sync);
  window.addEventListener("pagehide", () => { clearTimeout(timer); epoch++; });
  window.addEventListener("pageshow", sync);
  return { refresh: sync };
}
