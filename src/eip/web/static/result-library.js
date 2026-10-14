import { api, download } from "./api.js";
import { svgIcon } from "./icons.js";

const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const statuses = {succeeded:"已完成", completed_with_errors:"部分失败", failed:"失败", cancelled:"已取消"};
const ended = job => Object.hasOwn(statuses, job.status);
const trial = job => job.mode === "trial" || job.task === "trial";
const viewable = job => ["succeeded", "completed_with_errors"].includes(job.status);
const reusable = job => viewable(job) && !trial(job) && (!job.task || job.task === "run");
const bytes = value => value >= 1024**3 ? (value / 1024**3).toFixed(2) + " GiB" : (value / 1024**2).toFixed(1) + " MiB";
const iconButton = (action, icon, label, disabled = false) => `<button type="button" class="icon-button" data-result-action="${action}" title="${label}" aria-label="${label}" ${disabled ? "disabled" : ""}>${svgIcon(icon)}</button>`;

export function setupResultLibrary(bridge) {
  let selected = null, generation = 0, deletion = null, filter = "all", listKey = "", exporting = false;
  const guard = fn => async (...args) => { try { return await fn(...args); } catch (error) { bridge.report(error.message, true); } };
  const jobs = () => {
    const saved = bridge.jobs().filter(ended), known = new Set(saved.map(job => job.id));
    const references = bridge.sources().filter(source => source.role === "result" && !known.has(source.run_id));
    return [...saved, ...references.map(source => ({id:`reference:${source.id}`, reference_id:source.id, run_id:source.run_id, name:source.display_name || source.meta.name, source_name:source.meta.name, mode:"run", status:"succeeded", created:0, thumbnail:source.thumbnail, summary:source.meta.contract?.label || `${source.meta.shape.join(" × ")} · ${source.meta.dtype}`, source_path:source.source.path}))];
  };
  const output = () => {
    const [position, side] = $("managed-output").value.split("|");
    return {source:Number(position) || 0, side_output:side || null, lag_index:Number($("managed-lag").value) || 0, target:"input"};
  };
  const item = () => {
    const choice = output(), parent = selected?.run?.results?.[choice.source];
    return choice.side_output ? parent?.side_outputs?.find(side => side.id === choice.side_output) : parent;
  };
  function renderList() {
    const all = jobs(), query = $("result-search").value.trim().toLocaleLowerCase();
    const rows = all.filter(job => (filter === "all" || (filter === "trial") === trial(job)) &&
      `${job.name} ${job.source_name || ""} ${job.id}`.toLocaleLowerCase().includes(query));
    $("result-count").textContent = all.length;
    const errors = bridge.recoveryErrors?.() || [];
    $("result-recovery-note").hidden = !errors.length;
    $("result-recovery-note").textContent = `${errors.length} 项未恢复`;
    $("result-recovery-note").title = errors.join("\n");
    $("result-manager-empty").hidden = !!rows.length;
    $("result-manager-empty").textContent = all.length ? "无匹配结果" : "运行流程后在这里查看结果";
    const thumbnails = new Map(bridge.sources().filter(source => source.role === "result").map(source => [source.run_id, source.thumbnail]));
    const key = JSON.stringify([selected?.id, rows.map(job => [job.id, job.name, job.status, thumbnails.get(job.id)?.length])]);
    if (key === listKey) return;
    listKey = key;
    const list = $("result-list"), scroll = list.scrollTop;
    const focusId = document.activeElement?.closest("[data-result-id]")?.dataset.resultId;
    const focusAction = document.activeElement?.dataset.resultAction;
    list.innerHTML = rows.map(job => {
      const thumb = thumbnails.get(job.id) || job.thumbnail, date = new Date(job.created * 1000);
      const time = job.created ? date.toLocaleDateString(undefined, {month:"2-digit", day:"2-digit"}) + " " + date.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"}) : "来源引用";
      const active = selected?.id === job.id;
      return `<article class="result-entry ${active ? "selected" : ""}" data-result-id="${esc(job.id)}" data-manage-result="${esc(job.id)}" ${job.engine === "sequence" ? `data-sequence-result="${esc(job.id)}"` : job.mode === "split" ? `data-split-result="${esc(job.id)}"` : ""} tabindex="0" role="button" aria-pressed="${active}" aria-label="预览 ${esc(job.name)}">
        <div class="result-thumbnail">${thumb ? `<img src="${thumb}" alt=""/>` : svgIcon(viewable(job) ? "image" : "warning")}</div>
        <div class="result-entry-copy"><strong title="${esc(job.name)}">${esc(job.name)}</strong><small><span class="result-status-dot ${job.status}" title="${statuses[job.status]}"></span>${trial(job) ? "试算" : "正式"}<span> · ${time}</span></small></div>
        <div class="result-entry-actions">${iconButton("promote", "input", "作为新输入", !reusable(job))}${iconButton("export", "download", "导出运行包", !job.directory)}</div>
      </article>`;
    }).join("");
    list.scrollTop = scroll;
    for (const row of list.querySelectorAll("[data-result-id]")) {
      row.onclick = guard(event => {
        const action = event.target.closest("[data-result-action]")?.dataset.resultAction;
        return action ? quickAction(row.dataset.resultId, action) : select(row.dataset.resultId);
      });
      row.onkeydown = guard(event => {
        if (event.target !== row) return;
        if (["Enter", " "].includes(event.key)) { event.preventDefault(); event.stopPropagation(); return select(row.dataset.resultId); }
        if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
          event.preventDefault();
          event.stopPropagation();
          const entries = [...list.querySelectorAll("[data-result-id]")], position = entries.indexOf(row);
          const next = event.key === "Home" ? entries[0] : event.key === "End" ? entries.at(-1) : entries[Math.max(0, Math.min(entries.length - 1, position + (event.key === "ArrowDown" ? 1 : -1)))];
          next.focus(); return select(next.dataset.resultId);
        }
      });
    }
    if (focusId) {
      const row = [...list.children].find(element => element.dataset.resultId === focusId);
      (focusAction ? row?.querySelector(`[data-result-action="${focusAction}"]`) : row)?.focus({preventScroll:true});
    }
  }
  function update() {
    if (selected && !jobs().some(job => job.id === selected.id)) {
      selected = null; generation++; $("result-manager-content").hidden = true;
    }
    const latest = selected && jobs().find(job => job.id === selected.id);
    if (latest && $("managed-name").readOnly && selected.name !== latest.name) {
      selected.name = latest.name; $("managed-name").value = latest.name;
    }
    renderList();
  }
  function paintSelection() {
    const result = item(), contract = result?.contract;
    $("managed-status").textContent = trial(selected) ? "试算" : statuses[selected.status];
    $("managed-status").className = `result-status ${selected.status}`;
    $("managed-description").textContent = contract ? `${contract.full_axes || contract.axes || "标量"} · ${contract.shape?.join(" × ") || "标量"} · ${contract.length} 样本 · ${contract.dtype}` : selected.summary || selected.error || "无完整输出";
    $("managed-view").disabled = !viewable(selected);
    $("managed-view").title = viewable(selected) ? "预览" : "无完整输出，可查看参数与来源";
    $("managed-edit").disabled = !!selected.reference_id;
    $("managed-edit").title = selected.reference_id ? "重新打开完整运行记录后可改名" : "重命名";
    $("managed-promote").disabled = !reusable(selected);
    $("managed-promote").title = reusable(selected) ? "作为新输入" : trial(selected) ? "试算不能作为正式输入" : "此结果没有可复用的完整数据";
    $("managed-export").disabled = !selected.directory || exporting;
    $("managed-export").title = exporting ? "正在准备运行包" : selected.directory ? "导出运行包" : "此结果没有可导出的磁盘记录";
    const available = new Set(result?.artifacts || []);
    const files = {array:"values.npy", csv:"measurements.csv", references:"references.jsonl", image:"first.tif", png:"curve.png", svg:"curve.svg"};
    for (const button of $("managed-formats").querySelectorAll("[data-result-export]")) {
      const format = button.dataset.resultExport;
      button.disabled = !selected.directory || (format !== "manifest" && (selected.engine !== "sequence" || !available.has(files[format])));
      button.title = button.disabled ? "此输出未生成该格式" : `导出 ${button.textContent.trim()}`;
    }
    $("managed-path").textContent = selected.directory || selected.source_path || "无磁盘输出";
    $("managed-source").textContent = selected.source_specs?.map(source => source.path).join("\n") || selected.source_name || "未知";
    $("managed-error").hidden = !selected.error && viewable(selected);
    $("managed-error").textContent = selected.error || selected.run?.results?.find(result => result.error)?.error || statuses[selected.status];
  }
  async function preview() {
    const job = selected, choice = output();
    if (!job || !viewable(job)) return;
    const current = bridge.currentOutput?.();
    if (bridge.currentResult?.() === job.id && current?.source === choice.source && current?.side_output === choice.side_output && (job.mode !== "split" || current?.lag_index === choice.lag_index)) { bridge.reveal?.(); return; }
    await bridge.view(job, choice);
  }
  async function select(id, {show = true} = {}) {
    if (selected?.id === id) { if (show) await preview(); return true; }
    const ticket = ++generation;
    $("result-list").setAttribute("aria-busy", "true");
    try {
      const reference = jobs().find(job => job.id === id && job.reference_id);
      const job = reference || await api(`jobs/${id}`);
      if (ticket !== generation) return false;
      selected = job;
      $("result-manager-content").hidden = false;
      $("managed-name").value = job.name; finishRename();
      $("managed-metadata").open = false;
      const results = job.run?.results || [];
      $("managed-output").innerHTML = results.flatMap((result, position) => result.status !== "succeeded" ? [] : [
        `<option value="${position}">${results.length > 1 ? `来源 ${position + 1} · ` : ""}主输出</option>`,
        ...(result.side_outputs || []).filter(side => side.status === "succeeded").map(side => `<option value="${position}|${esc(side.id)}">${results.length > 1 ? `来源 ${position + 1} · ` : ""}${esc(side.name || side.port)}</option>`)
      ]).join("");
      $("managed-output").parentElement.hidden = job.engine !== "sequence" || !$("managed-output").options.length;
      const current = bridge.currentResult?.() === id ? bridge.currentOutput?.() : null;
      const preferred = current ? `${current.source || 0}${current.side_output ? "|" + current.side_output : ""}` : null;
      if (preferred && [...$("managed-output").options].some(option => option.value === preferred)) $("managed-output").value = preferred;
      $("managed-lag").innerHTML = (job.run?.lags || [0]).map((lag, index) => `<option value="${index}">τ = ${lag}</option>`).join("");
      $("managed-lag").value = current?.lag_index || 0;
      $("managed-lag").parentElement.hidden = job.mode !== "split" || job.task !== "run";
      paintSelection(); renderList();
      if (show) await preview();
      return ticket === generation;
    } finally { if (ticket === generation) $("result-list").removeAttribute("aria-busy"); }
  }
  async function open(id = null) {
    if (id) {
      $("result-search").value = ""; filter = "all";
      document.querySelectorAll("[data-result-filter]").forEach(button => { const active = button.dataset.resultFilter === "all"; button.classList.toggle("selected", active); button.setAttribute("aria-pressed", String(active)); });
    }
    bridge.openLibrary("results"); update();
    const rows = jobs();
    if (rows.length) await select(rows.some(job => job.id === id) ? id : selected?.id || rows.find(job => job.id === bridge.currentResult?.())?.id || rows[0].id);
  }
  function finishRename() {
    $("managed-name").readOnly = true;
    $("managed-edit").hidden = false; $("managed-rename").hidden = true;
  }
  async function rename() {
    const id = selected.id, name = $("managed-name").value.trim();
    if (!name) { $("managed-name").focus(); return; }
    await api(`jobs/${id}/rename`, {name});
    if (selected?.id === id) { selected.name = name; finishRename(); }
    await bridge.refresh(); bridge.renamed?.(id, name); renderList();
    bridge.report("名称已保存");
  }
  async function promote() {
    const job = selected, choice = output();
    if (!job || !reusable(job)) return;
    let opened;
    if (job.reference_id) opened = await api(`sources/${job.reference_id}/promote`, {});
    else if (job.engine === "sequence") opened = await api(`jobs/${job.id}/sequence/promote`, choice);
    else if (job.mode === "split") opened = await api(`split/jobs/${job.id}/promote`, {lag_index:choice.lag_index});
    else {
      const source = bridge.sources().find(source => source.role === "result" && source.run_id === job.id);
      if (!source) throw Error("请通过项目重新打开该结果");
      opened = await api(`sources/${source.id}/promote`, {});
    }
    await bridge.promoted(opened); bridge.openLibrary("sources"); bridge.report("已作为新输入");
  }
  async function exportPackage() {
    const id = selected?.id;
    if (!id || exporting) return;
    exporting = true; paintSelection();
    $("managed-export").classList.add("is-loading"); $("managed-export").setAttribute("aria-busy", "true");
    try {
      const data = await api(`results/${id}/package`, {}), link = document.createElement("a");
      link.href = data.url; link.download = data.filename; document.body.append(link); link.click(); link.remove();
      bridge.report("运行包开始下载");
    } finally {
      exporting = false; $("managed-export").classList.remove("is-loading"); $("managed-export").removeAttribute("aria-busy");
      if (selected) paintSelection();
    }
  }
  async function quickAction(id, action) {
    if (!await select(id, {show:false})) return;
    return action === "promote" ? promote() : exportPackage();
  }
  function deleteMode() {
    const files = $("result-delete-files").checked;
    $("result-delete-target").open = files;
    $("result-delete-warning").hidden = !files;
    $("result-delete-confirm").textContent = files ? "永久删除" : "移除";
    $("result-delete-confirm").classList.toggle("danger", files);
  }
  async function deletePreview() {
    deletion = selected.reference_id ? {id:selected.id, source_id:selected.reference_id, name:selected.name, path:selected.source_path, file_count:0, size_bytes:0, dependencies:[], can_delete_files:false} : await api(`results/${selected.id}/delete-preview`);
    $("result-delete-name").textContent = deletion.name;
    $("result-delete-description").textContent = deletion.source_id ? "仅移除来源引用，保留文件" : `${deletion.file_count} 个文件 · ${bytes(deletion.size_bytes)}`;
    $("result-delete-path").textContent = deletion.path || "无磁盘文件";
    $("result-delete-dependencies").hidden = !deletion.dependencies.length && !deletion.error;
    $("result-delete-dependencies").textContent = deletion.error || `仍被引用：${deletion.dependencies.map(item => item.name).join("、")}`;
    $("result-delete-files").disabled = !deletion.can_delete_files;
    $("result-delete-reference").checked = true; deleteMode();
    $("result-delete-dialog").showModal();
  }
  $("manage-results").onclick = guard(() => open());
  $("result-manager-shortcut").onclick = guard(() => open(bridge.currentResult?.()));
  $("result-search").oninput = renderList;
  $("result-search").onkeydown = event => { if (event.key === "Escape") { event.preventDefault(); $("result-search").value = ""; renderList(); } };
  document.querySelectorAll("[data-result-filter]").forEach(button => button.onclick = () => {
    filter = button.dataset.resultFilter;
    document.querySelectorAll("[data-result-filter]").forEach(option => { option.classList.toggle("selected", option === button); option.setAttribute("aria-pressed", String(option === button)); });
    renderList();
  });
  $("managed-output").onchange = $("managed-lag").onchange = guard(async () => { paintSelection(); await preview(); });
  $("managed-view").onclick = guard(() => preview());
  $("managed-promote").onclick = guard(promote);
  $("managed-export").onclick = guard(exportPackage);
  $("managed-delete").onclick = guard(deletePreview);
  $("managed-detail").onclick = guard(() => selected.reference_id ? bridge.sourceDetail(selected.reference_id) : bridge.detail(selected.id));
  $("managed-edit").onclick = () => { $("managed-name").readOnly = false; $("managed-edit").hidden = true; $("managed-rename").hidden = false; $("managed-name").focus(); $("managed-name").select(); };
  $("managed-name").ondblclick = () => $("managed-edit").click();
  $("managed-name").onkeydown = guard(async event => {
    if (event.key === "Enter" && !$("managed-name").readOnly) { event.preventDefault(); await rename(); }
    else if (event.key === "Escape") { $("managed-name").value = selected.name; finishRename(); }
  });
  $("managed-rename").onclick = guard(rename);
  $("managed-formats").querySelectorAll("[data-result-export]").forEach(button => button.onclick = guard(async () => {
    const choice = output(); $("managed-formats").open = false;
    await download(selected.id, button.dataset.resultExport, choice.source, choice.side_output);
  }));
  $("managed-formats").ontoggle = () => {
    if (!$("managed-formats").open) return;
    const menu = $("managed-formats").querySelector(".managed-export-popover"), rect = $("managed-formats").querySelector("summary").getBoundingClientRect();
    menu.style.left = Math.max(8, Math.min(innerWidth - menu.offsetWidth - 8, rect.left)) + "px";
    menu.style.top = Math.max(8, rect.top - menu.offsetHeight - 6) + "px";
  };
  document.addEventListener("click", event => { if (!event.target.closest("#managed-formats")) $("managed-formats").open = false; });
  $("result-delete-files").onchange = $("result-delete-reference").onchange = deleteMode;
  $("result-delete-confirm").onclick = guard(async () => {
    const id = deletion.id;
    if (deletion.source_id) {
      await bridge.removeReference(deletion.source_id); $("result-delete-dialog").close(); selected = null; generation++; $("result-manager-content").hidden = true; update(); return;
    }
    await api(`results/${id}/delete`, {mode:$("result-delete-files").checked ? "files" : "reference", confirmation:deletion.confirmation});
    $("result-delete-dialog").close();
    if (selected?.id === id) { selected = null; generation++; $("result-manager-content").hidden = true; }
    await bridge.removed(id); await bridge.refresh(); update(); bridge.report("结果已移除");
  });
  $("result-restore-open").onclick = () => $("result-restore-dialog").showModal();
  $("result-restore-pick").onclick = guard(() => bridge.pickResult());
  $("result-restore").onclick = guard(async () => {
    const data = await api("results/restore", {path:$("result-restore-path").value.trim()});
    $("result-restore-dialog").close(); await bridge.refresh(); await open(data.id); bridge.report("结果已打开");
  });
  $("result-restore-path").onkeydown = event => { if (event.key === "Enter") { event.preventDefault(); $("result-restore").click(); } };
  for (const id of ["result-delete-dialog", "result-restore-dialog"]) $(id).querySelectorAll("[data-close-dialog]").forEach(button => button.onclick = () => $(id).close());
  return {open:guard(open), update};
}
