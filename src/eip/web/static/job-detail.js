import { api } from "./api.js";
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const statuses={queued:"排队",running:"运行中",cancelling:"取消中",cancelled:"已取消",failed:"失败",succeeded:"完成",completed_with_errors:"部分失败"};
const taskNames={statistics:"全源统计",trial:"当前中心系数图",curve:"τ 扫描曲线",run:"空间系数图"};
const rect=r=>r?`(${r.x}, ${r.y}) · ${r.width} × ${r.height}`:"全幅";
export function detailHTML(job) {
  const run=job.run||{},c=run.config||job.parameters||run.spec||job.payload||{},p=c.photometry||{},g=c.geometry;
  const row=(k,v)=>`<dt>${esc(k)}</dt><dd>${esc(v??"未知")}</dd>`;
  let rows=row("任务类型",job.engine==="sequence"?(job.mode==="trial"?"模块流程试算":"模块流程处理"):job.mode==="split"?taskNames[job.task]:job.mode==="trial"?"当前帧试算":"通用处理")+row("任务标识",job.id)+row("状态",statuses[job.status]||job.status)+row("提交时间",new Date(job.created*1000).toLocaleString())+row("完成时间",run.completed_at||run.finished_at||"尚未完成")+row("原始来源",run.source?.path||(job.source_specs||c.sources||[]).map(s=>s.path).join("\n")||job.source_name||job.name)+row("进度",`${job.completed} / ${job.total??"未知"}`);
  if(job.mode==="split") {
    const lags=run.lags||c.lag_list;
    rows+=row("平均窗口",`${c.window} 帧`)+row("时间步长",`${c.stride} 帧`)+row("延迟",lags?lags.join(", ")+" 帧":`${c.lag_min}…${c.lag_max}，步长 ${c.lag_step} 帧`)+row("中心帧",c.center==null?"自动滑动":Number(c.center)+1)+row("分析范围",`帧 ${(c.start??0)+1} 至 ${c.end??"来源末尾"}（界面含末帧）`)+row("分析 ROI",rect(c.roi))+row("点",c.points?.map(x=>`(${x.join(", ")})`).join("；")||"无")+row("ROI 估计",c.roi_estimator==="intensity_sum"?"区域总强度 g²":"逐像素 g² 空间平均")+row("强度曲线",p.lut_path?`自定义 LUT：${p.lut_path}`:`${p.curve||"identity"} · 固定输入尺度 ${p.input_max??"—"} · γ=${p.gamma??1}`)+row("偏置 / 饱和",`左 ${p.black_left}，右 ${p.black_right}；饱和阈值 ${p.saturation_code??"未排除"}`)+row("固定归一尺度",p.normalization)+row("全源统计",run.statistics?`${run.statistics.frames} 帧 · L / R 尺度 ${run.statistics.scales.join(" / ")}`:"运行时统计全部输入帧")+row("最少有效样本",c.min_samples)+row("最小局部均值",c.min_mean)+row("计算设备",run.backend||c.backend)+row("保存精度",c.output_dtype)+row("工作内存",`${c.memory_mb} MiB`)+row("数值含义",run.numeric_semantics||p.input_semantics);
    if(g)rows+=row("左右有效区",`左 ${rect(g.left)}；右 ${rect(g.right)}`)+row("固定配准",`dx=${g.dx}, dy=${g.dy} · ${g.interpolation} · 镜像 X=${g.mirror_x}, Y=${g.mirror_y}`);
    if(run.array_shape)rows+=row("结果形状",run.array_shape.join(" × ")+"（时间 × 延迟 × 高 × 宽）");
    if(run.curve_count!=null)rows+=row("曲线数量",run.curve_count);
    if(run.result_semantics)rows+=row("结果定义",run.result_semantics==="processed_g2"?"模块后处理的 g²（完整顺序见下方）":"标准 g²");
  } else if(job.engine==="sequence") {
    rows+=row("流程契约",`${run.contract_version||3} · 轴、形状、含义逐步校验`)+row("范围 / ROI", "由流程中的时间裁剪和空间 ROI 模块确定")+row("W / τ / 步长单位","关联模块直接上游的编辑后样本；真实时间差见样本映射 / CSV")+row("输出形状",(run.results||[]).map((r,i)=>`${i+1}. ${r.contract?.label||r.error||r.status} · ${r.contract?.dtype||""} · ${r.contract?.meaning||""}`).join("\n")||"待执行确定");
  } else {
    rows+=row("处理范围",JSON.stringify(c.selection||run.spec?.selection||{}))+row("算法步骤",(c.steps||run.spec?.steps||[]).filter(s=>s.enabled!==false).map((s,i)=>`${i+1}. ${s.algorithm} ${JSON.stringify(s.parameters)}`).join("\n")||"直接导出");
  }
  rows+=row("输出目录",job.directory||"尚未发布")+row("来源修订",run.source_revision||"见运行记录");
  const stages=run.results?.find(r=>r.status==="succeeded")?.modules?.stages||[];
  const flow=(c.workflow||c.steps||[]).map((n,i)=>{const stage=stages.find(s=>s.id===n.id);return `<li>${esc(i+1+". "+n.algorithm)} · 实例 ${esc(n.id||"旧记录")} · v${esc(n.version||"1.0")}${n.enabled===false?"（停用）":""}${stage?`<p>${esc(stage.input.label)} → ${esc(stage.output.label)}<br>${esc(stage.output.meaning)}</p>`:""}<pre>${esc(JSON.stringify({parameters:n.parameters,execution:n.execution||{}},null,2))}</pre></li>`;}).join("");
  return `<div class="job-rename"><input id="job-rename-input" maxlength="160" value="${esc(job.name)}" aria-label="任务与结果名称"/><button id="job-rename-save" class="quiet">保存名称</button></div><dl class="job-detail-grid">${rows}</dl>${job.summary?`<p class="hint">${esc(job.summary)}</p>`:""}${job.error?`<pre>${esc(job.error)}</pre>`:""}${flow?`<details><summary>模块顺序与各实例参数</summary><ol class="job-module-list">${flow}</ol></details>`:""}<details><summary>完整运行参数</summary><pre>${esc(JSON.stringify(c,null,2))}</pre></details>${job.logs?.length?`<details open><summary>任务日志</summary><pre>${esc(job.logs.join("\n"))}</pre></details>`:""}`;
}
export function bindRename(job, refreshed, report) {
  document.getElementById("job-rename-save").onclick=async()=>{
    try {await api(`jobs/${job.id}/rename`,{name:document.getElementById("job-rename-input").value});await refreshed();report("任务与结果名称已保存");}
    catch(error){report(error.message,true);}
  };
}
