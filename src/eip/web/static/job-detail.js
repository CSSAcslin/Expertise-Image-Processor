import { api } from "./api.js";
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const statuses={queued:"排队",running:"运行中",cancelling:"取消中",cancelled:"已取消",failed:"失败",succeeded:"完成",completed_with_errors:"部分失败"};
const taskNames={statistics:"全源统计",trial:"当前中心系数图",curve:"τ 扫描曲线",run:"空间系数图"};
const rect=r=>r?`(${r.x}, ${r.y}) · ${r.width} × ${r.height}`:"全幅";
const units={T:"样本",W:"窗长",Q:"延迟点",N:"记录"};
function executionHTML(job){
  const parts=(job.run?.results||[]).filter(r=>r.modules?.execution?.optimization_enabled!=null);
  if(!parts.length)return "";
  const row=(k,v)=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`;
  const number=v=>Number(v||0).toLocaleString();
  return `<details><summary>执行优化与数据范围</summary>${parts.map((r,i)=>{
    const e=r.modules.execution,s=e.spatial_execution;
    let rows=row("执行策略",e.optimization_enabled?"按模块依赖规划范围与批次":"兼容逐样本执行");
    if(e.full_frame_cpu_policy)rows+=row("全幅CPU策略","逐帧交接，避免额外批量复制");
    if(s){
      rows+=row("物理批次",`${number(s.blocks)} 批 · ${number(s.samples)} 个样本请求`)+row("范围路径读取 / 块内复用",`${number(s.source_reads)} / ${number(s.source_reuses)}`);
      const stages=e.physical_plan?.logical_stages||[];
      rows+=row("实际输出元素",Object.entries(s.region_elements||{}).map(([stage,n])=>`${stages.find(x=>String(x.output_stage)===stage)?.algorithm||"阶段 "+stage} · ${number(n)}`).join("\n")||"未使用局部执行");
      rows+=row("缓冲区交接",`${number(s.ownership_transfers)} 次直接交接 · ${number(s.compact_copies)} 次压紧复制`);
      if(Object.keys(s.full_barriers||{}).length)rows+=row("完整输入边界",Object.entries(s.full_barriers).map(([reason,n])=>`${reason} · ${number(n)} 次`).join("\n"));
    }
    for(const [label,key] of [["GPU连续输出","gpu_output_blocks"],["CPU连续输出","cpu_output_blocks"],["τ批量输出","lag_output_blocks"]]){
      const records=Object.values(e[key]||{}).filter(x=>x.output_blocks);
      if(records.length)rows+=row(label,records.map(x=>`${number(x.output_samples)} 个结果 · ${number(x.output_blocks)} 批`).join("\n"));
    }
    const lag=Object.values(e.lag_output_blocks||{}).filter(x=>x.output_blocks);
    if(lag.length)rows+=row("τ关联配对",lag.map(x=>`${number(x.pair_updates)} 对 · ${String(x.implementation).startsWith("cupy")?number(x.kernel_calls)+" 次GPU发射":"CPU保序累计"}`).join("\n"));
    const storage=e.main_storage;
    if(storage?.range_requests)rows+=row("连续信号块",`${number(storage.range_requests)} 次请求 · ${number(storage.range_samples)} 个样本`);
    const refs=r.modules.reference_execution;
    if(refs?.lag_time_summaries)rows+=row("τ来源时间",`${number(refs.lag_time_summaries)} 个窗口 · ${number(refs.lag_time_source_queries)} 次样本时间读取`);
    return `<p>${esc(r.source?.name||r.source?.path||"来源 "+(i+1))}</p><dl class="job-detail-grid">${rows}</dl>`;
  }).join("")}<p class="hint">逻辑模块顺序、精度与来源保留。实际元素数表示模块输出的物理工作量；跨设备边界及完整输入依赖仍按实例设置执行。</p></details>`;
}
function gpuTimingRows(p,row,duration){
  const t=p.gpu_timing;
  if(!t)return p.schema_version>=2?row("GPU 设备区间","无已覆盖的 GPU 内核调用"):row("GPU 内核耗时",p.gpu_execution_seconds==null?"未采集（旧记录）":duration(p.gpu_execution_seconds));
  const timed=Number(t.timed_calls)||0,warmup=Number(t.warmup_excluded_calls)||0,capacity=Number(t.capacity_excluded_calls)||0,pending=Number(t.pending_calls)||0,signature=Number(t.signature_excluded_calls)||0,nested=Number(t.nested_excluded_calls)||0,sampling=Number(t.sampling_excluded_calls)||0;
  const measured=timed>0&&Number.isFinite(p.gpu_device_seconds);
  const state=measured?duration(p.gpu_device_seconds)+(t.sample_every>1?"（抽样累计）":""):t.error?"采集异常":warmup?"仅首次调用，已排除编译预热":"无已覆盖的 GPU 内核调用";
  const coverage=[`g² 融合内核 · 已计时 ${timed} 次`,sampling?`未抽取 ${sampling} 次`:null,warmup?`首调排除 ${warmup} 次`:null,capacity?`采集容量排除 ${capacity} 次`:null,signature?`签名容量排除 ${signature} 次`:null,nested?`嵌套排除 ${nested} 次`:null,pending?`待完成 ${pending} 次`:null,t.status==="partial"||t.status==="error"?"部分采集":null,t.error?"采集异常，详见运行记录":null].filter(Boolean).join("；");
  return row("GPU 设备区间",state)+row("GPU 采集范围",coverage)+(Number.isFinite(p.gpu_execution_seconds)?row("GPU 纯内核总耗时",duration(p.gpu_execution_seconds)):"");
}
export function progressUnit(job) {
  if(job.progress_unit)return job.progress_unit;
  if(job.engine!=="sequence")return "帧";
  const axes=new Set((job.run?.results||[]).map(r=>r.progress_axis||r.contract?.sequence_axis).filter(Boolean));
  return axes.size===1?(units[[...axes][0]]||"输出"):"输出";
}
function performanceHTML(job) {
  let records=(job.run?.results||[]).filter(r=>r.performance).map((r,i)=>({name:r.data_spec?.name||r.source?.path||`来源 ${i+1}`,p:r.performance,unit:r.progress_unit||units[r.contract?.sequence_axis]||"输出"}));
  if(!records.length&&job.performance)records=[{name:job.progress_name||job.source_name||"当前来源",p:job.performance,unit:progressUnit(job)}];
  if(!records.length)return job.engine==="sequence"?"<details><summary>运行耗时</summary><p class=\"hint\">此记录尚无分段计时。旧运行不会补算耗时。</p></details>":"";
  const number=v=>Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:2}):"—";
  const duration=v=>!Number.isFinite(v)?"—":v>0&&v<.1?(v<.00001?"<0.01 毫秒":`${number(v*1000)} 毫秒`):`${number(v)} 秒`;
  const row=(k,v)=>`<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`;
  return `<details><summary>运行耗时与吞吐</summary>${records.map(({name,p,unit})=>`<p>${esc(name)}</p><dl class="job-detail-grid">${row("来源总耗时",duration(p.elapsed_seconds))}${row("输出吞吐",`${number(p.output_per_second)} ${unit}/秒 · ${p.output_samples??0} ${unit}`)}${row("输入读取吞吐",`${number(p.input_reads_per_second)} 样本读取/秒 · ${p.input_read_samples??0} 次`)}${Object.entries(p.host_seconds||{}).filter(([,v])=>v>0).sort((a,b)=>b[1]-a[1]).map(([k,v])=>row(p.labels?.[k]||k,duration(v))).join("")}${gpuTimingRows(p,row,duration)}</dl>`).join("")}<p class="hint">主机分段互斥，包含读取、传输和同步等待。GPU 设备区间使用 CUDA 事件抽样，仅累计已覆盖的 g² 融合内核调用，不外推总量；排除每种签名的首次调用，可能包含设备调度间隙，不代表全部模块的纯内核总耗时，也不与主机分段相加。GPU 调度时间不能视为内核时间。输入读取次数包含视频建索引、重复读取及缓存命中，并非唯一原始帧数；字节为解码后数组大小。吞吐按来源总耗时计算，包含准备与输出，不含排队和计算进程启动。</p></details>`;
}
export function detailHTML(job) {
  const run=job.run||{},c=run.config||job.parameters||run.spec||job.payload||{},p=c.photometry||{},g=c.geometry;
  const row=(k,v)=>`<dt>${esc(k)}</dt><dd>${esc(v??"未知")}</dd>`;
  let rows=row("任务类型",job.engine==="sequence"?(job.mode==="trial"?"模块流程试算":"模块流程处理"):job.mode==="split"?taskNames[job.task]:job.mode==="trial"?"当前帧试算":"通用处理")+row("任务标识",job.id)+row("状态",statuses[job.status]||job.status)+row("提交时间",new Date(job.created*1000).toLocaleString())+row("完成时间",run.completed_at||run.finished_at||"尚未完成")+row("原始来源",run.source?.path||(job.source_specs||c.sources||[]).map(s=>s.path).join("\n")||job.source_name||job.name)+row("进度",`${job.completed} / ${job.total??"未知"} ${progressUnit(job)}`);
  if(job.mode==="split") {
    const lags=run.lags||c.lag_list;
    rows+=row("平均窗口",`${c.window} 帧`)+row("时间步长",`${c.stride} 帧`)+row("延迟",lags?lags.join(", ")+" 帧":`${c.lag_min}…${c.lag_max}，步长 ${c.lag_step} 帧`)+row("中心帧",c.center==null?"自动滑动":Number(c.center)+1)+row("分析范围",`帧 ${(c.start??0)+1} 至 ${c.end??"来源末尾"}（界面含末帧）`)+row("分析 ROI",rect(c.roi))+row("点",c.points?.map(x=>`(${x.join(", ")})`).join("；")||"无")+row("ROI 估计",c.roi_estimator==="intensity_sum"?"区域总强度 g²":"逐像素 g² 空间平均")+row("强度曲线",p.lut_path?`自定义 LUT：${p.lut_path}`:`${p.curve||"identity"} · 固定输入尺度 ${p.input_max??"—"} · γ=${p.gamma??1}`)+row("偏置 / 饱和",`左 ${p.black_left}，右 ${p.black_right}；饱和阈值 ${p.saturation_code??"未排除"}`)+row("固定归一尺度",p.normalization)+row("全源统计",run.statistics?`${run.statistics.frames} 帧 · L / R 尺度 ${run.statistics.scales.join(" / ")}`:"运行时统计全部输入帧")+row("最少有效样本",c.min_samples)+row("最小局部均值",c.min_mean)+row("计算设备",run.backend||c.backend)+row("保存精度",c.output_dtype)+row("工作内存",`${c.memory_mb} MiB`)+row("数值含义",run.numeric_semantics||p.input_semantics);
    if(g)rows+=row("左右有效区",`左 ${rect(g.left)}；右 ${rect(g.right)}`)+row("固定配准",`dx=${g.dx}, dy=${g.dy} · ${g.interpolation} · 镜像 X=${g.mirror_x}, Y=${g.mirror_y}`);
    if(run.array_shape)rows+=row("结果形状",run.array_shape.join(" × ")+"（时间 × 延迟 × 高 × 宽）");
    if(run.curve_count!=null)rows+=row("曲线数量",run.curve_count);
    if(run.result_semantics)rows+=row("结果定义",run.result_semantics==="processed_g2"?"模块后处理的 g²（完整顺序见下方）":"标准 g²");
  } else if(job.engine==="sequence") {
    if(c.execution_scope?.kind==="prefix")rows+=row("运行终点",`${c.steps?.at(-1)?.name||c.steps?.at(-1)?.algorithm||c.execution_scope.stop_after} · 包含此模块，后序模块未执行`);
    rows+=row("流程契约",`${run.contract_version||3} · 轴、形状、含义逐步校验`)+row("范围 / ROI", "由流程中的时间裁剪和空间 ROI 模块确定")+row("W / τ / 步长单位","关联模块直接上游的编辑后样本；真实时间差见样本映射 / CSV")+row("输出形状",(run.results||[]).map((r,i)=>`${i+1}. ${r.contract?.label||r.error||r.status} · ${r.contract?.dtype||""} · ${r.contract?.meaning||""}`).join("\n")||"待执行确定");
  } else {
    rows+=row("处理范围",JSON.stringify(c.selection||run.spec?.selection||{}))+row("算法步骤",(c.steps||run.spec?.steps||[]).filter(s=>s.enabled!==false).map((s,i)=>`${i+1}. ${s.algorithm} ${JSON.stringify(s.parameters)}`).join("\n")||"直接导出");
  }
  rows+=row("输出目录",job.directory||"尚未发布")+row("来源修订",run.source_revision||"见运行记录");
  if(c.resources)rows+=row("统一资源快照",`内存 ${c.resources.memory_mb} MiB · 预览 ${c.resources.preview_mb} MiB · 计算缓存 ${c.resources.cache_mb} MiB · 显存 ${c.resources.gpu_mb} MiB · 默认 ${c.resources.backend}`);
  for(const r of run.results||[])if(r.modules?.devices)rows+=row("实际计算设备",Object.entries(r.modules.devices).map(([id,d])=>`${id.slice(0,8)} · ${d.actual.toUpperCase()}${d.reason?" · "+d.reason:""}`).join("\n"));
  const stages=run.results?.find(r=>r.status==="succeeded")?.modules?.stages||[];
  const flow=(c.workflow||c.steps||[]).map((n,i)=>{const stage=stages.find(s=>s.id===n.id);return `<li>${esc(i+1+". "+n.algorithm)} · 实例 ${esc(n.id||"旧记录")} · v${esc(n.version||"1.0")}${n.enabled===false?"（停用）":""}${stage?`<p>${esc(stage.input.label)} → ${esc(stage.output.label)}<br>${esc(stage.output.meaning)}</p>`:""}<pre>${esc(JSON.stringify({parameters:n.parameters,execution:n.execution||{}},null,2))}</pre></li>`;}).join("");
  return `<div class="job-rename"><input id="job-rename-input" maxlength="160" value="${esc(job.name)}" aria-label="任务与结果名称"/><button id="job-rename-save" class="quiet">保存名称</button></div><dl class="job-detail-grid">${rows}</dl>${job.summary?`<p class="hint">${esc(job.summary)}</p>`:""}${job.error?`<pre>${esc(job.error)}</pre>`:""}${performanceHTML(job)}${executionHTML(job)}${flow?`<details><summary>模块顺序与各实例参数</summary><ol class="job-module-list">${flow}</ol></details>`:""}<details><summary>完整运行参数</summary><pre>${esc(JSON.stringify(c,null,2))}</pre></details>${job.logs?.length?`<details open><summary>任务日志</summary><pre>${esc(job.logs.join("\n"))}</pre></details>`:""}`;
}
export function bindRename(job, refreshed, report) {
  document.getElementById("job-rename-save").onclick=async()=>{
    try {await api(`jobs/${job.id}/rename`,{name:document.getElementById("job-rename-input").value});await refreshed();report("任务与结果名称已保存");}
    catch(error){report(error.message,true);}
  };
}
