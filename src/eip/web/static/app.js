import { api, session, download } from "./api.js";
import { Viewer } from "./viewer.js";
import { FileBrowser } from "./file-browser.js";
import { validROI, moveROI } from "./geometry.js";
import { SplitController } from "./split.js";
import { ModuleEditor, moduleNode } from "./flow.js";
import { SequenceResults } from "./sequence.js";
import { WorkbenchLayout } from "./layout.js";
import { detailHTML, bindRename } from "./job-detail.js";
import { PreviewEngine } from "./preview.js";
import { setupWorkbench } from "./workbench.js";
import { svgIcon } from "./icons.js";

const $ = (id) => document.getElementById(id);
const escape = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (char) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        char
      ],
  );
const icons = { search:"M10 17a7 7 0 1 0 0-14 7 7 0 0 0 0 14Zm5-2 6 6", plugin:"M8 3v5H3v8h5v5h8v-5h5V8h-5V3Z", undo:"M9 5 3 11l6 6M3 11h11a6 6 0 0 1 6 6", redo:"m15 5 6 6-6 6M21 11H10a6 6 0 0 0-6 6", float:"M14 3h7v7M21 3l-9 9M11 3H3v18h18v-8", maximize:"M3 9V3h6M15 3h6v6M21 15v6h-6M9 21H3v-6", tap:"M3 5h18v4H3ZM12 9v10M8 15l4 4 4-4",
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
    .forEach((el) => (el.innerHTML = icons[el.dataset.icon]?icon(el.dataset.icon):svgIcon(el.dataset.icon)));
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
let resultOwner="general",generalROI=null;
let axisEditing=null,axisStage=false;
let inspectionGeneration=0,inspectionRequest=null,stageSnapshot=null;
function cancelInspection(){inspectionGeneration++;const id=inspectionRequest;inspectionRequest=null;$("inspection-state").hidden=true;if(id)api("modules/inspect/cancel",{inspection_id:id}).catch(()=>{});}
const preview=new PreviewEngine({read:(id,value)=>api(`sources/${id}/frame`,value),generation:()=>sourceGeneration,state:value=>{playing=value;$("play").innerHTML=icon(value?"pause":"play");$("play").setAttribute("aria-label",value?"暂停":"播放");}});
let selectionRequest = 0;
function frameRequest(id, value, generation = sourceGeneration) {
  return preview.request(id,value,generation);
}
function splitActive(){return false;}
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
  geometry: value=>workbench.commitGeometry(value,axisEditing),
  redraw,
  fit,
  roi: () => {
    if(axisEditing){commitModuleROI();return;}
    if(splitActive()){split.commitROI();return;}
    rememberROI();
    changed();
    updateROI();
  },
  roiPreview: () => {
    if(axisEditing){redraw();return;}
    if(splitActive()){split.draw();return;}
    updateROI();
    $("stale").hidden = !result || resultSignature === signature();
  },
  pixel: (snapshot, x, y, generic=false) => {
    if(snapshot.signal_values){$("pixel-info").textContent=`时间信号 · 无空间像素 · ${snapshot.signal_values.map(v=>v==null?"无效":Number(v).toPrecision(8)).join(" / ")}`;return;}
    if(splitActive()&&!generic){split.pixel(snapshot,x,y);return;}
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
const resultScene={...scene,roi:null};
const originalViewer = new Viewer($("input-canvas"), scene, callbacks),
  resultViewer = new Viewer($("result-canvas"), resultScene, {...callbacks,fit:()=>fit(resultViewer),roi:()=>{},roiPreview:()=>{},pixel:(snapshot,x,y)=>resultOwner==="split"?split.pixel(snapshot,x,y,true):callbacks.pixel(snapshot,x,y,true)});
const layout = new WorkbenchLayout(scheduleDraft);
const split = new SplitController({
  viewer:originalViewer,resultViewer,seek:seekFrame,stopPreview:stopPlay,
  resultShown:snapshot=>{
    sequenceResults.hide();
    resultOwner="split";result=snapshot;formalResult=null;resultSignature=null;
    $("split-result-controls").hidden=false;$("split-result-title").hidden=false;$("stale").hidden=true;$("result-label").textContent="处理结果";$("result-meta").textContent="";
    if(snapshot)resultDisplay={...snapshot.display};
    if($("viewers").dataset.mode==="input")setMode("compare");
    fillDisplayControls($("display-target").value==="result"?snapshot:input,$("display-target").value==="result"?resultDisplay:display);
  },
  context:()=>({source:active,input,output:$("output-dir").value,devices:state?.split_devices,modules:state?.modules}),
  changed:()=>{updateSummary();scheduleDraft();},report:toast,poll:pollJobs,
  openParameters:()=>layout.openParameters("split"),
  previewModule,drawModuleROI,submitModules,
  activate:()=>setAnalysisMode("split_correlation"),
  useModules:async(nodes,config)=>{steps=structuredClone(nodes);generalSequenceContext={sourceId:active?.id,config};renderSteps();await setAnalysisMode("general");toast("流程已复制到通用预设；模块参数与顺序保持一致");},
  promote:async(opened)=>{
    sources=sources.filter(s=>s.id!==opened.source.id);sources.push(opened.source);
    updateSources();await setAnalysisMode("general");await selectSource(opened.source.id);
  },
});
let generalSequenceContext=null;
const generalEditor=new ModuleEditor({nodes:()=>steps,catalog:()=>state?.modules,source:()=>active,host:()=>$("steps"),status:()=>$("module-status"),inspector:()=>$("general-module-fields"),title:()=>$("general-module-title"),instance:()=>$("general-module-instance"),preview:previewModule,drawROI:drawModuleROI,tools:moduleTools,inspect:(open=true)=>{if(open){layout.openParameters("general");$("general-module-inspector").scrollIntoView({block:"nearest"});}},selectionChanged:()=>{cancelInspection();updateSummary();scheduleDraft();},change:changed,afterRender:()=>{$("step-count").textContent=steps.length;},report:toast});
const sequenceResults=new SequenceResults({
  report:toast,changed:scheduleDraft,display:()=>resultDisplay,
  reset:()=>Object.assign(resultScene,{zoom:1,panX:0,panY:0,roi:null,annotation:null,tool:"pan"}),
  promoted:async(opened,target)=>{sources=(await api("state")).sources;updateSources();if(target==="input"){generalSequenceContext=null;generalEditor.replace([]);$("batch").checked=false;$("scope").value="all";await selectSource(opened.source.id);}},
  image:async snapshot=>{
    result=snapshot;resultOwner="sequence";formalResult=null;resultSignature=null;
    if(resultScene.width!==snapshot.shape[1]||resultScene.height!==snapshot.shape[0])Object.assign(resultScene,{zoom:1,panX:0,panY:0});
    resultScene.width=snapshot.shape[1];resultScene.height=snapshot.shape[0];await resultViewer.set(snapshot);
    if($("display-target").value==="result")fillDisplayControls(snapshot,resultDisplay);
  },
  signals:()=>{resultOwner="sequence";result=null;formalResult=null;resultViewer.clear();if($("display-target").value==="result")fillDisplayControls(null,resultDisplay);},
  shown:reveal=>{resultOwner="sequence";$("stale").hidden=true;$("welcome").hidden=true;if(reveal)setMode(!active||$("viewers").dataset.mode==="result"?"result":"compare");redraw();},
  promote:async(id,position)=>{sources=(await api("state")).sources;const job=await api(`jobs/${id}`),path=job.directory+"/"+job.run.results[position].output_directory+"/dataset.json";const entry=sources.find(s=>s.role==="result"&&s.run_id===id&&s.source.path.replaceAll("\\","/")===path.replaceAll("\\","/"));if(!entry)throw Error("该结果未能自动加入来源，请查看任务日志");formalResult=entry;await promoteResult();}
});

const workbench=setupWorkbench({editor:generalEditor,layout,source:()=>active,input:()=>input,viewer:originalViewer,resultViewer,scene,resultScene,notify:toast,save:scheduleDraft,inspect:(side,tool=null)=>previewModule(generalEditor,side,!!tool,tool==="point",tool),raw:async()=>{axisEditing=null;axisStage=false;scene.annotation=null;scene.tool="pan";$("axis-stage-note").hidden=true;$("module-preview-tools").hidden=true;if(input)await setInput(input);},currentJobs:()=>jobs,poll:pollJobs,submit});
function moduleTools(node,host,editor){workbench.moduleTools(node,host,editor);}
async function submitModules(nodes,mode="run",name=null,extra={}){
  if(!active||!input)return;
  const response=await api("jobs",{contract_version:3,mode,name,source_ids:extra.source_ids||[active.id],steps:structuredClone(nodes),stop_after:extra.stop_after||null,preview_dependencies:extra.dependencies||[],sample_index:extra.sample_index||0,output_dir:$("output-dir").value,config_revision:JSON.stringify(extra.stop_after?nodes.slice(0,nodes.findIndex(n=>n.id===extra.stop_after)+1):nodes)});
  toast(mode==="trial"?"试算已排队":"流程已排队");await pollJobs();return response;
}
async function previewModule(editor,side="input",drawing=false,point=false,tool=null){
  let generation;
  try{
    if(!active||!editor.selected)throw Error("请先选择输入来源和模块");
    if(!$("stage-sample").reportValidity())return;
    cancelInspection();generation=inspectionGeneration;
    stopPlay();const i=editor.nodes.indexOf(editor.selected),nodes=editor.nodes.slice(0,i+(side==="output"?1:0));
    let dependencies=[];if(nodes.some(n=>n.enabled&&n.algorithm==="calibration"&&n.parameters.black_left!==n.parameters.black_right)){const end=editor.nodes.findIndex((n,j)=>j>=nodes.length&&n.enabled&&n.algorithm==="alignment");if(end>=0)dependencies=structuredClone(editor.nodes.slice(nodes.length,end+1));}
    const request={editor,nodeId:editor.selected.id,side,drawing,point,tool:tool||(drawing?"roi":null),sourceId:active.id,sampleIndex:editor.previewSample||0,prefix:JSON.stringify(nodes)},key=JSON.stringify([active.id,active.meta.revision,nodes,dependencies,editor.previewSample||0,display]);
    let data=stageSnapshot?.key===key&&originalViewer.snapshot===stageSnapshot.data.frame?stageSnapshot.data:null;
    if(!data){inspectionRequest=crypto.randomUUID();$("inspection-state").hidden=false;$("inspection-message").textContent="正在读取此阶段 · 可继续预览原始数据";data=await api("modules/inspect",{inspection_id:inspectionRequest,source_id:active.id,steps:structuredClone(nodes),dependencies,sample_index:editor.previewSample||0,display});}
    if(generation!==inspectionGeneration||active?.id!==request.sourceId||editor.selected?.id!==request.nodeId||(editor.previewSample||0)!==request.sampleIndex)return;
    const current=editor.nodes.indexOf(editor.selected);if(JSON.stringify(editor.nodes.slice(0,current+(side==="output"?1:0)))!==request.prefix)return;
    stageSnapshot={key,data};await showStageResult(data,request);
  }catch(e){if(generation===undefined||generation===inspectionGeneration)toast(e.message,true);}
  finally{if(generation===inspectionGeneration){inspectionRequest=null;$("inspection-state").hidden=true;}}
}
async function drawModuleROI(editor=generalEditor,point=false){
  try{if(!active)throw Error("请先导入来源");const added=!["roi_crop","spatial_roi"].includes(editor.selected?.algorithm);if(added)editor.add(state.modules.roi_crop?"roi_crop":"spatial_roi");const fresh=added||!editor.selected.parameters.width&&!editor.selected.parameters.height;await previewModule(editor,"input",true,point,fresh?"roi-new":"roi");}catch(e){toast(e.message,true);}
}
function commitModuleROI(roi=scene.roi,remember=true){
  if(axisEditing?.tool&&!["roi","roi-new"].includes(axisEditing.tool)){workbench.commitRectangle(roi,axisEditing);return;}
  const bound=axisEditing,node=bound?.editor.nodes.find(n=>n.id===bound.nodeId);if(!node)return;
  const q=structuredClone(roi||{x:0,y:0,width:scene.width,height:scene.height});if(bound.point){q.width=q.height=1;node.parameters.mode="mean";}
  if(!validROI(q,scene.width,scene.height))throw Error("ROI 必须位于此模块输入的空间范围内");
  if(remember&&JSON.stringify(q)!==JSON.stringify(bound.committed)){bound.undo.push(structuredClone(bound.committed));bound.undo=bound.undo.slice(-32);bound.redo=[];}
  bound.committed=structuredClone(q);Object.assign(node.parameters,q);scene.roi=q;bound.editor.changed();updateROI();redraw();scheduleDraft();
}
async function showStageResult(data,request){
  if(active?.id!==request.sourceId)return;
  if(data.kind!=="image"){if(request.drawing)throw Error("此模块输入没有 Y/X 空间轴，请在前序空间数据阶段取点或绘制 ROI");await sequenceResults.inspect(data,`${request.editor.specs[request.editor.selected.algorithm].name} · ${request.side==="input"?"输入":"输出"}检查`);return;}
  const node=request.editor.nodes.find(n=>n.id===request.nodeId);if(!node)return;
  axisStage=true;axisEditing=request.drawing?request:null;stopPlay();scene.width=data.frame.shape[1];scene.height=data.frame.shape[0];scene.zoom=1;scene.panX=scene.panY=0;scene.tool=request.drawing?"roi":"pan";scene.roi=request.drawing?{x:node.parameters.x,y:node.parameters.y,width:node.parameters.width||scene.width-node.parameters.x,height:node.parameters.height||scene.height-node.parameters.y}:null;
  if(axisEditing)Object.assign(axisEditing,{undo:[],redo:[],committed:structuredClone(scene.roi)});
  workbench.stageShown(data,request);
  updateROI();if(splitActive()){$("split-tool").value="pan";split.view="stage";}
  await originalViewer.set(data.frame);$("axis-stage-note").hidden=false;$("axis-stage-caption").textContent=`${request.editor.specs[node.algorithm].name} · ${request.side==="input"?"此步输入":"此步输出"} · ${data.contract.label}${data.contract.axes.startsWith("P")?" · 显示左路":""}${request.drawing?" · 画布编辑当前实例":""}`;$("input-label").textContent="模块阶段检查";setMode($("viewers").dataset.mode==="result"?"compare":$("viewers").dataset.mode);redraw();
  if($("display-target").value==="input")fillDisplayControls(data.frame,display);
}
$("axis-restore-raw").onclick=safe(async()=>{axisEditing=null;axisStage=false;$("axis-stage-note").hidden=true;if(input)await setInput(input);});
$("cancel-inspection").onclick=()=>{cancelInspection();toast("阶段检查已取消");};
$("stage-sample").addEventListener("change",cancelInspection);
async function setAnalysisMode(mode) {
  $("analysis-mode").value=["split_correlation","split_curve"].includes(mode)?mode:"general";
  document.body.dataset.analysis="general";
  document.querySelector(".workspace").dataset.analysis="general";
  layout.apply();redraw();scheduleDraft();
}function redraw(viewer) {
  if(splitActive()&&!axisStage)split.draw();else originalViewer.draw();
  if(resultOwner==="split")split.drawResult();else resultViewer.draw();
  $("zoom-value").textContent =
    `${Math.round(visibleViewer().scale() * 100) || 100}%`;
  for(const tool of["pan","roi","point"])$("tool-"+tool).classList.toggle("selected",scene.tool===tool);
}
function visibleViewer() {
  return $("viewers").dataset.mode === "result" ? resultViewer : originalViewer;
}
function fit(viewer=visibleViewer()) {
  if(viewer===resultViewer&&resultOwner!=="split"){resultScene.zoom=1;resultScene.panX=resultScene.panY=0;redraw();return;}
  if(viewer===resultViewer&&resultOwner==="split"){split.fitResult();redraw();return;}
  scene.zoom = 1;
  scene.panX = scene.panY = 0;
  redraw();
}
function signature() {
  return JSON.stringify({
    active: active?.id,
    ref: input?.ref,
    steps,
    roi: splitActive()?generalROI:scene.roi,
  });
}
function changed() {
  cancelInspection();
  $("stale").hidden = resultOwner==="split" || !result || resultSignature === signature();
  updateSummary();
  updateROIStatistics();
  scheduleDraft();
}
function updateROI() {
  const q = splitActive()?generalROI:scene.roi;
  $("roi-info").textContent = q
    ? `x ${q.x} · y ${q.y}  /  ${q.width} × ${q.height} px`
    : "全幅图像";
  for (const key of ["x", "y", "width", "height"])
    $("roi-" + key).value = q?.[key] ?? "";
  $("roi-undo").disabled = !(axisEditing?.undo||roiUndo).length;
  $("roi-redo").disabled = !(axisEditing?.redo||roiRedo).length;
  $("roi-editor").hidden = !q;
  $("clear-roi").disabled = !q;
}
let roiStatisticsKey = "",
  roiStatisticsTimer;
function updateROIStatistics() {
  if (!input) return;
  const token = input.token,
    roi = splitActive()?generalROI:scene.roi,
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
  if(axisEditing){commitModuleROI(roi);return;}
  scene.roi = roi;
  rememberROI();
  updateROI();
  changed();
  redraw();
}
function stashView() {
  if (active && input) {
    const transform=splitActive()?split.previewTransform():scene;
    sourceViews.set(active.id, {
      roi: structuredClone(splitActive()?generalROI:scene.roi),
      display: { ...display },
      ref: input.ref,
      zoom: transform.zoom,
      panX: transform.panX,
      panY: transform.panY,
      seek_mode: $("seek-mode").value,
    });
  }
}
function updateSummary() {
  $("step-count").textContent=steps.length;
  const count = (splitActive()?split.cfg?.workflow||[]:steps).filter((s) => s.enabled).length;
  $("run-summary").textContent = active
    ? `${$("batch").checked ? sources.filter((s) => s.role === "input").length : 1} 个来源 · ${count} 个处理步骤${scene.roi ? " · ROI" : ""}`
    : "打开数据以开始处理";
  $("trial").disabled = $("run").disabled = !active || !input || requestBusy;
  $("run-current").disabled = !active || !input || !generalEditor.selected || requestBusy;
  $("run-current").title = generalEditor.selected?`从输入运行至 ${generalEditor.selected.name||generalEditor.specs[generalEditor.selected.algorithm]?.name}（包含此模块）`:"选择运行终点模块";
  for (const id of [
    "previous",
    "next",
    "play",
    "seek",
    "seek-value",
    "seek-mode",
  ])
    $(id).disabled = !active || !input;
  for (const id of ["seek", "seek-value", "seek-mode"])
    $(id).disabled ||= requestBusy;
  if (active && input && indexed()) {
    $("play").disabled = (active.meta.frame_count || 1) <= 1;
    $("previous").disabled = (input.ref.index ?? 0) <= 0;
    $("next").disabled =
      (input.ref.index ?? 0) >= (active.meta.frame_count || 1) - 1;
  }
  $("scrubber").disabled =
    !active || !input || (timePosition() && !active.meta.duration);
  $("promote-result").disabled = !formalResult;
  $("flow-summary").textContent = active
    ? `${active.lineage?.length ? "阶段 " + (active.lineage.length + 1) + " · " : ""}${active.display_name || active.meta.name} → ${count} 个处理步骤 → ${formalResult ? "已保存输出" : result ? "当前帧试算" : "待处理"}`
    : "导入来源 → 配置算法 → 检查结果";
}
function updateSources() {
  const inputs = sources.filter((s) => s.role === "input"),
    results = sources.filter((s) => s.role === "result"&&s.source.kind!=="dataset");
  $("source-count").textContent = inputs.length;
  const splitResults=jobs.filter(j=>j.mode==="split"&&j.status==="succeeded");
  const sequenceJobs=jobs.filter(j=>j.engine==="sequence"&&["succeeded","completed_with_errors"].includes(j.status));
  $("result-count").textContent = results.length+splitResults.length+sequenceJobs.length;
  const card = (s) =>
    `<div class="source-card ${s.id === active?.id || s.id === formalResult?.id ? "selected" : ""}" tabindex="0" role="button" data-source="${s.id}" aria-label="预览 ${escape(s.display_name || s.meta.name)}"><img src="${s.thumbnail}" alt=""><div class="source-copy"><strong title="${escape(s.source.path)}">${escape(s.display_name || s.meta.name)}</strong><span class="source-kind">${s.role === "result" ? "处理结果" : s.lineage?.length ? "派生输入 · 阶段 " + (s.lineage.length + 1) : s.meta.kind === "video" ? "视频" : s.meta.kind === "sequence" ? "图像序列" : "图像"}</span><small>${s.meta.contract?escape(s.meta.contract.label):`${s.meta.shape[1]} × ${s.meta.shape[0]}`} · ${escape(s.meta.contract?.dtype||s.meta.dtype)}</small><button class="text-button source-info" data-info="${s.id}">属性 / 来源</button></div><button class="icon-button source-remove" data-remove="${s.id}" aria-label="移除 ${escape(s.display_name || s.meta.name)}">${icon("close")}</button></div>`;
  $("source-list").innerHTML = inputs.length
    ? inputs.map(card).join("")
    : '<div class="library-empty">导入图像、视频<br>或图像序列目录</div>';
  $("result-list").innerHTML = results.map(card).join("")+splitResults.map(j=>`<div class="split-result-card ${split.resultId===j.id?"selected":""}" data-split-result="${j.id}" tabindex="0" role="button" aria-label="打开处理结果 ${escape(j.name)}"><strong>${escape(j.name)}</strong><span class="source-kind">${({statistics:"全源统计",trial:"空间图试算",curve:"τ 扫描曲线",run:"空间系数图"}[j.task])}</span><small>${escape(j.summary||j.source_name||"")}</small><small>${new Date(j.created*1000).toLocaleString()} · ${escape(j.id.slice(0,6))}</small></div>`).join("");
  $("result-list").insertAdjacentHTML("beforeend",sequenceJobs.map(j=>`<div class="split-result-card" data-sequence-result="${j.id}" tabindex="0" role="button"><strong>${escape(j.name)}</strong><span class="source-kind">${j.mode==="trial"?"模块试算":"模块处理"}</span><small>${escape(j.summary||"")}</small><small>${new Date(j.created*1000).toLocaleString()} · ${escape(j.id.slice(0,6))}</small></div>`).join(""));
  document.querySelectorAll("[data-sequence-result]").forEach(el=>{el.onclick=safe(()=>sequenceResults.load(el.dataset.sequenceResult));el.onkeydown=e=>{if(e.key==="Enter")el.click();};});
  document.querySelectorAll("[data-split-result]").forEach(el=>{el.onclick=safe(async()=>{sequenceResults.hide();await split.loadResult(el.dataset.splitResult);});el.onkeydown=e=>{if(e.key==="Enter")el.click();};});
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
    cine_time64: "CINE 文件逐帧时间记录",
    correlation_window_ref: "关联窗口中心的原始时间记录",
    unavailable: "未知 / 未指定",
  };
  const rows = [
    ["来源", entry.display_name || entry.meta.name],
    ["路径", entry.source.path],
    ["类型", entry.meta.kind],
    ["数据尺寸", entry.meta.contract?.label||entry.meta.shape.join(" × ")],
    ["像素类型", entry.meta.dtype],
    [
      "有效位深",
      entry.meta.sample_bits ? entry.meta.sample_bits + " bit" : "未声明",
    ],
    [
      "按帧跳转",
      indexed(entry) ? "支持精确帧号" : "按时间定位，已知帧号会显示",
    ],
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
    `<table class="metadata-table">${rows.map(([k, v]) => `<tr><td>${escape(k)}</td><td>${escape(v)}</td></tr>`).join("")}</table><p class="small-copy">${escape((entry.meta.notes || []).join("；"))}</p>${(entry.lineage || []).map((p, i) => `<div class="lineage-item"><strong>上游阶段 ${i + 1} · ${escape(p.run_id)}</strong>${p.crop_origin?`<br>裁剪原点：${escape(p.crop_origin.join(", "))} px`:""}${p.lag_frames!=null?`<br>延迟：${escape(p.lag_frames)} 帧 · 平均窗口：${escape(p.window)} 帧`:""}<br>来源：${escape(p.parent_source.path)}<br>运行记录：${escape(p.manifest)}</div>`).join("")}`;
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
  generalSequenceContext=null;
  generalEditor.replace([]);
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
  for(const key of mode==="compare"?["input","result"]:[mode])layout.state.windows[key].hidden=false;
  layout.apply();
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.classList.toggle("selected", b.dataset.view === mode));
  requestAnimationFrame(redraw);
  scheduleDraft();
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
  cancelInspection();
  axisStage=false;axisEditing=null;scene.annotation=null;scene.tool="pan";$("module-preview-tools").hidden=true;$("axis-stage-note").hidden=true;
  $("input-label").textContent="原始数据";
  pixelGeneration++;
  input = snapshot;
  await split.contextChanged();
  if(splitActive())split.inputChanged(snapshot);
  scene.width = snapshot.base_shape[1];
  scene.height = snapshot.base_shape[0];
  if (reset) {
    if(!splitActive())scene.roi = null;
    fit();
    updateROI();
  }
  await originalViewer.set(snapshot);
  if (input !== snapshot) return;
  $("welcome").hidden = true;
  $("preview-title").textContent = active.display_name || active.meta.name;
  $("dtype-badge").textContent = snapshot.contract?.dtype||snapshot.dtype;
  $("shape-badge").textContent = snapshot.contract?.label||`${snapshot.shape[1]} × ${snapshot.shape[0]}`;
  const ref = snapshot.ref;
  $("frame-info").textContent =
    `${ref.index == null ? "帧号未知" : `第 ${ref.index + 1} 帧`}${active.meta.frame_count ? " / " + active.meta.frame_count : ""}  |  ${ref.time == null ? "时间未知" : `${formatTime(ref.time)} s`}${snapshot.downsampled ? "  |  显示降采样" : ""}`;
  updatePosition();
  fillDisplayControls(
    $("display-target").value === "input" ? snapshot : result,
    $("display-target").value === "input" ? display : resultDisplay,
  );
  changed();
  redraw();
}
function indexed(entry = active) {
  return (
    !!entry &&
    (entry.meta.kind !== "video" || entry.meta.frame_indexed === true)
  );
}
function timePosition() {
  return $("seek-mode").value === "time";
}
function formatTime(value) {
  return Number(value).toFixed(
    (active?.meta.rate || 0) > 10000 ||
      active?.meta.timestamp_source === "cine_time64"
      ? 6
      : 4,
  );
}
function updatePosition() {
  if (!input) return;
  const ref = input.ref;
  $("seek-value").value = timePosition()
    ? active.meta.frame_indexed
      ? String(ref.time ?? 0)
      : formatTime(ref.time ?? 0)
    : (ref.index ?? 0) + 1;
  $("scrubber").value = timePosition() ? (ref.time ?? 0) : (ref.index ?? 0);
  $("timeline-note").textContent =
    ref.source_number != null
      ? `采集帧 ${ref.source_number}${ref.trigger_time != null ? " · 触发 " + formatTime(ref.trigger_time) + " s" : ""}`
      : `${ref.index == null ? "帧号未建立" : "当前第 " + (ref.index + 1) + " 帧"}${ref.time == null ? " · 时间未知" : " · " + formatTime(ref.time) + " s"}`;
  $("frame-info").title =
    ref.source_number != null
      ? `文件帧号从 1 开始；采集源帧 ${ref.source_number}；文件时间相对首帧，触发时间另列`
      : "实际返回帧与时间；未知索引不会用帧率推算";
  $("timeline-note").title =
    `${$("timeline-note").textContent}${ref.time == null ? "" : "；实际文件时间 " + ref.time + " s"}`;
  $("seek-value").title = timePosition()
    ? `实际文件时间 ${ref.time ?? "未知"} s`
    : `文件帧号 ${(ref.index ?? 0) + 1}`;
}
function configurePositionControls(entry = active) {
  if (!entry) return;
  const video = timePosition();
  $("scrubber").max = video
    ? entry.meta.duration || 1
    : Math.max(0, (entry.meta.frame_count || 1) - 1);
  $("scrubber").step = video
    ? (entry.meta.rate || 0) > 10000
      ? "0.000001"
      : "0.001"
    : "1";
  $("scrubber").disabled = video && !entry.meta.duration;
  $("seek-value").step = video ? "any" : "1";
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
  updatePosition();
  updateSummary();
}
function configureTimeline(entry) {
  const video = entry.meta.kind === "video";
  const noTimeRange = video && entry.meta.timestamp_source === "unavailable";
  $("scope").querySelector('[value="range"]').disabled = noTimeRange;
  if (noTimeRange && $("scope").value === "range") {
    $("scope").value = "current";
    $("range-fields").hidden = true;
  }
  $("seek-mode").querySelector('[value="frame"]').disabled = !indexed(entry);
  $("seek-mode").querySelector('[value="time"]').disabled =
    !video || entry.meta.timestamp_source === "unavailable";
  if (!indexed(entry)) $("seek-mode").value = "time";
  else if (!video || entry.meta.timestamp_source === "unavailable")
    $("seek-mode").value = "frame";
  configurePositionControls(entry);
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
  cancelInspection();
  stopPlay();
  pendingSeek = null;
  const entry = sources.find((s) => s.id === id);
  if (!entry) return;
  if (entry.role === "result") {
    if(entry.source.kind==="dataset"&&entry.run_id&&jobs.some(j=>j.id===entry.run_id&&j.engine==="sequence")){await sequenceResults.load(entry.run_id,entry.source_position||0,entry.side_output||null);return;}
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
  sequenceResults.hide();
  const view = sourceViews.get(id);
  $("seek-mode").value = view?.seek_mode || (indexed(entry) ? "frame" : "time");
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
  resultOwner="general";generalROI=null;$("split-result-controls").hidden=true;$("split-result-title").hidden=true;$("split-curve").hidden=true;
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
    if(splitActive())generalROI=structuredClone(view.roi);else scene.roi = structuredClone(view.roi);
    committedROI = structuredClone(view.roi);
  }
  if (view?.zoom) {
    scene.zoom = view.zoom;
    scene.panX = view.panX;
    scene.panY = view.panY;
  }
  if (view?.ref && (view.ref.index !== 0 || view.ref.time > 0))
    await seekFrame(
      !indexed(entry)
        ? { action: "time", time: view.ref.time ?? 0 }
        : { action: "index", index: view.ref.index ?? 0 },
    );
  updateROI();
  redraw();
  changed();
  if (entry.meta.notes?.length)
    toast(
      entry.meta.frame_indexed
        ? `${entry.meta.frame_count} 帧 · 精确帧号定位 · ${entry.meta.sample_bits!=null?`保存值 ${entry.meta.sample_bits} bit`:entry.meta.dtype}；读取说明见“属性 / 来源”`
        : entry.meta.notes.join("；"),
    );
}
async function seekFrame(value) {
  if (!active || !input) return;
  if (requestBusy) {
    pendingSeek = value;
    return;
  }
  requestBusy = true;
  updateSummary();
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
    updatePosition();
    updateSummary();
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
  preview.stop();
}
function step(direction) {
  if (!active || !input) return;
  if (indexed()) {
    const next = (input.ref.index ?? 0) + direction;
    if (next < 0 || next >= (active.meta.frame_count || 1)) {
      if (direction > 0) stopPlay();
      return;
    }
  }
  return seekFrame(
    !indexed()
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
  if (!active || !input) return;
  if(splitActive()&&!playing)await split.showRaw(input);
  await preview.play({advance:()=>step(1),token:()=>input?.token,interval:()=>Math.max(25,1000/(active?.meta.rate||10))});
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
  $("step-count").textContent=steps.length;
  generalEditor.render();updateSummary();
}
function addStep(){workbench.openLibrary();}
async function submit(mode,stopAfter=null) {
  if (!active || !input) return;
  if(stopAfter&&!steps.some(n=>n.id===stopAfter))throw Error("请先选择运行终点模块");
  const error=stopAfter?null:generalEditor.validate();if(error)throw Error(error);
  for (const el of $("general-module-fields").querySelectorAll("input"))
    if (!el.reportValidity()) return;
  const end=steps.find(n=>n.id===stopAfter),name=$("job-name").value.trim();
  return submitModules(steps,mode,end?`${name||active.display_name||active.meta.name} · 至 ${end.name||generalEditor.specs[end.algorithm]?.name}`:name||null,{stop_after:stopAfter,sample_index:generalEditor.previewSample||0,source_ids:mode==="run"&&$("batch").checked?sources.filter(s=>s.role==="input").map(s=>s.id):[active.id]});
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
  workbench.jobsChanged();
  updateSources();
  split.updateJobs(list);
  $("job-count").textContent = list.length;
  $("queue-note").textContent = list.some((j) => j.status === "running") ? "计算中" : "就绪";
  $("job-list").innerHTML = list.length
    ? list
        .map(
          (j) =>
            `<div class="job-row"><span class="job-kind">${j.mode==="split"?({statistics:"全源统计",trial:"窗口关联",curve:"τ 曲线",run:"时序关联"}[j.task]||"分幅关联"):j.mode === "trial" ? "帧试算" : "正式运行"}</span><span class="job-name" title="${escape(j.progress_name||j.name)}">${escape(j.name)}<small>${escape(j.summary||j.source_name||"")}</small>${j.source_count > 1 ? " · " + j.source_count + " 个来源" : ""}${j.mode==="split"&&j.status==="running"?" · "+escape(j.progress_name||"准备"):""}</span><span class="job-status ${j.status}">${statusText[j.status] || escape(j.status)}</span><span class="job-progress">${j.completed}${j.total ? " / " + j.total : " 帧"}</span>${["running", "queued", "cancelling"].includes(j.status) ? `<button data-cancel="${j.id}" class="quiet">取消</button>` : ""}<button data-detail="${j.id}" class="quiet">详情</button></div>`,
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
      if(j.engine==="sequence"){
        if(j.status==="succeeded"||j.status==="completed_with_errors"){
          sources=(await api("state")).sources;updateSources();
          if(active?.id===j.source_ids[0]&&!axisEditing&&!inspectionRequest)await sequenceResults.load(j.id);
          toast(j.status==="succeeded"?"模块流程完成，完整结果已保存":"部分来源失败，请查看任务详情",j.status!=="succeeded");
        }else if(full.error||full.run?.results?.[0]?.error)toast(full.error||full.run.results[0].error,true);
      } else if (j.mode === "split" || split.requested.has(j.id)) {
        await split.completed(full);
      } else if (j.mode === "trial" && full.result) {
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
  if(job.mode==="split") return split.jobDetail(job);
  $("job-dialog").showModal();
  const parts = [
    detailHTML(job),
  ];
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
  if (job.directory)
    parts.push(
      `<div class="downloads"><button data-download="manifest" class="quiet">运行记录 JSON</button>${(job.run?.results || []).map((r, i) => (r.status === "succeeded" ? `<button data-download="csv" data-index="${i}" class="quiet">来源 ${i + 1} CSV</button>${!r.contract||r.contract.axes.includes("Y")?`<button data-download="image" data-index="${i}" class="quiet">来源 ${i + 1} 首帧 TIFF</button>`:`<button data-download="svg" data-index="${i}" class="quiet">来源 ${i + 1} 曲线 SVG</button>`}${r.contract?`<button data-download="array" data-index="${i}" class="quiet">来源 ${i + 1} 完整 NPY</button>`:""}` : "")).join("")}</div>`,
    );
  if(job.engine==="sequence"&&["succeeded","completed_with_errors"].includes(job.status))parts.push(`<button id="show-sequence-result" class="primary">查看模块流程结果</button><p class="hint">各步输入输出形状、参数和完整样本映射保存在运行记录。信号曲线与矩阵使用相同流程引擎。</p>`);
  $("job-detail").innerHTML = parts.join("");
  bindRename(job,async()=>{await pollJobs();await jobDetail(id);},toast);
  $("show-sequence-result")?.addEventListener("click",safe(async()=>{await sequenceResults.load(id);$("job-dialog").close();}));
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
        !indexed()
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
      state.modules=await api("modules");
      algorithms=Object.values(state.modules);
      updateAlgorithms();
      $("algorithm").value = spec.id;
      generalEditor.renderLibrary();
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
      const data = await api("project/load", { path: fileBrowser.selected[0], migrate:true });
      stopPlay();
      pendingSeek = null;
      sourceGeneration++;
      selectionRequest++;
      active = input = result = formalResult = null;
      state = { ...state, ...data.state };
      sources = state.sources;
      algorithms = Object.values(state.modules);
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
      await restoreProjectOptions(data.view);
      changed();
      if(data.view.migration){$("project-title").textContent=data.state.project_path.split(/[\\/]/).pop();toast("旧项目已另存迁移副本；原文件保持完整");workbench.showMigration(data.view.migration);}else toast("项目已恢复");
    }
    $("file-dialog").close();
  } finally {
    fileBrowser.render();
  }
}
function projectView() {
  stashView();
  return {
    workbench_schema:2,
    web_module_contract:3,
    analysis_mode:$("analysis-mode").value,
    split_mode:split.serialize(),
    web_job_name:$("job-name").value,
    web_sequence_context:generalSequenceContext,
    web_sequence_result:resultOwner==="sequence"&&sequenceResults.id?{id:sequenceResults.id,source:sequenceResults.source,index:sequenceResults.index,side_output:sequenceResults.sideOutput,plot_axis:$("sequence-plot-axis").value,series:$("sequence-series").value,x:$("sequence-x").value,channel:$("sequence-channel").value}:null,
    web_selected_module:generalEditor.selected?.id||null,
    web_docks:layout.serialize(),
    web_preview:{schema:1,view:$("viewers").dataset.mode},
    steps,
    output_dir: $("output-dir").value,
    active_source: active?.id,
    web_display: display,
    web_result_display: resultDisplay,
    web_roi: splitActive()?generalROI:scene.roi,
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
async function restoreProjectOptions(view) {
  split.restore(view.split_mode);
  generalEditor.selectedId=view.web_selected_module||null;
  generalSequenceContext=view.web_sequence_context||null;
  $("job-name").value=view.web_job_name||"";
  await setAnalysisMode(view.analysis_mode);
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
  layout.restore(view.web_docks);
  if(active&&view.split_mode?.resultId&&(split.resultId!==view.split_mode.resultId||!split.resultData)){
    try{await split.loadResult(view.split_mode.resultId,false);}catch(error){toast(`历史结果未能恢复：${error.message}；原始数据仍可预览`,true);}
  }
  if(active&&view.web_sequence_result&&(sequenceResults.id!==view.web_sequence_result.id||resultOwner!=="sequence")){
    try{const saved=view.web_sequence_result;$("sequence-x").value=saved.x||"samples";$("sequence-channel").value=saved.channel||"0";$("sequence-plot-axis").value=saved.plot_axis||"sequence";await sequenceResults.load(saved.id,saved.source||0,saved.side_output||null);sequenceResults.index=saved.index||0;$("sequence-channel").value=saved.channel||"0";await sequenceResults.show();$("sequence-series").value=saved.series||"0";if(sequenceResults.plotData)sequenceResults.plot(sequenceResults.plotData);}catch(error){toast(`模块结果未能恢复：${error.message}；可从处理结果列表重开`,true);}
  }
  const previewMode=view.web_preview?.view||({both:"compare",data:"input",result:"result"}[view.split_mode?.layout]);
  if(["input","compare","result"].includes(previewMode))setMode(previewMode);
  await migrateGeneralFlow(view);
  generalEditor.render();
  updateSummary();
}
async function migrateGeneralFlow(view){
  if(view.workbench_schema>=2)return;
  if(view.web_module_contract!==3&&!active)return;
  if(view.web_module_contract!==3){
  if(!steps.some(n=>n.algorithm==="time_crop")){const time=moduleNode(state.modules.time_crop,active),range=view.web_range,ctx=view.web_sequence_context?.config;
    if(ctx)Object.assign(time.parameters,{first:(ctx.start||0)+1,last:ctx.end||null});
    else if(view.web_scope==="range"&&range){if(active.meta.kind==="video"&&!active.meta.frame_indexed)Object.assign(time.parameters,{unit:"seconds",start:Number(range[0]),end:Number(range[1])});else Object.assign(time.parameters,{first:Number(range[0]),last:Number(range[1])-1});}
    steps.unshift(time);
  }
  const q=view.web_roi;if(q&&!steps.some(n=>n.algorithm==="spatial_roi")){const roi=moduleNode(state.modules.spatial_roi,active);Object.assign(roi.parameters,q);steps.push(roi);}
  scene.roi=null;generalROI=null;$("scope").value="all";view.web_module_contract=3;
  }
  const migrated=await api("modules/migrate",{steps});
  steps=migrated.nodes;view.steps=structuredClone(steps);view.workbench_schema=2;
  renderSteps();if(migrated.report.length)toast("旧流程草稿已迁移：ROI、统计和关联已拆分，原保存项目不变。");
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
  layout.apply();
  requestAnimationFrame(redraw);
  scheduleDraft();
}
document.addEventListener("workbench-layout",updateLayout);
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
$("job-name").oninput=scheduleDraft;
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
$("add-step").onclick = ()=>workbench.openLibrary();
$("fit").onclick = () => fit();
$("actual").onclick = () => {
  const viewer=visibleViewer(),canvas=viewer.canvas,target=viewer===resultViewer&&resultOwner!=="general"?resultScene:scene;
  if(target===resultScene&&resultOwner==="split")split.resultAutoFit=false;
  target.zoom =
    1 /
    Math.min(
      (canvas.clientWidth - 40) / target.width,
      (canvas.clientHeight - 50) / target.height,
    );
  target.panX = target.panY = 0;
  redraw();
};
for (const tool of ["pan", "roi", "point"])
  $("tool-" + tool).onclick = safe(async() => {
    if(tool==="roi"){await drawModuleROI();return;}
    if(tool==="point"){if(!active)throw Error("请先导入来源");if(generalEditor.selected?.algorithm!=="point_extract")generalEditor.add("point_extract");await previewModule(generalEditor,"input",true,true,"point");return;}
    if(splitActive()){$("split-tool").value=tool;split.setTool(tool);}
    scene.tool = tool;
    $("tool-pan").classList.toggle("selected", tool === "pan");
    $("tool-roi").classList.toggle("selected", tool === "roi");
    $("tool-point").classList.toggle("selected", tool === "point");
  });
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
  if(axisEditing){if(axisEditing.undo.length){axisEditing.redo.push(structuredClone(scene.roi));commitModuleROI(axisEditing.undo.pop(),false);}return;}
  if (!roiUndo.length) return;
  roiRedo.push(structuredClone(scene.roi));
  scene.roi = roiUndo.pop();
  committedROI = structuredClone(scene.roi);
  updateROI();
  changed();
  redraw();
};
$("roi-redo").onclick = () => {
  if(axisEditing){if(axisEditing.redo.length){axisEditing.undo.push(structuredClone(scene.roi));commitModuleROI(axisEditing.redo.pop(),false);}return;}
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
$("run-current").onclick = safe(() => submit("run",generalEditor.selected?.id));
$("previous").onclick = safe(() => step(-1));
$("next").onclick = safe(() => step(1));
$("play").onclick = safe(play);
$("scrubber").oninput = safe(() => {
  stopPlay();
  return seekFrame(
    timePosition()
      ? { action: "time", time: Number($("scrubber").value) }
      : { action: "index", index: Number($("scrubber").value) },
  );
});
$("seek").onclick = safe(() => {
  if (!$("seek-value").reportValidity()) return;
  stopPlay();
  return seekFrame(
    timePosition()
      ? { action: "time", time: Number($("seek-value").value) }
      : { action: "index", index: Number($("seek-value").value) - 1 },
  );
});
$("seek-mode").onchange = () => {
  stopPlay();
  pendingSeek = null;
  configurePositionControls();
  scheduleDraft();
};
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
  sequenceResults.hide();
  resultOwner="general";$("split-result-controls").hidden=true;$("split-result-title").hidden=true;$("split-curve").hidden=true;$("split-result-empty").hidden=true;
  resultScene.width=snapshot.base_shape[1];resultScene.height=snapshot.base_shape[0];
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
    displaySnapshot(),
    $("display-target").value === "result" ? resultDisplay : display,
  );
function displaySnapshot(target=$("display-target").value){return target==="result"?result:axisStage?originalViewer.snapshot:input;}
async function renderDisplay(
  auto = false,
  supplied = null,
  windowChanged = false,
) {
  const target = $("display-target").value,
    snapshot = displaySnapshot(target);
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
    displaySnapshot(target)?.token !== token
  )
    return;
  if (target === "result") {
    resultDisplay = settings;
    result = rendered;
    if(resultOwner==="split"){await resultViewer.set(rendered);split.drawResult();}else if(resultOwner==="sequence"){await resultViewer.set(rendered);redraw();}else await paintResult(result);
  } else {
    display = settings;
    if(axisStage){await originalViewer.set(rendered);redraw();}else await setInput(rendered);
  }
  fillDisplayControls(rendered,settings);
  scheduleDraft();
}
$("copy-display").onclick = safe(async () => {
  const from = $("display-target").value,
    snapshot = displaySnapshot(from);
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
$("toggle-tasks").onclick = () => {
  document.querySelector(".workspace").classList.toggle("task-collapse");
  layout.apply();scheduleDraft();
};
document.addEventListener("keydown", (e) => {
  if(e.defaultPrevented)return;
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
    const moved=moveROI(
        scene.roi,
        e.key === "ArrowRight" ? delta : e.key === "ArrowLeft" ? -delta : 0,
        e.key === "ArrowDown" ? delta : e.key === "ArrowUp" ? -delta : 0,
        scene.width,
        scene.height,
      );
    if(axisEditing)commitROI(moved);
    else if(splitActive()){scene.roi=moved;split.commitROI();redraw();}
    else commitROI(moved);
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    if(!axisEditing){e.preventDefault();generalEditor.undo(e.shiftKey);return;}
    if(splitActive()&&!axisEditing)return;
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
  algorithms = Object.values(state.modules);
  steps=[moduleNode(state.modules.time_crop,null)];
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
    await restoreProjectOptions(draft);
    renderSteps();
  }
  const initial =
    sources.find((s) => s.role === "input" && s.id === draft?.active_source) ||
    sources.find((s) => s.role === "input");
  if (initial) await selectSource(initial.id);
  if (draft) await restoreProjectOptions(draft);
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
