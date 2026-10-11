import { api, downloadSplit } from "./api.js";
import { Viewer } from "./viewer.js";
import { FlowEditor } from "./flow.js";
import { detailHTML, bindRename } from "./job-detail.js";

const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const clone = (v) => structuredClone(v);
const label = {statistics:"全源统计",trial:"空间图试算",curve:"τ 扫描曲线",run:"空间系数图"};
const bytes = (n) => n >= 1024**3 ? `${(n/1024**3).toFixed(2)} GiB` : `${(n/1024**2).toFixed(2)} MiB`;
const number = (n) => n == null ? "未知" : (Math.abs(n)>0 && (Math.abs(n)<.001||Math.abs(n)>1e5) ? Number(n).toExponential(3) : Number(n).toPrecision(5));
const coefficient = n => n==null?"无效":Number(n).toPrecision(10);

export class SplitController {
  constructor(bridge) {
    this.bridge=bridge;
    this.records={};
    this.sourceId=null;
    this.mode="general";
    this.jobs=[];
    this.requested=new Map();
    this.stage="geometry";
    this.view="raw";
    this.generation=0;
    this.autoFit=true;
    this.scene={width:640,height:480,zoom:1,panX:0,panY:0,tool:"pan",roi:null};
    this.viewer=new Viewer($("split-canvas"),this.scene,{redraw:()=>this.draw(),fit:()=>this.fit(),roi:()=>this.commitROI(),roiPreview:()=>this.draw(),pixel:(frame,x,y)=>this.pixel(frame,x,y)});
    this.resultScene={width:640,height:480,zoom:1,panX:0,panY:0,tool:"pan",roi:null};
    this.resultAutoFit=true;
    this.resultViewer=new Viewer($("split-result-canvas"),this.resultScene,{redraw:()=>this.drawResult(),fit:()=>this.fitResult(),pixel:(frame,x,y)=>this.pixel(frame,x,y,true)});
    $("split-result-canvas").addEventListener("wheel",()=>{this.resultAutoFit=false;},{capture:true,passive:true});
    $("split-result-canvas").addEventListener("pointerdown",()=>{this.resultAutoFit=false;},true);
    this.flow=new FlowEditor(this);
    this.bind();
  }
  wrap(fn) {
    return async (...args)=>{try{await fn(...args);}catch(error){this.stop();this.bridge.report(error.message,true);$("split-status").textContent=error.message;}};
  }
  get context() {return this.bridge.context();}
  get cfg() {return this.sourceId ? this.records[this.sourceId]?.config : null;}
  get record() {return this.sourceId ? this.records[this.sourceId] : null;}
  pixel(frame,x,y,output=false) {
    const target=$(output?"split-result-pixel":"split-pixel"),viewer=output?this.resultViewer:this.viewer;
    const key=`${frame.token}:${x}:${y}`;
    const property=output?"resultPixelKey":"pixelKey",timer=output?"resultPixelTimer":"pixelTimer";
    clearTimeout(this[timer]);this[property]=key;
    target.textContent=`${output||this.view!=="raw"?"参考":"传感器"}坐标 x ${x} · y ${y}`;
    if(x<frame.origin[0]||y<frame.origin[1]||x>=frame.origin[0]+frame.shape[1]||y>=frame.origin[1]+frame.shape[0])return;
    this[timer]=setTimeout(async()=>{try {
      const data=await api(`frames/${frame.token}/pixel?x=${x}&y=${y}`);
      if(this[property]!==key||viewer.snapshot?.token!==frame.token)return;
      target.textContent+=` · ${output&&this.resultView==="result"?"g²":"数值"} ${Array.isArray(data.value)?data.value.map(coefficient).join(", "):coefficient(data.value)}`;
    }catch(error){if(this[property]===key)target.textContent+=` · ${error.message}`;}},100);
  }
  bind() {
    $("split-open-context").onclick=()=>{$("split-run-context").open=!$("split-run-context").open;$("split-run-context").scrollIntoView({block:"nearest"});};
    $("split-open-results").onclick=()=>{$("split-results-tools").open=!$("split-results-tools").open;$("split-results-tools").scrollIntoView({block:"nearest"});};
    $("split-use-general").onclick=this.wrap(()=>this.bridge.useModules(clone(this.cfg?.workflow||[]),this.executionContext()));
    $("split-refresh-preview").onclick=this.wrap(()=>this.updateGeometry());
    $("split-layout").onchange=()=>{document.querySelector(".split-views").dataset.layout=$("split-layout").value;this.fit();this.fitResult();this.bridge.changed();};
    $("split-roi-draw").onclick=this.wrap(async()=>{await this.setStage("trial");$("split-tool").value="roi";this.setTool("roi");});
    $("split-point-draw").onclick=this.wrap(async()=>{await this.setStage("trial");$("split-tool").value="point";this.setTool("point");});
    $("split-result-fit").onclick=()=>this.fitResult();
    document.querySelectorAll("[data-split-analysis]").forEach(b=>b.onclick=()=>this.setAnalysis(b.dataset.splitAnalysis));
    $("split-job-name").oninput=()=>{if(this.record){this.record.names??={};this.record.names[this.record.analysis||"map"]=$("split-job-name").value;this.bridge.changed();}};
    $("split-result-rename").onclick=this.wrap(async()=>{if(!this.resultId)throw Error("请先选择完成的结果");await api(`jobs/${this.resultId}/rename`,{name:$("split-result-name").value});await this.bridge.poll();await this.loadResult(this.resultId,false);this.bridge.report("结果名称已保存");});
    document.querySelectorAll("[data-split-stage]").forEach(b=>b.onclick=this.wrap(()=>this.setStage(b.dataset.splitStage)));
    $("split-view").onchange=this.wrap(()=>this.showView($("split-view").value));
    $("split-tool").onchange=()=>this.setTool($("split-tool").value);
    $("split-fit").onclick=()=>this.fit();
    for(const id of ["split-prev","split-next"]) $(id).onclick=this.wrap(()=>this.seek(this.cfg.center+(id==="split-prev"?-1:1)));
    $("split-seek").onclick=this.wrap(()=>this.seek(Number($("split-center").value)-1));
    $("split-center").onchange=()=>{if(this.cfg){this.cfg.center=Number($("split-center").value)-1;this.changed();}};
    $("split-scrubber").onchange=this.wrap(()=>this.seek(Number($("split-scrubber").value)-1));
    $("split-play").onclick=this.wrap(()=>this.play(false));
    $("split-result-play").onclick=this.wrap(()=>this.play(true));
    $("split-suggest").onclick=this.wrap(()=>this.updateGeometry(true));
    $("split-preview").onclick=this.wrap(()=>this.updateGeometry());
    $("split-calibrate").onclick=this.wrap(()=>this.updateGeometry(false,true));
    $("split-apply-proposal").onclick=this.wrap(async()=>{
      if(!this.proposal) return;
      this.cfg.geometry.dx=this.cfg.geometry.interpolation==="nearest"?Math.round(this.proposal.dx):this.proposal.dx;
      this.cfg.geometry.dy=this.cfg.geometry.interpolation==="nearest"?Math.round(this.proposal.dy):this.proposal.dy;
      this.fill();this.changed(true);await this.updateGeometry();await this.showView("overlay");
    });
    $("split-accept").onclick=this.wrap(async()=>{
      await this.updateGeometry();
      this.record.accepted=JSON.stringify(this.cfg.geometry);
      this.record.calibrationIndex=this.snapshots.raw.ref.index;
      this.changed();
      await this.setStage("preprocess");
      this.bridge.report("固定标定已接受；正式任务将复用该映射");
    });
    $("split-statistics").onclick=this.wrap(()=>this.submit("statistics"));
    $("split-check-stats").onclick=this.wrap(()=>this.showStatistics());
    $("split-stat-view").onchange=this.wrap(()=>this.record?.statisticsJob?this.showStatistics():null);
    $("split-budget").onclick=this.wrap(()=>this.budget("run"));
    $("split-trial").onclick=this.wrap(()=>this.submit("trial"));
    $("split-run").onclick=this.wrap(()=>this.submit("run"));
    $("split-curve-run").onclick=this.wrap(()=>this.submit("curve"));
    $("split-set-point").onclick=()=>{
      if(!this.cfg) return;
      this.cfg.points=[[Number($("split-point-x").value),Number($("split-point-y").value)]];
      this.record.pointsAuto=false;this.changed();this.draw();
    };
    $("split-clear-points").onclick=()=>{if(this.cfg){this.cfg.points=[];this.record.pointsAuto=false;this.changed();this.draw();}};
    $("split-clear-analysis-roi").onclick=()=>{if(this.cfg){this.cfg.roi=null;this.scene.roi=null;this.changed();this.draw();}};
    $("split-apply-roi").onclick=this.wrap(()=>{
      if(!this.cfg||!this.snapshots) throw Error("请先完成分幅预览");
      const q=Object.fromEntries(["x","y","width","height"].map(key=>[key,Number($("split-roi-"+key).value)]));
      const [h,w]=this.snapshots.mapping?.shape||this.rawSnapshot.base_shape;
      if(Object.values(q).some(v=>!Number.isInteger(v))||q.x<0||q.y<0||q.width<1||q.height<1||q.x+q.width>w||q.y+q.height>h) throw Error(`ROI 需为 ${w}×${h} 参考幅面内的整数矩形`);
      this.cfg.roi=q;if(this.record.pointsAuto)this.cfg.points=[];
      this.scene.roi=clone(q);
      this.changed();this.draw();
    });
    $("split-result-select").onchange=this.wrap(()=>this.loadResult($("split-result-select").value));
    $("split-result-time").onchange=this.wrap(()=>{this.timeIndex=Number($("split-result-time").value);return this.showSlice();});
    $("split-result-lag").onchange=this.wrap(()=>{this.lagIndex=Number($("split-result-lag").value);return this.resultData?.run.array_shape?this.showSlice():this.showCurve();});
    $("split-show-map").onclick=this.wrap(()=>this.showSlice());
    $("split-show-curve").onclick=()=>this.showCurve();
    $("split-curve-axis").onchange=()=>this.showCurve();
    for(const id of ["split-display-low","split-display-high"]) $(id).onchange=this.wrap(()=>this.resultData?.run.array_shape?this.showSlice():null);
    document.querySelectorAll("[data-split-export]").forEach(b=>b.onclick=this.wrap(async()=>{
      if(!this.resultId) throw Error("请选择完成的关联结果");
      await downloadSplit(this.resultId,b.dataset.splitExport,this.timeIndex||0,this.lagIndex||0);
    }));
    $("split-promote").onclick=this.wrap(async()=>{
      if(!this.resultId) throw Error("请选择系数图结果");
      const opened=await api(`split/jobs/${this.resultId}/promote`,{lag_index:this.lagIndex||0});
      await this.bridge.promote(opened);
      this.bridge.report("选定延迟序列已作为下一阶段输入，来源与窗口记录保留");
    });
    $("split-restore").onclick=this.wrap(async()=>{
      const response=await api("split/restore",{path:$("split-restore-path").value.trim()});
      await this.bridge.poll();await this.loadResult(response.id);
    });
    const excluded=new Set(["split-center","split-point-x","split-point-y","split-black-threshold","split-result-time","split-display-low","split-display-high","split-roi-x","split-roi-y","split-roi-width","split-roi-height"]);
    document.querySelectorAll(".split-parameters input,.split-parameters select").forEach(el=>{
      if(excluded.has(el.id)||["split-result-select","split-stat-view","split-restore-path","split-curve-axis","split-module-type","split-module-position","split-result-name"].includes(el.id)) return;
      el.addEventListener("change",()=>{if(this.cfg){this.readFields();this.changed(el.closest('[data-module-panel="alignment"]')!=null);}});
    });
    const canvas=$("split-canvas");
    canvas.addEventListener("wheel",()=>{this.autoFit=false;},{capture:true,passive:true});
    canvas.addEventListener("pointerdown",()=>{this.autoFit=false;},true);
    canvas.addEventListener("pointerdown",e=>{
      if(!this.cfg||!this.viewer.snapshot||e.button!==0) return;
      const tool=$("split-tool").value;
      if(tool!=="point"&&tool!=="midline") return;
      e.stopImmediatePropagation();canvas.setPointerCapture(e.pointerId);
      const p=this.viewer.native(this.viewer.point(e));
      if(tool==="point") {
        if(this.view==="raw") {this.bridge.report("请先切到左幅参考或系数图，再选点",true);return;}
        const point=[Math.floor(p.x),Math.floor(p.y)];
        if(point.some((v,i)=>v<0||v>=(i===0?this.scene.width:this.scene.height))) return;
        this.cfg.points=e.shiftKey?[...this.cfg.points,point].slice(0,32):[point];
        this.record.pointsAuto=false;this.fillPoints();this.changed();this.draw();
      } else if(this.view==="raw") {
        const line=this.cfg.geometry.midline,h=this.scene.height;
        const ends=[{x:line[0],y:0},{x:line[1],y:h-1}];
        const near=ends.findIndex(v=>Math.hypot(v.x-p.x,v.y-p.y)<12/this.viewer.scale());
        this.lineDrag={start:p,end:near,old:[...line]};
      }
    },true);
    canvas.addEventListener("pointermove",e=>{
      if(!this.lineDrag||!this.cfg) return;
      e.stopImmediatePropagation();
      const p=this.viewer.native(this.viewer.point(e)),d=this.lineDrag,w=this.scene.width,h=this.scene.height;
      const clamp=x=>Math.max(1,Math.min(w-1,x));
      if(d.end>=0) this.cfg.geometry.midline[d.end]=clamp(p.x);
      else if(Math.abs(p.y-d.start.y)>2) {
        const slope=(p.x-d.start.x)/(p.y-d.start.y);
        this.cfg.geometry.midline=[clamp(d.start.x-slope*d.start.y),clamp(d.start.x+slope*(h-1-d.start.y))];
      }
      $("split-line-top").value=this.cfg.geometry.midline[0].toFixed(2);
      $("split-line-bottom").value=this.cfg.geometry.midline[1].toFixed(2);
      this.changed(true);this.draw();
    },true);
    canvas.addEventListener("pointerup",e=>{if(this.lineDrag){e.stopImmediatePropagation();this.lineDrag=null;this.changed(true);}},true);
    canvas.addEventListener("pointercancel",()=>{if(this.lineDrag){this.cfg.geometry.midline=this.lineDrag.old;this.lineDrag=null;this.fill();this.draw();}});
  }
  defaults(source,input) {
    const n=source.meta.frame_count||1,lag=Math.min(10,Math.max(0,Math.floor((n-3)/4)));
    let window=Math.min(101,n-2*lag);if(window%2===0) window--;window=Math.max(3,window);
    const center=Math.max(0,Math.min(n-1,Math.max(Math.floor(window/2)+lag,Math.min(n-1-Math.floor(window/2)-lag,input?.ref.index||0))));
    return {geometry:null,photometry:{lut_path:null,black_left:0,black_right:0,saturation_code:null,normalization:"none",input_semantics:"sensor_codes",curve:"linear",input_max:1023,gamma:1},window,stride:100,lag_min:-lag,lag_max:lag,lag_step:1,lag_list:null,start:0,end:n,center,roi:null,points:[],roi_estimator:"pixel_mean",min_mean:1e-9,min_samples:2,memory_mb:256,backend:"cpu",output_dtype:"float64"};
  }
  async contextChanged() {
    const {source,input}=this.context;
    if(!source||source.role!=="input") return;
    if(this.sourceId===source.id) return;
    this.stop();this.generation++;
    this.sourceId=source.id;
    if(!this.records[source.id]||this.records[source.id].revision!==source.meta.revision) this.records[source.id]={config:this.defaults(source,input),revision:source.meta.revision,accepted:null,pointsAuto:true,stage:"geometry"};
    this.snapshots=null;this.proposal=null;
    this.resultData=null;this.resultId=this.record.resultId||null;
    this.stage=this.record.stage||"geometry";
    this.view="raw";this.scene.roi=null;this.fit();
    this.fill();this.updateJobs(this.jobs);
    if(this.mode==="split_correlation") {
      await this.showRaw(input);await this.ensureGeometry();
    }
  }
  async setMode(mode) {
    this.mode=mode;this.stop();
    $("split-workspace").hidden=mode!=="split_correlation";
    document.querySelector(".workspace").dataset.analysis=mode;
    document.body.dataset.analysis=mode;
    if(mode==="split_correlation") {
      await this.contextChanged();
      if(this.cfg) {await this.showRaw(this.rawSnapshot||this.context.input);await this.ensureGeometry();this.flow.render();this.modulePanel();if(this.resultId&&!this.resultData)await this.loadResult(this.resultId,false);}
      else this.updateStatus();
    }
  }
  serialize() {return {records:clone(this.records),resultId:this.resultId||null,layout:$("split-layout").value};}
  restore(value) {
    if(value?.records&&typeof value.records==="object") {
      const fingerprint=JSON.stringify(value);
      if(this.restoredFingerprint===fingerprint) return;
      this.restoredFingerprint=fingerprint;
      this.records=clone(value.records);this.sourceId=null;
      this.resultId=value.resultId||null;
      const layout=["both","data","result"].includes(value.layout)?value.layout:"both";
      $("split-layout").value=layout;document.querySelector(".split-views").dataset.layout=layout;
    }
  }
  signature() {return JSON.stringify({source:this.sourceId,config:this.cfg});}
  activeNode(algorithm) {return this.cfg?.workflow?.find(n=>n.enabled&&n.algorithm===algorithm);}
  get selectedNode() {return this.cfg?.workflow?.find(n=>n.id===this.record?.selectedModule);}
  accepted() {return !this.activeNode("alignment")||this.record.accepted===JSON.stringify(this.cfg.geometry);}
  executionContext() {return Object.fromEntries(["start","end","center","roi","points","memory_mb","backend","output_dtype"].map(k=>[k,clone(this.cfg[k])]));}
  modulePanel() {
    const node=this.selectedNode,spec=this.context.modules?.[node?.algorithm];
    document.querySelectorAll("[data-module-panel]").forEach(el=>el.hidden=el.dataset.modulePanel!==node?.algorithm);
    $("split-module-inspector").hidden=!["alignment","calibration","statistics","correlation"].includes(node?.algorithm);
    $("split-module-title").textContent=spec?.name||"模块参数";
    $("split-module-instance").textContent=node?`实例 ${node.id.slice(0,8)} · v${node.version||"1.0"}${node.enabled?"":" · 已停用"}`:"";
  }
  previewError(error) {const el=$("split-preview-error");el.hidden=!error;el.textContent=error?`流程预览：${error.message||error}。原始帧仍可定位和播放。`:"";}
  async showRaw(snapshot) {
    if(!snapshot)return;const keep=this.view==="raw"&&this.scene.width===snapshot.base_shape[1]&&this.scene.height===snapshot.base_shape[0];this.rawSnapshot=snapshot;this.view="raw";$("split-view").value="raw";$("split-empty").hidden=true;
    this.scene.width=snapshot.base_shape[1];this.scene.height=snapshot.base_shape[0];const tool=$("split-tool").value;this.scene.roi=["left","right"].includes(tool)?clone(this.cfg?.geometry?.[tool]||null):!this.activeNode("alignment")&&tool==="roi"?clone(this.cfg?.roi):null;if(!keep){this.scene.zoom=1;this.scene.panX=this.scene.panY=0;}
    await this.viewer.set(snapshot);this.draw();$("split-view-caption").textContent=`${snapshot.shape[1]} × ${snapshot.shape[0]} · 原始帧 ${snapshot.ref.index==null?"未知":snapshot.ref.index+1}${snapshot.ref.time==null?"":` · ${number(snapshot.ref.time)} s`}`;
  }
  changed(geometry=false) {
    if(geometry&&this.record) {this.record.accepted=null;this.record.calibrationIndex=null;}
    this.flow.sync();this.fillPoints();this.updateStatus();this.windowNote();this.bridge.changed();
  }
  updateStatus() {
    const source=this.context.source,corr=this.activeNode("correlation"),stats=this.activeNode("statistics"),alignment=this.activeNode("alignment"),error=this.flow.validate();
    $("split-source-name").textContent=source?.display_name||source?.meta.name||"请选择输入来源";
    let type="sensor";for(const n of this.cfg?.workflow||[]){if(!n.enabled)continue;const out=this.context.modules?.[n.algorithm]?.output;if(out&&out!=="same")type=out;}
    const incomplete=type==="pair"?"流程输出为双路，请添加选择幅面或二阶关联模块":null;
    const missingLut=(this.cfg?.workflow||[]).some(n=>n.enabled&&this.record?.customLutNodes?.includes(n.id)&&!n.parameters.lut_path);
    const valid=!!source&&!error&&!incomplete&&!missingLut&&this.accepted();
    for(const id of["split-trial","split-run","split-curve-run"])$(id).disabled=!valid;
    $("split-statistics").disabled=!stats||!source||!!this.flow.validate(this.taskConfig("statistics").workflow)||!this.accepted();
    $("split-trial").textContent=corr?"当前中心系数图":"当前帧试算";
    $("split-run").textContent=corr?"运行固定 τ 系数图":stats?"运行全源统计":"运行图像流程";
    const curve=!!corr&&this.record?.analysis==="curve";$("split-curve-run").hidden=!curve;$("split-trial").hidden=curve||!!stats&&!corr;$("split-run").hidden=curve;
    $("split-status").textContent=error||incomplete||(missingLut?"请填写校准模块的 LUT 路径":alignment&&!this.accepted()?"请检查并接受当前分幅标定":corr?`${stats?"复用全源统计":"直接强度关联 · 未启用全源统计"} · 固定标定已接受`:`${(this.cfg?.workflow||[]).filter(n=>n.enabled).length} 个模块 · 逐帧处理`);
    let stale=false;if(this.resultData?.run&&this.cfg){const saved=clone(this.resultData.run.config),current=this.taskConfig(this.resultData.run.task);if(this.resultData.run.task==="run"){delete saved.center;delete current.center;}stale=JSON.stringify(saved)!==JSON.stringify(current)||this.resultData.run.provenance?.source_id!==this.sourceId;}
    $("split-stale").hidden=!stale;
    const devices=this.context.devices,gpu=$("split-backend").querySelector('option[value="gpu"]');gpu.disabled=!devices?.gpu;gpu.textContent=devices?.gpu?`GPU · ${devices.gpu_name}`:"GPU（不可用）";
    $("split-device-note").textContent=devices?.gpu?`关联后端可用：${devices.gpu_name}；逐帧模块使用其声明的设备。`:"CPU 完整可用；未检测到可用 CuPy / CUDA 环境。";
  }
  fill() {
    if(!this.cfg) {this.updateStatus();return;}
    this.flow.sync();const c=this.cfg,g=this.selectedNode?.algorithm==="alignment"?this.selectedNode.parameters:c.geometry,p=this.selectedNode?.algorithm==="calibration"?this.selectedNode.parameters:c.photometry,n=this.context.source.meta.frame_count||1;
    $("split-center").value=c.center+1;$("split-center").max=n;
    $("split-scrubber").max=n;$("split-scrubber").value=c.center+1;
    if(g) {
      $("split-line-top").value=g.midline[0];$("split-line-bottom").value=g.midline[1];
      $("split-dx").value=g.dx;$("split-dy").value=g.dy;
      $("split-mirror-x").checked=g.mirror_x;$("split-mirror-y").checked=g.mirror_y;$("split-interpolation").value=g.interpolation;
      for(const role of ["left","right"]) document.querySelectorAll(`[data-split-rect="${role}"] input`).forEach(el=>el.value=g[role][el.dataset.key]);
    }
    for(const [id,key] of [["split-window","window"],["split-stride","stride"],["split-lag-min","lag_min"],["split-lag-max","lag_max"],["split-lag-step","lag_step"],["split-min-mean","min_mean"],["split-min-samples","min_samples"],["split-memory","memory_mb"],["split-backend","backend"],["split-roi-estimator","roi_estimator"]]) $(id).value=c[key];
    $("split-range-start").value=c.start+1;$("split-range-end").value=(c.end??n)+1;$("split-lag-list").value=c.lag_list?.join(",")||"";
    $("split-output-dtype").value=c.output_dtype||"float32";
    $("split-semantics").value=p.input_semantics||"sensor_codes";$("split-lut").value=p.lut_path||"";
    $("split-black-left").value=p.black_left;$("split-black-right").value=p.black_right;
    $("split-saturation").value=p.saturation_code??"";$("split-normalization").value=(this.selectedNode?.algorithm==="statistics"?this.selectedNode.parameters.normalization:c.photometry.normalization)||"none";
    $("split-photo-curve").value=p.lut_path?"custom":p.curve||"identity";
    $("split-photo-max").value=p.input_max??1023;$("split-photo-gamma").value=p.gamma??1;
    $("split-map-lag").value=this.record.mapLag??0;
    this.setAnalysis(this.record.analysis||"map",false);this.photoPreview();this.flow.render();
    this.fillPoints();this.updateStatus();this.windowNote();
  }
  fillPoints() {
    if(!this.cfg) return;
    const p=this.cfg.points[0];
    if(p) {$("split-point-x").value=p[0];$("split-point-y").value=p[1];}
    $("split-point-list").innerHTML=this.cfg.points.map((p,i)=>`<span>点 ${i+1} (${p[0]}, ${p[1]})</span>`).join("");
    const q=this.cfg.roi;
    $("split-roi-status").textContent=q?`分析 ROI：(${q.x}, ${q.y}) · ${q.width} × ${q.height}`:"分析区域：全幅共同有效区";
    for(const key of ["x","y","width","height"]) $("split-roi-"+key).value=q?.[key]??"";
  }
  readFields() {
    if(!this.cfg)return;
    const c=this.cfg,node=this.selectedNode,p=node?.parameters;
    if(node?.algorithm==="alignment"){
      p.midline=[Number($("split-line-top").value),Number($("split-line-bottom").value)];p.dx=Number($("split-dx").value);p.dy=Number($("split-dy").value);p.mirror_x=$("split-mirror-x").checked;p.mirror_y=$("split-mirror-y").checked;p.interpolation=$("split-interpolation").value;
      for(const role of["left","right"])document.querySelectorAll(`[data-split-rect="${role}"] input`).forEach(el=>p[role][el.dataset.key]=Number(el.value));
    }else if(node?.algorithm==="calibration"){
      const curve=$("split-photo-curve").value;p.curve=curve==="custom"?"identity":curve;p.lut_path=curve==="custom"?$("split-lut").value.trim()||null:null;
      p.input_semantics=$("split-semantics").value;p.input_max=Number($("split-photo-max").value);p.gamma=Number($("split-photo-gamma").value);p.black_left=Number($("split-black-left").value);p.black_right=Number($("split-black-right").value);p.saturation_code=$("split-saturation").value===""?null:Number($("split-saturation").value);
      this.record.customLutNodes=(this.record.customLutNodes||[]).filter(id=>id!==node.id);if(curve==="custom")this.record.customLutNodes.push(node.id);this.photoPreview();
    }else if(node?.algorithm==="statistics"){p.normalization=$("split-normalization").value;
    }else if(node?.algorithm==="correlation"){
      for(const[id,key]of[["split-window","window"],["split-stride","stride"],["split-lag-min","lag_min"],["split-lag-max","lag_max"],["split-lag-step","lag_step"],["split-min-mean","min_mean"],["split-min-samples","min_samples"]])p[key]=Number($(id).value);
      const lag=$("split-lag-list").value.trim();p.lag_list=lag?lag.split(/[,，;\s]+/).map(Number):null;p.map_lag=Number($("split-map-lag").value);p.analysis=this.record.analysis||"map";
    }
    const corr=this.activeNode("correlation");if(corr)corr.parameters.roi_estimator=$("split-roi-estimator").value;
    c.backend=$("split-backend").value;c.output_dtype=$("split-output-dtype").value;c.memory_mb=Number($("split-memory").value);c.start=Number($("split-range-start").value)-1;c.end=Number($("split-range-end").value)-1;c.center=Number($("split-center").value)-1;
    this.flow.sync();
  }
  windowNote() {
    if(!this.cfg) return;
    if(!this.activeNode("correlation")){$("split-window-note").textContent="图像模块按所选帧范围执行；ROI 在完整帧处理后裁剪。";return;}
    const c=this.taskConfig(this.record.analysis==="curve"?"curve":"trial"),h=Math.floor(c.window/2),lo=c.lag_list?Math.min(...c.lag_list):c.lag_min,hi=c.lag_list?Math.max(...c.lag_list):c.lag_max;
    const first=c.start+h+Math.max(0,-lo),last=(c.end??this.context.source.meta.frame_count)-1-h-Math.max(0,hi);
    $("split-window-note").textContent=`左路窗口：帧 ${c.center-h+1}～${c.center+h+1} · 右路覆盖：${c.center-h+lo+1}～${c.center+h+hi+1} · 公共有效中心：${first+1}～${last+1}`;
  }
  async ensureGeometry() {
    if(!this.cfg||this.snapshots)return;
    try{await this.updateGeometry(!this.record.previewInitialized&&!!this.activeNode("alignment"));}catch(error){this.previewError(error);await this.showRaw(this.rawSnapshot||this.context.input);}
  }
  async updateGeometry(suggest=false,calibrate=false) {
    if(!this.cfg) throw Error("请先导入并选择灰度序列");
    this.readFields();
    const generation=++this.generation;
    const g=this.cfg.geometry;
    const response=await api("split/geometry",{source_id:this.sourceId,index:Math.max(0,Math.min(this.context.source.meta.frame_count-1,this.cfg.center)),...(suggest?{midline:g?.midline||[this.context.source.meta.shape[1]/2,this.context.source.meta.shape[1]/2],threshold:Number($("split-black-threshold").value)}:{geometry:g}),photometry:this.cfg.photometry,calibrate,workflow_config:!suggest?{workflow:this.cfg.workflow}:null});
    if(generation!==this.generation) return;
    const changed=JSON.stringify(this.snapshots?.geometry||this.cfg.geometry)!==JSON.stringify(response.geometry);
    this.cfg.geometry=response.geometry;const alignment=this.activeNode("alignment");if(alignment)alignment.parameters=response.geometry;this.record.previewInitialized=true;this.rawSnapshot=response.raw;this.previewError(null);
    if(changed) {this.record.accepted=null;this.cfg.roi=null;this.cfg.points=[];this.record.pointsAuto=true;}
    this.snapshots=response;
    if(response.mapping&&this.record.pointsAuto&&!this.cfg.points.length) this.cfg.points=[[Math.floor(response.mapping.shape[1]/2),Math.floor(response.mapping.shape[0]/2)]];
    if(response.proposal) {
      this.proposal=response.proposal;
      $("split-apply-proposal").disabled=false;
      $("split-quality").textContent=`建议偏移 X ${number(this.proposal.dx)}，Y ${number(this.proposal.dy)} · 峰值/RMS ${number(this.proposal.peak_to_rms)}。应用后检查叠加，再接受标定。`;
    }
    this.fill();this.changed();
    if(!["result","curve"].includes(this.view)) {if(response[this.view])await this.showView(this.view);else await this.showRaw(response.raw);}
    return response;
  }
  async setStage(stage) {
    if(stage==="result"){$("split-results-tools").open=true;return;}
    const algorithm={geometry:"alignment",preprocess:"calibration",trial:"correlation"}[stage];
    const node=this.cfg?.workflow?.find(n=>n.algorithm===algorithm);if(node){this.record.selectedModule=node.id;this.fill();}
    if(stage==="trial"){$("split-run-context").open=true;if(this.snapshots?.left)await this.showView("left");}
    if(stage==="geometry"&&this.rawSnapshot)await this.showRaw(this.rawSnapshot);
    if(stage==="flow")this.flow.render();this.bridge.changed();
  }
  async showView(view) {
    if(view==="result") return this.showSlice();
    if(view==="curve") return this.showCurve();
    if(view==="statistics") return this.showStatistics();
    if(view==="raw")return this.showRaw(this.rawSnapshot||this.context.input);
    if(!this.snapshots?.[view])await this.updateGeometry();
    if(!this.snapshots?.[view])throw Error("该预览不可用，请检查当前流程并更新预览");
    this.view=view;$("split-view").value=view;$("split-empty").hidden=true;
    const snapshot=this.snapshots[view];
    this.scene.width=snapshot.base_shape[1];this.scene.height=snapshot.base_shape[0];
    this.scene.roi=$("split-tool").value==="roi"&&view!=="raw"?clone(this.cfg.roi):null;this.autoFit=true;this.scene.zoom=1;this.scene.panX=this.scene.panY=0;
    await this.viewer.set(snapshot);this.setTool($("split-tool").value);this.draw();
    $("split-view-caption").textContent=`${snapshot.shape[1]} × ${snapshot.shape[0]} · ${view==="raw"?"传感器坐标":"左幅参考坐标"}`;
  }
  setTool(tool) {
    if(!this.cfg)return;
    if(["left","right","midline"].includes(tool)&&!this.activeNode("alignment")){this.bridge.report("当前流程没有分幅模块",true);return;}
    this.scene.tool=["left","right","roi"].includes(tool)?"roi":"pan";
    if(["left","right"].includes(tool)) {
      if(this.view!=="raw") {this.showView("raw").catch(e=>this.bridge.report(e.message,true));return;}
      this.scene.roi=clone(this.cfg.geometry[tool]);
    } else if(tool==="roi") {
      if(this.snapshots?.left&&["raw","curve","statistics"].includes(this.view)) {this.showView("left").catch(e=>this.bridge.report(e.message,true));return;}
      this.scene.roi=clone(this.cfg.roi);
    } else {
      this.scene.roi=null;
      if(tool==="midline"&&this.view!=="raw") {this.showView("raw").catch(e=>this.bridge.report(e.message,true));return;}
      if(tool==="point"&&["raw","curve","statistics"].includes(this.view)) {this.showView("left").catch(e=>this.bridge.report(e.message,true));return;}
    }
    this.draw();
  }
  commitROI() {
    if(!this.cfg||!this.scene.roi) return;
    const role=$("split-tool").value;
    if(["left","right"].includes(role)) {this.cfg.geometry[role]=clone(this.scene.roi);this.fill();this.changed(true);}
    else if(role==="roi") {
      this.cfg.roi=clone(this.scene.roi);
      if(this.record.pointsAuto) this.cfg.points=[];
      this.fillPoints();this.changed();
    }
  }
  draw() {
    if(this.autoFit)this.fitTransform();
    this.viewer.draw();
    if(!this.viewer.snapshot||!this.cfg?.geometry||!this.activeNode("alignment"))return;
    const ctx=$("split-canvas").getContext("2d"),s=this.viewer.scale();
    ctx.lineWidth=1.5/s;ctx.font=`${11/s}px Segoe UI`;
    if(this.view==="raw") {
      for(const [role,color] of [["left","#f1a3a0"],["right","#8bddbd"]]) {
        const q=this.cfg.geometry[role];ctx.strokeStyle=color;ctx.fillStyle=color;
        ctx.strokeRect(q.x,q.y,q.width,q.height);ctx.fillText(role==="left"?"L":"R",q.x+4/s,q.y+14/s);
      }
      const line=this.cfg.geometry.midline;ctx.strokeStyle="#d9c483";ctx.fillStyle="#d9c483";ctx.setLineDash([5/s,4/s]);
      ctx.beginPath();ctx.moveTo(line[0],0);ctx.lineTo(line[1],this.scene.height-1);ctx.stroke();ctx.setLineDash([]);
      for(const [x,y] of [[line[0],0],[line[1],this.scene.height-1]]) ctx.fillRect(x-4/s,y-4/s,8/s,8/s);
    } else if(this.view!=="curve") {
      if(this.cfg.roi&&this.scene.tool!=="roi") {
        const q=this.cfg.roi;ctx.strokeStyle="#9edfe9";ctx.setLineDash([5/s,3/s]);ctx.strokeRect(q.x,q.y,q.width,q.height);ctx.setLineDash([]);
      }
      ctx.strokeStyle="#9edfe9";ctx.fillStyle="#9edfe9";
      for(const [i,[x,y]] of this.cfg.points.entries()) {ctx.beginPath();ctx.moveTo(x-4/s,y+.5);ctx.lineTo(x+4/s,y+.5);ctx.moveTo(x+.5,y-4/s);ctx.lineTo(x+.5,y+4/s);ctx.stroke();ctx.fillText(String(i+1),x+5/s,y-5/s);}
    }
  }
  fitTransform() {
    this.scene.zoom=1;this.scene.panX=this.scene.panY=0;
    if(this.view==="result"&&this.viewer.snapshot) {
      const frame=this.viewer.snapshot,r=$("split-canvas").getBoundingClientRect();
      const target=Math.min((r.width-40)/frame.shape[1],(r.height-50)/frame.shape[0]);
      this.scene.zoom=Math.min(30,target/this.viewer.scale());
      const scale=this.viewer.scale(),[x,y]=frame.origin;
      this.scene.panX=(this.scene.width/2-x-frame.shape[1]/2)*scale;
      this.scene.panY=(this.scene.height/2-y-frame.shape[0]/2)*scale;
    }
  }
  fit() {this.autoFit=true;this.draw();}
  async seek(index) {
    if(!this.cfg)return;if(!Number.isInteger(index)||index<0||index>=this.context.source.meta.frame_count)throw Error("帧号超出来源范围");
    const wanted=this.view,id=this.sourceId,generation=++this.generation,transform={zoom:this.scene.zoom,panX:this.scene.panX,panY:this.scene.panY},shape=[this.scene.width,this.scene.height],autoFit=this.autoFit;this.cfg.center=index;$("split-center").value=index+1;$("split-scrubber").value=index+1;
    const raw=await api(`sources/${id}/frame`,{action:"index",index});if(id!==this.sourceId||generation!==this.generation)return;
    this.snapshots=null;await this.showRaw(raw);if(wanted!=="raw")try{await this.updateGeometry();if(this.snapshots?.[wanted])await this.showView(wanted);}catch(error){this.previewError(error);}
    if(this.view===wanted&&this.scene.width===shape[0]&&this.scene.height===shape[1]){Object.assign(this.scene,transform);this.autoFit=autoFit;this.draw();}
    this.windowNote();this.changed();
  }
  stop() {clearTimeout(this.playTimer);this.playing=null;$("split-play").textContent="播放";$("split-result-play").textContent="播放结果";}
  async play(result) {
    if(this.playing) {this.stop();return;}
    if(result&&!this.resultData?.run.array_shape) throw Error("请选择时间序列系数图");
    this.playing=result?"result":"input";
    const button=$(result?"split-result-play":"split-play");button.textContent="暂停";
    const mode=this.playing;
    const next=async()=>{
      if(this.playing!==mode) return;
      try {
        if(result) {if((this.timeIndex||0)+1>=this.resultData.run.array_shape[0]){this.stop();return;}this.timeIndex=(this.timeIndex||0)+1;$("split-result-time").value=this.timeIndex;await this.showSlice();}
        else {if(this.cfg.center+1>=this.context.source.meta.frame_count){this.stop();return;}await this.seek(this.cfg.center+1);}
      } catch(e){this.stop();this.bridge.report(e.message,true);return;}
      if(this.playing===mode) this.playTimer=setTimeout(next,90);
    };
    await next();
  }
  async budget(task) {
    this.readFields();
    if(!this.activeNode("correlation")){$("split-budget-info").textContent="逐帧模块流程：使用选定的帧范围，完整帧处理后裁剪 ROI。";return null;}
    const estimate=await api("split/preflight",{source_id:this.sourceId,config:this.taskConfig(task),task});
    $("split-budget-info").textContent=`公共中心帧 ${estimate.first_center+1}～${estimate.last_center+1}\n${estimate.window_count} 个窗口 × ${estimate.lags.length} 个延迟\n输出系数与计数：${bytes(estimate.output_bytes)}\n工作数组估算：${bytes(estimate.working_array_bytes)}\n完整背景：${estimate.full_source_frames.toLocaleString()} 帧（缓存可复用）`;
    $("split-budget-info").textContent+=`\n单遍配对更新约 ${estimate.pair_update_upper_estimate.toLocaleString()} 次 × ${estimate.region.width*estimate.region.height} 像素${task==="curve"&&this.cfg.roi&&this.cfg.roi_estimator==="pixel_mean"?"（ROI 空间平均使用两遍）":""}`;
    return estimate;
  }
  async submit(task) {
    if(!this.accepted()) throw Error("请先检查分幅与配准，并接受当前固定标定");
    this.readFields();
    if(!this.accepted()) throw Error("标定参数已变更，请更新预览并重新接受");
    if(!this.activeNode("correlation")&&this.activeNode("statistics"))task="statistics";
    if(task!=="statistics"&&this.activeNode("correlation"))await this.budget(task);
    const error=this.flow.validate(this.taskConfig(task).workflow);if(error)throw Error(error);
    const signature=this.signature();
    const response=await api("split/jobs",{source_id:this.sourceId,config:this.taskConfig(task),name:$("split-job-name").value.trim()||null,task,output_dir:this.context.output,config_revision:signature,calibration_index:this.record.calibrationIndex??null});
    this.requested.set(response.id,signature);
    this.bridge.report(`${label[task]}已加入后台任务；可继续预览或取消`);
    await this.bridge.poll();
  }
  updateJobs(jobs) {
    this.jobs=jobs;
    const records=jobs.filter(j=>j.mode==="split"&&j.status==="succeeded");
    const selected=this.resultId||$("split-result-select").value;
    $("split-result-select").innerHTML='<option value="">选择已保存结果</option>'+records.map(j=>`<option value="${esc(j.id)}">${esc(label[j.task]||"关联")} · ${esc(j.name)} · ${esc(j.id.slice(0,6))}</option>`).join("");
    if(records.some(j=>j.id===selected)) $("split-result-select").value=selected;
    const current=records.find(j=>j.id===this.resultId);
    if(current&&this.resultData) {
      this.resultData.run.display_name=current.name;
      $("split-result-title").textContent=current.name;
      if(document.activeElement!==$("split-result-name"))$("split-result-name").value=current.name;
    }
  }
  async completed(job) {
    this.updateJobs(this.jobs);
    if(job.status!=="succeeded") {if(job.error)this.bridge.report(job.error,true);return;}
    if(this.sourceId!==job.source_ids[0]||this.mode!=="split_correlation") {this.bridge.report(`${label[job.task]||"模块处理"}完成，结果保存在任务记录`);return;}
    if(this.requested.get(job.id)!==this.signature()) {this.bridge.report("关联完成；当前配置已变更，原结果保留在结果列表");return;}
    if(job.mode!=="split") {
      await this.loadFrameResult(job.id);this.bridge.report("模块处理已完成");return;
    } else if(job.task==="statistics") {
      this.record.statisticsJob=job.id;
      $("split-stat-status").textContent=`完整全源统计已保存：${job.run.statistics.frames.toLocaleString()} 帧 · 固定尺度 ${job.run.statistics.scales.map(number).join(" / ")}`;
      await this.loadResult(job.id,false);this.record.selectedModule=this.activeNode("statistics")?.id;this.fill();
    } else await this.loadResult(job.id);
    this.bridge.report(`${label[job.task]}已完成`);
  }
  async showStatistics() {
    const id=this.record?.statisticsJob||this.resultId;
    if(!id)throw Error("请先计算完整全源统计，或完成一次关联任务");
    const data=await api(`split/jobs/${id}/statistics`,{view:$("split-stat-view").value,index:this.cfg?.center??0});
    this.resultView="statistics";$("split-curve").hidden=true;$("split-result-empty").hidden=true;
    this.resultScene.width=data.frame.base_shape[1];this.resultScene.height=data.frame.base_shape[0];
    await this.resultViewer.set(data.frame);this.fitResult();
    const current=this.cfg?this.taskConfig("statistics"):null;
    const upstream=cfg=>(cfg.workflow||[]).filter(n=>n.enabled).slice(0,(cfg.workflow||[]).filter(n=>n.enabled).findIndex(n=>n.algorithm==="statistics")+1);
    const canonical=v=>Array.isArray(v)?v.map(canonical):v&&typeof v==="object"?Object.fromEntries(Object.keys(v).sort().map(k=>[k,canonical(v[k])])):v;
    const compatible=current&&data.source_id===this.sourceId&&JSON.stringify(canonical(upstream(data.config)))===JSON.stringify(canonical(upstream(current)));
    $("split-stat-status").textContent=`${compatible?"完整全源统计已保存":"历史全源统计"}：${data.frames.toLocaleString()} 帧 · 固定尺度 L ${number(data.scales[0])} / R ${number(data.scales[1])}`;
    $("split-result-caption").textContent=$("split-stat-view").selectedOptions[0].textContent;
    $("split-result-title").textContent=this.resultData?.run.display_name||"全源统计";
  }
  async loadResult(id,switchStage=true) {
    if(!id) return;
    this.stop();this.resultId=id;
    if(this.record) this.record.resultId=id;
    const known=this.jobs.find(j=>j.id===id);if(known&&known.mode!=="split")return this.loadFrameResult(id);
    const data=await api(`split/jobs/${id}`);
    if(this.resultId!==id) return;
    const saved=this.record?.resultViews?.[id];
    this.resultData=data;this.timeIndex=Math.max(0,Math.min((data.run.array_shape?.[0]||1)-1,saved?.timeIndex||0));
    this.lagIndex=Math.max(0,Math.min(data.run.lags.length-1,saved?.lagIndex??Math.max(0,data.run.lags.indexOf(0))));
    if(["time","frame"].includes(saved?.axis))$("split-curve-axis").value=saved.axis;
    if(this.record&&data.run.statistics?.frames) this.record.statisticsJob=id;
    $("split-result-select").value=id;
    $("split-result-name").value=data.run.display_name||id;$("split-result-title").textContent=data.run.display_name||label[data.run.task];
    $("split-result-summary").textContent=`${label[data.run.task]}${data.run.result_semantics==="processed_g2"?" · g² 后处理":""} · ${data.run.numeric_semantics==="sensor_codes"?"码值关联":"校准 / 确认强度"} · W=${data.run.config.window} · τ ${data.run.lags[0]}～${data.run.lags.at(-1)} 帧\n${data.directory}`;
    const nav=$("split-result-nav");nav.hidden=data.run.task==="statistics";
    $("split-result-time").max=(data.run.array_shape?.[0]||1)-1;$("split-result-time").value=this.timeIndex;
    $("split-result-time").disabled=!data.run.array_shape;
    $("split-result-play").disabled=!data.run.array_shape||data.run.array_shape[0]<2;
    $("split-result-lag").innerHTML=data.run.lags.map((q,i)=>`<option value="${i}">${q>=0?"+":""}${q} 帧</option>`).join("");
    $("split-result-lag").value=this.lagIndex;
    $("split-promote").disabled=!data.run.array_shape;
    $("split-show-map").disabled=!data.run.array_shape;
    $("split-show-curve").disabled=!data.curve_result;
    $("split-show-map").hidden=!data.run.array_shape;$("split-show-curve").hidden=!data.curve_result;
    document.querySelectorAll("[data-split-export]").forEach(b=>{const format=b.dataset.splitExport;b.disabled=["csv","png","svg","curve_json"].includes(format)?!data.curve_result:["image","windows"].includes(format)?!data.run.array_shape:false;});
    if(switchStage) await this.setStage("result");
    if(data.curve_result) this.showCurve();
    else if(data.run.array_shape) await this.showSlice();
    else await this.showStatistics();
    this.updateStatus();this.bridge.changed();
  }
  async loadFrameResult(id) {
    const job=await api(`jobs/${id}`);let frame=job.result;
    if(!frame){const state=await api("state"),source=state.sources.find(s=>s.role==="result"&&s.run_id===id);if(source)frame=await api(`sources/${source.id}/frame`,{action:"open"});}
    if(!frame)throw Error("该逐帧结果的缓存不可用，请从处理结果打开已保存序列");
    this.resultId=id;this.resultData=null;if(this.record)this.record.resultId=id;this.resultView="frame";$("split-curve").hidden=true;$("split-result-empty").hidden=true;$("split-result-nav").hidden=true;$("split-result-title").textContent=job.name;
    this.resultScene.width=frame.base_shape[1];this.resultScene.height=frame.base_shape[0];await this.resultViewer.set(frame);this.fitResult();$("split-result-caption").textContent=`逐帧模块结果 · ${frame.shape[1]} × ${frame.shape[0]}`;$("split-result-note").textContent="完整运行的序列与来源记录保存在处理结果中。";
    $("split-result-summary").textContent=job.summary||"逐帧模块处理";$("split-result-name").value=job.name;$("split-promote").disabled=true;$("split-show-map").hidden=true;$("split-show-curve").hidden=true;document.querySelectorAll("[data-split-export]").forEach(b=>b.disabled=true);
  }
  saveResultView() {
    if(!this.record||!this.resultId) return;
    this.record.resultViews??={};
    this.record.resultViews[this.resultId]={timeIndex:this.timeIndex||0,lagIndex:this.lagIndex||0,axis:$("split-curve-axis").value};
    const keys=Object.keys(this.record.resultViews);
    if(keys.length>64)delete this.record.resultViews[keys[0]];
    this.bridge.changed();
  }
  async showSlice() {
    if(!this.resultId||!this.resultData?.run.array_shape)throw Error("该结果没有系数图；τ 扫描曲线可独立运行");
    const id=this.resultId,ti=this.timeIndex||0,qi=this.lagIndex||0;
    const data=await api(`split/jobs/${id}/slice`,{time_index:ti,lag_index:qi,display:{low:Number($("split-display-low").value),high:Number($("split-display-high").value),gamma:1,colormap:"heat"}});
    if(this.resultId!==id||ti!==(this.timeIndex||0)||qi!==(this.lagIndex||0))return;
    const fit=this.resultView!=="result"||this.displayResultId!==id;
    this.resultView="result";$("split-curve").hidden=true;$("split-result-empty").hidden=true;
    this.resultScene.width=data.frame.base_shape[1];this.resultScene.height=data.frame.base_shape[0];
    await this.resultViewer.set(data.frame);this.displayResultId=id;if(fit)this.fitResult();else this.drawResult();
    const ref=data.window_ref;
    $("split-result-caption").textContent=`中心帧 ${ref.center_index+1} · τ ${ref.lag_frames>=0?"+":""}${ref.lag_frames} 帧 · ${data.valid_pixels.toLocaleString()} 有效像素`;
    $("split-result-note").textContent=`左窗 ${ref.left_start+1}～${ref.left_end} · 右窗 ${ref.right_start+1}～${ref.right_end} · 实际时差 ${number(ref.lag_seconds)} s · 有效样本 ${data.count_min}～${data.count_max}`;
    $("split-result-lag").value=qi;this.saveResultView();
  }
  showCurve() {
    const data=this.resultData?.curve_result;
    if(!data) {this.bridge.report("选定中心帧和点 / ROI，点击“固定帧曲线”计算该结果",true);return;}
    this.resultView="curve";$("split-curve").hidden=false;$("split-result-empty").hidden=true;
    const refs=data.window_refs,curves=data.curves,time=$("split-curve-axis").value==="time";
    const available=refs.every(r=>r.lag_seconds!=null);
    $("split-curve-axis").querySelector('option[value="time"]').disabled=!available;
    if(time&&!available) {$("split-curve-axis").value="frame";return this.showCurve();}
    const rawX=refs.map(r=>time?r.lag_seconds:r.lag_frames),factor=time&&Math.max(...rawX.map(Math.abs))<.01?1e6:1;
    const xs=rawX.map(x=>x*factor),finite=curves.flatMap(c=>c.g2.filter(v=>v!=null));
    const reference=data.coefficient_reference===undefined?1:data.coefficient_reference;
    const bounds=finite.concat(reference==null?[]:[reference]);if(!bounds.length)bounds.push(1);
    let lo=Math.min(...bounds),hi=Math.max(...bounds);if(hi-lo<1e-10){lo-=.05;hi+=.05;}else{const pad=(hi-lo)*.12;lo-=pad;hi+=pad;}
    let xmin=Math.min(...xs),xmax=Math.max(...xs);if(xmin===xmax){xmin-=1;xmax+=1;}
    const X=x=>72+(x-xmin)/(xmax-xmin)*620,Y=y=>350-(y-lo)/(hi-lo)*306;
    const colors=["#87dfc5","#e6b386","#9db4f3","#dc9dd9","#82cadd"];
    const parts=['<svg viewBox="0 0 720 425" role="img" aria-label="g² 随正负延迟变化曲线">'];
    for(let i=0;i<=5;i++) {const x=xmin+(xmax-xmin)*i/5;parts.push(`<line x1="${X(x)}" y1="44" x2="${X(x)}" y2="350" stroke="#25384a"/><text x="${X(x)}" y="372" text-anchor="middle" fill="#9ab0c8" font-size="11">${esc(number(x))}</text>`);}
    for(let i=0;i<=4;i++) {const y=lo+(hi-lo)*i/4;parts.push(`<line x1="72" y1="${Y(y)}" x2="692" y2="${Y(y)}" stroke="#25384a"/><text x="63" y="${Y(y)+4}" text-anchor="end" fill="#9ab0c8" font-size="11">${esc(number(y))}</text>`);}
    if(xmin<=0&&xmax>=0) parts.push(`<line x1="${X(0)}" y1="44" x2="${X(0)}" y2="350" stroke="#7597ad" stroke-dasharray="4 5"/>`);
    if(reference!=null)parts.push(`<line x1="72" y1="${Y(reference)}" x2="692" y2="${Y(reference)}" stroke="#7597ad" stroke-dasharray="4 5"/>`);
    for(const [i,c] of curves.entries()) {
      let path="",connected=false;
      c.g2.forEach((v,j)=>{if(v==null){connected=false;return;}path+=`${connected?"L":"M"}${X(xs[j])},${Y(v)} `;connected=true;});
      parts.push(`<path d="${path}" stroke="${colors[i%colors.length]}" fill="none" stroke-width="1.7"/>`);
      if(i<7||curves.length===8&&i===7) parts.push(`<text x="${80+(i%4)*150}" y="${18+Math.floor(i/4)*15}" fill="${colors[i%colors.length]}" font-size="10">${esc(c.name)}${c.estimator==="pixel_mean"?" · 像素平均":c.estimator==="intensity_sum"?" · 总强度":""}</text>`);
    }
    if(curves.length>8) parts.push(`<text x="530" y="33" fill="#aac2d8" font-size="10">共 ${curves.length} 条，数值见右侧</text>`);
    const selected=Math.max(0,Math.min(refs.length-1,this.lagIndex||0));
    parts.push(`<text x="382" y="405" text-anchor="middle" fill="#aac2d8" font-size="12">${time?`实际延迟 τ (${factor===1e6?"μs":"s"})`:"有符号帧延迟 q（右路 i+q）"}</text><text x="22" y="198" fill="#aac2d8" font-size="13">g²</text><line id="split-curve-cursor" x1="${X(xs[selected])}" x2="${X(xs[selected])}" y1="44" y2="350" stroke="#c7ddec"/></svg>`);
    $("split-curve").innerHTML=parts.join("");
    const svg=$("split-curve").querySelector("svg");
    const hover=e=>{
      const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(svg.getScreenCTM().inverse());
      const target=xmin+(p.x-72)/620*(xmax-xmin);
      let idx=0;xs.forEach((v,j)=>{if(Math.abs(v-target)<Math.abs(xs[idx]-target))idx=j;});
      const cursor=$("split-curve-cursor");cursor.setAttribute("x1",X(xs[idx]));cursor.setAttribute("x2",X(xs[idx]));cursor.setAttribute("visibility","visible");
      $("split-curve-readout").textContent=`中心帧 ${refs[idx].center_index+1} · q=${refs[idx].lag_frames>=0?"+":""}${refs[idx].lag_frames} · τ=${number(refs[idx].lag_seconds)} s\n${curves.map(c=>`${c.name}：${coefficient(c.g2[idx])} (${c.pixel_count} 像素，有效样本 ${c.counts_min[idx]}～${c.counts_max[idx]})`).join("\n")}`;
      return idx;
    };
    svg.onpointermove=hover;svg.onclick=e=>{this.lagIndex=hover(e);$("split-result-lag").value=this.lagIndex;this.saveResultView();};
    $("split-result-caption").textContent=`固定中心帧 ${refs[0].center_index+1} · W=${data.config.window} · ${refs.length} 个有符号延迟${data.result_semantics==="processed_g2"?" · g² 后处理":""}`;
    const ref=refs[selected];
    $("split-result-note").textContent=`左窗 ${ref.left_start+1}～${ref.left_end} · 右窗 ${ref.right_start+1}～${ref.right_end} · 实际时差 ${number(ref.lag_seconds)} s`;
    $("split-curve-readout").textContent=`${curves.map(c=>c.name+" · "+(c.estimator==="pixel_mean"?"逐像素 g² 空间平均":c.estimator==="intensity_sum"?"先汇总强度再算 g²":"单像素 g²")).join("\n")}\n选定 q=${ref.lag_frames}：${curves.map(c=>coefficient(c.g2[selected])).join(" / ")}\n曲线使用完整 W 帧窗口，鼠标移动查看数值。`;
    this.saveResultView();
  }
  async jobDetail(job) {
    if(!$("job-dialog").open)$("job-dialog").showModal();
    $("job-detail").innerHTML=detailHTML(job)+(job.status==="succeeded"?'<button id="split-job-open" class="primary">在处理结果视窗中打开</button>':"");
    bindRename(job,async()=>{await this.bridge.poll();await this.jobDetail(await api(`jobs/${job.id}`));},this.bridge.report);
    if($("split-job-open"))$("split-job-open").onclick=this.wrap(async()=>{await this.bridge.activate();await this.loadResult(job.id);$("job-dialog").close();});
  }

  taskConfig(task) {
    this.flow.sync();const c=clone(this.cfg);
    if(task==="trial"||task==="run") {
      const lag=this.record.mapLag??0;c.lag_list=[lag];c.lag_min=c.lag_max=lag;c.lag_step=1;c.points=[];
    }
    if(task==="statistics"){const index=c.workflow.findIndex(n=>n.enabled&&n.algorithm==="statistics");c.workflow=c.workflow.slice(0,index+1);c.points=[];c.roi=null;c.window=1;c.min_samples=1;c.lag_list=[0];c.lag_min=c.lag_max=0;c.lag_step=1;}
    return c;
  }
  setAnalysis(kind,save=true) {
    if(!this.record)return;
    this.record.analysis=kind;if(save&&this.activeNode("correlation"))this.activeNode("correlation").parameters.analysis=kind;
    document.querySelectorAll("[data-split-analysis]").forEach(b=>b.classList.toggle("selected",b.dataset.splitAnalysis===kind));
    const curve=kind==="curve";
    $("split-curve-lags").hidden=!curve;$("split-curve-lag-list-field").hidden=!curve;
    $("split-map-lag-field").hidden=curve;$("split-stride").closest("label").hidden=curve;
    $("split-curve-run").hidden=!curve;$("split-trial").hidden=curve;$("split-run").hidden=curve;
    $("split-analysis-note").textContent=curve?"固定中心帧与点 / ROI，独立扫描负、零、正 τ 并输出曲线。":"固定一个 τ，对共同有效区或 ROI 计算随时间变化的系数图。";
    $("split-job-name").value=this.record.names?.[kind]||"";
    if(save)this.changed();
  }
  fitResult() {
    if(!this.resultViewer?.snapshot)return;
    this.resultAutoFit=true;
    const scene=this.resultScene,frame=this.resultViewer.snapshot;
    scene.zoom=1;scene.panX=scene.panY=0;
    const box=$("split-result-canvas").getBoundingClientRect();
    if(box.width<1||box.height<1)return;
    const target=Math.min((box.width-40)/frame.shape[1],(box.height-50)/frame.shape[0]);
    scene.zoom=Math.max(.08,Math.min(30,target/this.resultViewer.scale()));
    const scale=this.resultViewer.scale(),[x,y]=frame.origin;
    scene.panX=(scene.width/2-x-frame.shape[1]/2)*scale;
    scene.panY=(scene.height/2-y-frame.shape[0]/2)*scale;this.resultViewer.draw();
  }
  drawResult() {
    if(!this.resultViewer)return;
    if(this.resultAutoFit)this.fitResult();else this.resultViewer.draw();
  }
  photoPreview() {
    const curve=$("split-photo-curve").value,max=Number($("split-photo-max").value),gamma=Number($("split-photo-gamma").value);
    $("split-lut").disabled=curve!=="custom";$("split-photo-gamma").disabled=curve!=="gamma";
    const formulas={identity:"y = x（原值）",linear:"y = x / 固定输入尺度；不截断、不逐帧拉伸",sqrt:"y = √(x / 固定输入尺度)",square:"y = (x / 固定输入尺度)²",gamma:`y = (x / 固定输入尺度)^${gamma}`,log:"y = ln(1 + 9x / 固定输入尺度) / ln(10)",custom:"按外部 LUT 逐码值转换"};
    $("split-lut-formula").textContent=formulas[curve]+(["sqrt","square","gamma","log"].includes(curve)?"；非线性变换会改变强度涨落及 g²。":"");
    const f=x=>curve==="sqrt"?Math.sqrt(x):curve==="square"?x*x:curve==="gamma"?x**gamma:curve==="log"?Math.log1p(9*x)/Math.log(10):x;
    const points=Array.from({length:65},(_,i)=>{const x=i/64;return `${20+210*x},${90-75*f(x)}`;}).join(" ");
    $("split-lut-preview").innerHTML=`<svg viewBox="0 0 260 110" role="img" aria-label="强度预设曲线"><path d="M20 12V90H240" fill="none" stroke="#4a657b"/><path d="M20 90L230 15" stroke="#3b5265" stroke-dasharray="4 4"/><polyline points="${points}" fill="none" stroke="#8ee0c6" stroke-width="2"/><text x="20" y="104" font-size="9" fill="#9db4c7">0</text><text x="195" y="104" font-size="9" fill="#9db4c7">${curve==="identity"?"原值":max}</text></svg>`;
  }

}
