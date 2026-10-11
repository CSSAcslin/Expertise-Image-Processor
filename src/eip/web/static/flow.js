import { api } from "./api.js";
const copy=v=>structuredClone(v);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const types={sensor:"单幅图像",pair:"对齐双路",moments:"双路与全源统计",map:"系数图"};
const labels={both:"双路 / 当前数据",left:"左路",right:"右路",identity:"原值",linear:"线性归一",sqrt:"平方根",square:"平方",gamma:"幂函数",log:"对数",none:"保留强度尺度",fixed_rms:"上游全序列固定 RMS",map:"固定 τ 系数图",curve:"τ 扫描曲线",nearest:"最近邻",bilinear:"双线性",sensor_codes:"保存码值",linear_intensity:"已确认线性强度",pixel_mean:"由空间模块决定",intensity_sum:"旧区域总强度（需迁移）",pixels:"保留区域矩阵",mean:"均值",sum:"总和",quantile:"分位值",samples:"当前样本号",seconds:"真实时间（秒）",joint:"当前样本两路共同有效像素",independent:"两路分别取有限像素（集合可不同）"};
const get=(o,p)=>p.split(".").reduce((v,k)=>v?.[k],o);
function set(o,p,v){const keys=p.split(".");for(const k of keys.slice(0,-1))o=o[k]??={};o[keys.at(-1)]=v;}

export function moduleNode(s,source){
  const p=copy(s.defaults);
  if(s.id==="alignment"&&source){const[h,w]=source.meta.shape,m=Math.max(1,Math.floor(w/2));Object.assign(p,{left:{x:0,y:0,width:m,height:h},right:{x:m,y:0,width:w-m,height:h},midline:[m,m]});}
  if(s.id==="correlation"&&source?.meta.frame_count){const n=source.meta.frame_count,q=Math.min(10,Math.max(0,Math.floor((n-3)/4)));let w=Math.min(101,Math.max(1,n-2*q));if(w%2===0)w--;Object.assign(p,{window:w,min_samples:Math.min(2,w),lag_min:-q,lag_max:q});}
  return {id:crypto.randomUUID(),algorithm:s.id,parameters:p,enabled:true,target:"both",version:s.version,manifest:s.manifest,code_hash:s.code_hash,execution:{backend:"cpu",memory_mb:256,precision:"native"}};
}

export class ModuleEditor{
  constructor(options){this.options=options;this.drag=null;this.selectedId=null;}
  get nodes(){return this.options.nodes()||[];}
  get specs(){return this.options.catalog()||{};}
  get selected(){return this.nodes.find(n=>n.id===(this.options.selected?.()??this.selectedId));}
  select(id){this.selectedId=id;this.options.select?.(id);}
  choose(id){this.select(id);this.render();this.options.inspect?.();this.options.selectionChanged?.();}
  normalize(){this.nodes.forEach(n=>{n.id??=crypto.randomUUID();n.enabled??=true;n.parameters??={};n.target??=n.parameters.target||"both";delete n.parameters.target;const s=this.specs[n.algorithm];n.version??=s?.version||"1.0";n.manifest??=null;n.code_hash??=null;});}
  validate(){return this.planKey===this.key()?this.plan?.errors?.[0]||null:null;}
  key(){return JSON.stringify([this.options.source?.()?.id,this.nodes]);}
  async compile(){
    const source=this.options.source?.();if(!source)return null;const key=this.key();
    try{const plan=await api("modules/plan",{source_id:source.id,steps:this.nodes});if(key!==this.key())return null;this.plan=plan;this.planKey=key;this.paintContracts();this.status(false);this.options.planned?.(plan);return plan;}
    catch(error){if(key===this.key()){this.plan={errors:[error.message],stages:[]};this.planKey=key;this.status(false);}return null;}
  }
  paintContracts(){
    const host=this.options.host();for(const card of host?.querySelectorAll("[data-instance]")||[]){const stage=this.plan?.stages?.find(s=>s.id===card.dataset.instance);card.classList.toggle("contract-error",!!stage?.error);const line=card.querySelector(".flow-type");if(stage)line.textContent=stage.error||`${stage.input.label} → ${stage.output.label}${stage.enabled?"":" · 原样通过"}`;}
    const node=this.selected,stage=this.plan?.stages?.find(s=>s.id===node?.id),info=this.options.inspector?.()?.parentElement.querySelector(".module-contract");const meanings={intensity:"光强",regional_mean_signal:"区域平均强度信号",regional_sum_signal:"区域总强度信号",regional_quantile_signal:"区域分位值信号",pixel_g2:"逐像素标准 g²",regional_mean_signal_g2:"区域平均信号的 g²",regional_sum_signal_g2:"区域总强度的 g²",regional_quantile_signal_g2:"区域分位值信号的 g²",pixel_g2_spatial_mean:"逐像素 g² 的空间平均",pixel_g2_spatial_sum:"逐像素 g² 的空间总和",pixel_g2_spatial_quantile:"逐像素 g² 的空间分位值"};if(info)info.textContent=stage?(stage.error||`输入 ${stage.input.label}\n输出 ${stage.output.label}\n含义：${meanings[stage.output.meaning]||stage.output.semantic}${stage.warning?"\n"+stage.warning:""}`):"正在检查输入输出形状…";
  }
  add(algorithm,index=this.nodes.length){if(this.nodes.length>=32)throw Error("处理流程最多 32 个模块");const s=this.specs[algorithm];if(!s)throw Error("请选择已注册模块");if(algorithm==="time_crop"&&this.nodes.some(n=>n.algorithm==="time_crop"))throw Error("时间裁剪只需在起点设置一次");const n=moduleNode(s,this.options.source?.());this.nodes.splice(algorithm==="time_crop"?0:index,0,n);this.select(n.id);this.changed();this.options.inspect?.();}
  move(a,b){if(b<0||b>=this.nodes.length)return;this.nodes.splice(b,0,this.nodes.splice(a,1)[0]);this.changed();}
  changed(){this.options.change?.();this.render();}
  field(p,n,i){
    const value=get(n.parameters,p.key)??p.default,attrs=`data-param="${i}" data-key="${esc(p.key)}" data-kind="${p.kind}"${p.nullable?' data-nullable="true"':""}`;let control;
    if(p.choices?.length)control=`<select ${attrs}>${p.choices.map(v=>`<option value="${esc(v)}" ${v===value?"selected":""}>${esc(labels[v]||v)}</option>`).join("")}</select>`;
    else if(p.kind==="bool")control=`<input ${attrs} type="checkbox" ${value?"checked":""}/>`;
    else control=`<input ${attrs} type="${["int","float"].includes(p.kind)?"number":"text"}" value="${esc(p.kind==="array"?(value?.join(", ")||""):value)}" ${p.minimum!=null?`min="${p.minimum}"`:""} ${p.maximum!=null?`max="${p.maximum}"`:""} step="${p.kind==="int"?1:"any"}"/>`;
    return `<label>${esc(p.label)}${p.unit?" · "+esc(p.unit):""}${control}</label>`;
  }
  render(){
    this.normalize();const host=this.options.host(),nodes=this.nodes;if(!host)return;
    if(!this.selected)this.select(nodes[0]?.id??null);
    host.innerHTML=nodes.length?nodes.map((n,i)=>{const s=this.specs[n.algorithm]||{name:n.algorithm,parameters:[],inputs:[],output:"same"},selected=this.selected?.id===n.id;
      return `<article class="flow-node step-card ${n.enabled?"":"disabled"} ${selected?"selected":""}" data-node="${i}" data-algorithm="${esc(n.algorithm)}" data-instance="${esc(n.id)}" draggable="true" tabindex="0" aria-label="${i+1}. ${esc(n.name||s.name)}"><div class="flow-node-title step-header"><span class="flow-grip" title="拖动排序">⠿</span><span class="mono">${i+1}</span><strong>${esc(n.name||s.name)}</strong><button data-up="${i}" class="text-button" aria-label="上移模块" ${i===0?"disabled":""}>↑</button><button data-down="${i}" class="text-button" aria-label="下移模块" ${i===nodes.length-1?"disabled":""}>↓</button></div><p class="flow-type">${s.inputs.map(t=>types[t]).join(" / ")} → ${s.output==="same"?"保持网格":types[s.output]}</p><button data-edit="${esc(n.id)}" class="quiet small" aria-pressed="${selected}">${selected?"正在编辑此实例":"编辑此模块参数"}</button><div class="flow-actions"><label><input type="checkbox" data-enable="${i}" ${n.enabled?"checked":""}/>启用</label>${s.output==="same"&&s.execution_scope==="frame"&&s.id!=="spatial_roi"?`<label class="flow-target">幅面<select data-target="${i}">${["both","left","right"].map(t=>`<option value="${t}" ${t===n.target?"selected":""}>${labels[t]}</option>`).join("")}</select></label>`:""}<button data-duplicate="${i}" class="text-button">复制</button><button data-delete="${i}" class="text-button">删除</button></div></article>`;
    }).join(""):'<p class="empty-steps">流程为空。可添加任意已注册模块；原始数据可照常预览。</p>';
    host.querySelectorAll("[data-node]").forEach(el=>{const i=Number(el.dataset.node),n=nodes[i];
      el.ondragstart=e=>{if(e.target.closest("input,select,button")){e.preventDefault();return;}this.drag=i;e.dataTransfer.effectAllowed="move";e.dataTransfer.setData("text/plain",String(i));};el.ondragover=e=>{e.preventDefault();el.classList.add("drop-target");};el.ondragleave=()=>el.classList.remove("drop-target");el.ondrop=e=>{e.preventDefault();if(this.drag!=null)this.move(this.drag,i);this.drag=null;};
      el.querySelector("[data-up]").onclick=()=>this.move(i,i-1);el.querySelector("[data-down]").onclick=()=>this.move(i,i+1);
      el.querySelector("[data-delete]").onclick=()=>{nodes.splice(i,1);this.changed();this.options.inspect?.(false);};el.querySelector("[data-duplicate]").onclick=()=>{if(nodes.length>=32)return;const d={...copy(n),id:crypto.randomUUID()};nodes.splice(i+1,0,d);this.select(d.id);this.changed();this.options.inspect?.();};
      el.querySelector("[data-enable]").onchange=e=>{n.enabled=e.target.checked;this.changed();};const target=el.querySelector("[data-target]");if(target)target.onchange=e=>{n.target=e.target.value;this.changed();};
      el.querySelector("[data-edit]").onclick=()=>this.choose(n.id);
      el.onclick=e=>{if(!e.target.closest("button,input,select,label"))this.choose(n.id);};
      el.onkeydown=e=>{if(e.target===el&&["Enter"," "].includes(e.key)){e.preventDefault();this.choose(n.id);}};
    });this.renderInspector();this.status();this.options.afterRender?.();
  }
  renderInspector(){
    const host=this.options.inspector?.();if(!host)return;const n=this.selected,s=this.specs[n?.algorithm];
    if(this.options.title)this.options.title().textContent=n?(n.name||s?.name||n.algorithm):"选择一个模块";
    if(this.options.instance)this.options.instance().textContent=n?`实例 ${n.id.slice(0,8)} · v${n.version||"1.0"}${n.enabled?"":" · 已停用"}`:"在模块流程中添加或选择算法，然后编辑该实例的参数。";
    host.hidden=!!n&&!!this.options.special?.(n);
    const fields=(s?.parameters||[]).filter(p=>!(n?.algorithm==="correlation"&&p.key==="roi_estimator")&&!(n?.algorithm==="time_crop"&&(n.parameters.unit==="seconds"?["first","last"]:["start","end"]).includes(p.key)));
    host.innerHTML=n?(host.hidden?"":`<p class="step-description">${esc(s?.description||"尚未注册的模块")}</p><div class="flow-fields step-parameters">${fields.map(p=>this.field(p,n,this.nodes.indexOf(n))).join("")}</div>`):"";
    host.querySelectorAll("[data-param]").forEach(input=>input.onchange=()=>{try{if(!input.reportValidity())return;const kind=input.dataset.kind;let v=kind==="bool"?input.checked:["int","float"].includes(kind)?Number(input.value):kind==="array"?(input.value.trim()?input.value.split(/[,，;\s]+/).map(Number):null):input.value;if(input.dataset.nullable&&input.value==="")v=null;set(n.parameters,input.dataset.key,v);this.options.change?.();this.status();if(n.algorithm==="time_crop"&&input.dataset.key==="unit")this.renderInspector();}catch(error){this.options.report?.(error.message,true);}});
    const parent=host.parentElement;parent.querySelector(".module-execution")?.remove();parent.querySelector(".module-contract")?.remove();parent.querySelector(".module-stage-tools")?.remove();
    if(n){
      const contract=document.createElement("pre");contract.className="module-contract";host.before(contract);
      const tools=document.createElement("div");tools.className="module-stage-tools split-buttons";tools.innerHTML=`<label>检查样本<input data-stage-index type="number" min="1" step="1" value="${(this.previewSample||0)+1}" aria-label="阶段检查样本号"/></label><button data-stage="input" class="quiet small">查看此步输入</button><button data-stage="output" class="quiet small">查看此步输出</button>${n.algorithm==="spatial_roi"?'<button data-roi-module class="primary small">在输入画布选区</button>':""}`;parent.append(tools);
      tools.querySelector("[data-stage-index]").onchange=e=>{if(e.target.reportValidity())this.previewSample=Number(e.target.value)-1;};
      tools.querySelectorAll("[data-stage]").forEach(b=>b.onclick=()=>this.options.preview?.(this,b.dataset.stage));tools.querySelector("[data-roi-module]")?.addEventListener("click",()=>this.options.drawROI?.(this));
      const e=n.execution??={backend:"cpu",memory_mb:256,precision:"native"},section=document.createElement("details");section.className="module-execution settings-section";section.open=true;section.innerHTML=`<summary>执行设置 · 当前实例</summary><div class="two-fields"><label>设备<select data-execution="backend"><option value="cpu">CPU</option><option value="auto">Auto</option>${n.algorithm==="correlation"?'<option value="gpu">GPU（需可用设备）</option>':""}</select></label><label>内存预算 MiB<input data-execution="memory_mb" type="number" min="16" max="4096" step="16" value="${e.memory_mb||256}"/></label><label>数值精度<select data-execution="precision"><option value="native">算法默认</option><option value="float64">float64</option><option value="float32">float32</option></select></label></div>`;parent.append(section);
      for(const el of section.querySelectorAll("[data-execution]")){el.value=e[el.dataset.execution]??(el.dataset.execution==="precision"?"native":el.dataset.execution==="backend"?"cpu":256);el.onchange=()=>{if(!el.reportValidity())return;e[el.dataset.execution]=el.dataset.execution==="memory_mb"?Number(el.value):el.value;this.options.change?.();this.status();};}
      this.paintContracts();
    }
  }
  status(recompile=true){const error=this.validate(),el=this.options.status?.();if(el){el.textContent=error||"按实际形状与顺序执行 · 运行前再次检查连接 · 原始预览始终可用";el.classList.toggle("error",!!error);}if(recompile){clearTimeout(this.compileTimer);this.compileTimer=setTimeout(()=>this.compile(),180);}return error;}
}

export class FlowEditor extends ModuleEditor{
  constructor(controller){
    super({nodes:()=>controller.cfg?.workflow,catalog:()=>controller.context.modules,source:()=>controller.context.source,host:()=>document.getElementById("split-flow-list"),status:()=>document.getElementById("split-flow-status"),inspector:()=>document.getElementById("split-module-fields"),title:()=>document.getElementById("split-module-title"),instance:()=>document.getElementById("split-module-instance"),selected:()=>controller.record?.selectedModule,select:id=>{if(controller.record)controller.record.selectedModule=id;},special:n=>["alignment","calibration"].includes(n.algorithm),preview:(editor,side)=>controller.bridge.previewModule?.(editor,side),drawROI:editor=>controller.bridge.drawModuleROI?.(editor),inspect:(open=true)=>{controller.fill();if(open){controller.bridge.openParameters?.();document.getElementById("split-module-inspector").scrollIntoView({block:"nearest"});}},selectionChanged:()=>controller.bridge.changed(),change:()=>controller.changed(),planned:()=>controller.updateStatus(),afterRender:()=>controller.modulePanel?.(),report:controller.bridge.report});this.controller=controller;
    document.getElementById("split-module-add").onclick=controller.wrap(()=>this.add(document.getElementById("split-module-type").value,Number(document.getElementById("split-module-position").value)));
    document.getElementById("split-flow-reset").onclick=controller.wrap(()=>{if(controller.cfg){delete controller.cfg.workflow;delete controller.cfg.workflow_version;this.sync();controller.record.accepted=null;controller.fill();controller.changed();}});
    document.getElementById("split-flow-clear").onclick=()=>{if(controller.cfg){controller.cfg.workflow=[];controller.cfg.workflow_version=3;controller.record.selectedModule=null;controller.changed();this.render();}};
  }
  sync(){
    const c=this.controller.cfg;if(!c||!this.controller.context.modules)return;
    const legacy=c.workflow_version!==3&&(Array.isArray(c.workflow)||c.roi||c.points?.length||(c.start||0)>0);
    if(c.workflow==null||(!c.workflow.length&&![2,3].includes(c.workflow_version))){
      c.workflow=["calibration","alignment","statistics","correlation"].map(a=>moduleNode(this.specs[a],this.controller.context.source));Object.assign(c.workflow[0].parameters,c.photometry);delete c.workflow[0].parameters.normalization;if(c.geometry)c.workflow[1].parameters=copy(c.geometry);c.workflow[2].parameters={normalization:c.photometry.normalization};
      for(const k of["window","stride","min_mean","min_samples","lag_min","lag_max","lag_step","lag_list","roi_estimator"])c.workflow[3].parameters[k]=copy(c[k]);Object.assign(c.workflow[3].parameters,{analysis:this.controller.record.analysis||"map",map_lag:this.controller.record.mapLag||0});
    }
    if(c.workflow_version!==3){
      if(legacy){
        const notes=this.controller.record.migrationNotes??=[];
        const add=note=>{if(!notes.includes(note))notes.push(note);};
        if(c.workflow.some(n=>n.enabled&&n.algorithm==="statistics"))add("旧草稿已迁移：统计改为本节点直接上游的完整序列，包含时间与空间编辑；请核对范围后重新运行。历史结果保持原定义。");
        if(c.roi&&c.workflow.some(n=>n.algorithm==="correlation"&&n.parameters.analysis==="curve"&&n.parameters.roi_estimator==="intensity_sum"))add("旧区域总强度已迁移为 ROI 求和模块：默认使用逐样本两路共同有效像素，区别于旧关联窗口固定掩码；请核对无效像素规则。");
        if(c.points?.length>1){this.controller.record.legacyPoints=copy(c.points);add("旧草稿包含多个选点，原坐标已保留。请添加多个 ROI 流程分别计算，或重新指定本次区域；不会将多个点合并为单一 ROI。");}
      }
      if(!c.workflow.some(n=>n.algorithm==="time_crop")){const time=moduleNode(this.specs.time_crop,this.controller.context.source);Object.assign(time.parameters,{first:(c.start||0)+1,last:c.end||null});c.workflow.unshift(time);}
      const index=c.workflow.findIndex(n=>n.enabled&&n.algorithm==="correlation"),corr=c.workflow[index];
      if(corr){corr.parameters.center=c.center==null?null:c.center-(c.start||0)+1;corr.execution={backend:c.backend||"cpu",memory_mb:c.memory_mb||256,precision:c.output_dtype||"float64"};
        if(c.roi){const roi=moduleNode(this.specs.spatial_roi,this.controller.context.source);Object.assign(roi.parameters,c.roi,{mode:corr.parameters.analysis==="curve"&&corr.parameters.roi_estimator==="intensity_sum"?"sum":"pixels"});c.workflow.splice(index,0,roi);}
        else if(corr.parameters.analysis==="curve"&&c.points?.length===1){const roi=moduleNode(this.specs.spatial_roi,this.controller.context.source);Object.assign(roi.parameters,{x:c.points[0][0],y:c.points[0][1],width:1,height:1,mode:"mean"});c.workflow.splice(index,0,roi);}
        if(corr.parameters.analysis==="curve"&&corr.parameters.roi_estimator!=="intensity_sum"&&c.roi){const reduce=moduleNode(this.specs.spatial_roi,this.controller.context.source);reduce.parameters.mode="mean";c.workflow.splice(c.workflow.indexOf(corr)+1,0,reduce);}
        corr.parameters.roi_estimator="pixel_mean";
      }
      if(c.points?.length>1)this.controller.bridge.report("旧草稿有多个选点，请在空间 ROI 模块中重新指定本次区域；旧结果保留完整记录。",true);
      c.roi=null;c.points=[];c.workflow_version=3;
    }
    this.normalize();for(const n of c.workflow){const s=this.specs[n.algorithm];if(s)for(const p of s.parameters)if(get(n.parameters,p.key)===undefined)set(n.parameters,p.key,copy(p.default));}
    const enabled=c.workflow.filter(n=>n.enabled),g=enabled.find(n=>n.algorithm==="alignment"),photo=enabled.find(n=>n.algorithm==="calibration"),stats=enabled.find(n=>n.algorithm==="statistics"),corr=enabled.find(n=>n.algorithm==="correlation");if(g)c.geometry=g.parameters;
    c.photometry=photo?copy(photo.parameters):{curve:"identity",lut_path:null,black_left:0,black_right:0,saturation_code:null,input_max:1023,gamma:1,input_semantics:"sensor_codes"};c.photometry.normalization=stats?.parameters.normalization||"none";
    if(corr){for(const k of["window","stride","min_mean","min_samples","lag_min","lag_max","lag_step","lag_list","roi_estimator"])c[k]=copy(corr.parameters[k]);this.controller.record.analysis=corr.parameters.analysis;this.controller.record.mapLag=corr.parameters.map_lag;}
    if(!c.workflow.some(n=>n.id===this.controller.record.selectedModule))this.controller.record.selectedModule=g?.id||c.workflow[0]?.id||null;
  }
  render(){this.sync();if(!this.controller.cfg)return;super.render();const status=this.options.status();status.parentElement.querySelector(".module-migration")?.remove();if(this.controller.record.migrationNotes?.length){const info=document.createElement("details");info.className="module-migration hint small-copy";info.open=true;info.innerHTML=`<summary>旧配置迁移说明</summary>${this.controller.record.migrationNotes.map(n=>`<p>${esc(n)}</p>`).join("")}${this.controller.record.legacyPoints?.length?`<p>旧点坐标：${esc(this.controller.record.legacyPoints.map(p=>`(${p.join(", ")})`).join("；"))}</p>`:""}`;status.after(info);}const select=document.getElementById("split-module-type"),previous=select.value;select.innerHTML=Object.values(this.specs).map(s=>`<option value="${esc(s.id)}">${esc(s.name)}</option>`).join("");if(this.specs[previous])select.value=previous;const position=document.getElementById("split-module-position"),old=position.value;position.innerHTML=this.nodes.map((n,i)=>`<option value="${i}">${i+1}. ${esc(this.specs[n.algorithm]?.name||n.algorithm)} 之前</option>`).join("")+`<option value="${this.nodes.length}">流程末尾</option>`;position.value=old!==""&&Number(old)<=this.nodes.length?old:this.nodes.length;}
}
