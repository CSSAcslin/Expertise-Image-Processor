const copy=v=>structuredClone(v);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const types={sensor:"单幅图像",pair:"对齐双路",moments:"双路与全源统计",map:"系数图"};
const labels={both:"双路 / 当前图",left:"左路",right:"右路",identity:"原值",linear:"线性归一",sqrt:"平方根",square:"平方",gamma:"幂函数",log:"对数",none:"保留强度尺度",fixed_rms:"全源固定 RMS",map:"固定 τ 系数图",curve:"τ 扫描曲线",nearest:"最近邻",bilinear:"双线性",sensor_codes:"保存码值",linear_intensity:"已确认线性强度",pixel_mean:"逐像素 g² 空间平均",intensity_sum:"区域总强度 g²"};
const get=(o,p)=>p.split(".").reduce((v,k)=>v?.[k],o);
function set(o,p,v){const keys=p.split(".");for(const k of keys.slice(0,-1))o=o[k]??={};o[keys.at(-1)]=v;}

export function moduleNode(s,source){
  const p=copy(s.defaults);
  if(s.id==="alignment"&&source){const[h,w]=source.meta.shape,m=Math.max(1,Math.floor(w/2));Object.assign(p,{left:{x:0,y:0,width:m,height:h},right:{x:m,y:0,width:w-m,height:h},midline:[m,m]});}
  if(s.id==="correlation"&&source?.meta.frame_count){const n=source.meta.frame_count,q=Math.min(10,Math.max(0,Math.floor((n-3)/4)));let w=Math.min(101,Math.max(1,n-2*q));if(w%2===0)w--;Object.assign(p,{window:w,min_samples:Math.min(2,w),lag_min:-q,lag_max:q});}
  return {id:crypto.randomUUID(),algorithm:s.id,parameters:p,enabled:true,target:"both",version:s.version,manifest:s.manifest,code_hash:s.code_hash};
}

export class ModuleEditor{
  constructor(options){this.options=options;this.drag=null;}
  get nodes(){return this.options.nodes()||[];}
  get specs(){return this.options.catalog()||{};}
  normalize(){this.nodes.forEach(n=>{n.id??=crypto.randomUUID();n.enabled??=true;n.parameters??={};n.target??=n.parameters.target||"both";delete n.parameters.target;const s=this.specs[n.algorithm];n.version??=s?.version||"1.0";n.manifest??=null;n.code_hash??=null;});}
  validate(nodes=this.nodes){
    let type="sensor",kind="image",labelled=false;
    for(const[i,n]of nodes.entries()){
      if(!n.enabled)continue;const s=this.specs[n.algorithm];
      if(!s)return `第 ${i+1} 个模块未注册：${n.algorithm}`;
      if(!s.inputs.includes(type))return `第 ${i+1} 个模块「${s.name}」需要 ${s.inputs.map(t=>types[t]).join(" / ")}，当前为${types[type]}。`;
      if(n.target!=="both"&&type!=="pair")return `第 ${i+1} 个模块的左 / 右路范围需要对齐双路输入。`;
      if(["statistics","correlation"].includes(n.algorithm)&&labelled)return "标签不能隐式当作光强计算 g²，请调整阈值模块的位置。";
      if(n.algorithm==="correlation"&&(!Number.isInteger(n.parameters.window)||n.parameters.window<1||n.parameters.window%2!==1||n.parameters.min_samples>n.parameters.window))return "关联窗口须为奇数，且不小于最少有效样本数；原始预览仍可使用。";
      type=s.output==="same"?type:s.output;if(!["alignment","calibration","statistics","correlation","select_channel"].includes(n.algorithm))kind=s.output_kind;labelled||=kind==="label";
    }return null;
  }
  add(algorithm,index=this.nodes.length){if(this.nodes.length>=32)throw Error("处理流程最多 32 个模块");const s=this.specs[algorithm];if(!s)throw Error("请选择已注册模块");const n=moduleNode(s,this.options.source?.());this.nodes.splice(index,0,n);this.options.select?.(n.id);this.changed();if(this.options.special?.(n))this.options.inspect?.();}
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
    host.innerHTML=nodes.length?nodes.map((n,i)=>{const s=this.specs[n.algorithm]||{name:n.algorithm,description:"尚未注册的模块",parameters:[],inputs:[],output:"same"},special=this.options.special?.(n),selected=this.options.selected?.()===n.id;
      return `<article class="flow-node step-card ${n.enabled?"":"disabled"} ${selected?"selected":""}" data-node="${i}" data-algorithm="${esc(n.algorithm)}" data-instance="${esc(n.id)}" draggable="true"><div class="flow-node-title step-header"><span class="flow-grip" title="拖动排序">⠿</span><span class="mono">${i+1}</span><strong>${esc(n.name||s.name)}</strong><button data-up="${i}" class="text-button" aria-label="上移模块" ${i===0?"disabled":""}>↑</button><button data-down="${i}" class="text-button" aria-label="下移模块" ${i===nodes.length-1?"disabled":""}>↓</button></div><p class="flow-type">${s.inputs.map(t=>types[t]).join(" / ")} → ${s.output==="same"?"保持网格":types[s.output]}</p>${special?`<button data-edit="${esc(n.id)}" class="quiet small">${selected?"正在编辑此实例":"编辑此模块参数"}</button>`:`<p class="step-description">${esc(s.description)}</p><div class="flow-fields step-parameters">${s.parameters.map(p=>this.field(p,n,i)).join("")}</div>`}<div class="flow-actions"><label><input type="checkbox" data-enable="${i}" ${n.enabled?"checked":""}/>启用</label>${s.output==="same"?`<label class="flow-target">范围<select data-target="${i}">${["both","left","right"].map(t=>`<option value="${t}" ${t===n.target?"selected":""}>${labels[t]}</option>`).join("")}</select></label>`:""}<button data-duplicate="${i}" class="text-button">复制</button><button data-delete="${i}" class="text-button">删除</button></div></article>`;
    }).join(""):'<p class="empty-steps">流程为空。可添加任意已注册模块；原始数据可照常预览。</p>';
    host.querySelectorAll("[data-node]").forEach(el=>{const i=Number(el.dataset.node),n=nodes[i];
      el.ondragstart=e=>{if(e.target.closest("input,select,button")){e.preventDefault();return;}this.drag=i;e.dataTransfer.effectAllowed="move";e.dataTransfer.setData("text/plain",String(i));};el.ondragover=e=>{e.preventDefault();el.classList.add("drop-target");};el.ondragleave=()=>el.classList.remove("drop-target");el.ondrop=e=>{e.preventDefault();if(this.drag!=null)this.move(this.drag,i);this.drag=null;};
      el.querySelector("[data-up]").onclick=()=>this.move(i,i-1);el.querySelector("[data-down]").onclick=()=>this.move(i,i+1);
      el.querySelector("[data-delete]").onclick=()=>{nodes.splice(i,1);this.changed();this.options.inspect?.(false);};el.querySelector("[data-duplicate]").onclick=()=>{if(nodes.length>=32)return;const d={...copy(n),id:crypto.randomUUID()};nodes.splice(i+1,0,d);this.options.select?.(d.id);this.changed();if(this.options.special?.(d))this.options.inspect?.();};
      el.querySelector("[data-enable]").onchange=e=>{n.enabled=e.target.checked;this.changed();};const target=el.querySelector("[data-target]");if(target)target.onchange=e=>{n.target=e.target.value;this.changed();};
      if(el.querySelector("[data-edit]"))el.querySelector("[data-edit]").onclick=()=>{this.options.select(n.id);this.options.inspect?.();this.render();};
      el.querySelectorAll("[data-param]").forEach(input=>input.onchange=()=>{try{if(!input.reportValidity())return;const kind=input.dataset.kind;let v=kind==="bool"?input.checked:["int","float"].includes(kind)?Number(input.value):kind==="array"?(input.value.trim()?input.value.split(/[,，;\s]+/).map(Number):null):input.value;if(input.dataset.nullable&&input.value==="")v=null;set(n.parameters,input.dataset.key,v);this.options.change?.();this.status();}catch(error){this.options.report?.(error.message,true);}});
    });this.status();this.options.afterRender?.();
  }
  status(){const error=this.validate(),el=this.options.status?.();if(el){el.textContent=error||"按列表顺序执行；每个实例的参数、顺序和版本随结果保存。";el.classList.toggle("error",!!error);}return error;}
}

export class FlowEditor extends ModuleEditor{
  constructor(controller){
    super({nodes:()=>controller.cfg?.workflow,catalog:()=>controller.context.modules,source:()=>controller.context.source,host:()=>document.getElementById("split-flow-list"),status:()=>document.getElementById("split-flow-status"),selected:()=>controller.record?.selectedModule,select:id=>{if(controller.record)controller.record.selectedModule=id;},special:n=>["alignment","calibration","statistics","correlation"].includes(n.algorithm),inspect:(scroll=true)=>{controller.fill();if(scroll)document.getElementById("split-module-inspector").scrollIntoView({block:"nearest"});},change:()=>controller.changed(),afterRender:()=>controller.modulePanel?.(),report:controller.bridge.report});this.controller=controller;
    document.getElementById("split-module-add").onclick=controller.wrap(()=>this.add(document.getElementById("split-module-type").value,Number(document.getElementById("split-module-position").value)));
    document.getElementById("split-flow-reset").onclick=controller.wrap(()=>{if(controller.cfg){delete controller.cfg.workflow;this.sync();controller.record.accepted=null;controller.fill();controller.changed();}});
    document.getElementById("split-flow-clear").onclick=()=>{if(controller.cfg){controller.cfg.workflow=[];controller.cfg.workflow_version=2;controller.record.selectedModule=null;controller.changed();this.render();}};
  }
  sync(){
    const c=this.controller.cfg;if(!c||!this.controller.context.modules)return;
    if(c.workflow==null||(!c.workflow.length&&c.workflow_version!==2)){
      c.workflow=["calibration","alignment","statistics","correlation"].map(a=>moduleNode(this.specs[a],this.controller.context.source));Object.assign(c.workflow[0].parameters,c.photometry);delete c.workflow[0].parameters.normalization;if(c.geometry)c.workflow[1].parameters=copy(c.geometry);c.workflow[2].parameters={normalization:c.photometry.normalization};
      for(const k of["window","stride","min_mean","min_samples","lag_min","lag_max","lag_step","lag_list","roi_estimator"])c.workflow[3].parameters[k]=copy(c[k]);Object.assign(c.workflow[3].parameters,{analysis:this.controller.record.analysis||"map",map_lag:this.controller.record.mapLag||0});
    }
    c.workflow_version=2;this.normalize();for(const n of c.workflow){const s=this.specs[n.algorithm];if(s)for(const p of s.parameters)if(get(n.parameters,p.key)===undefined)set(n.parameters,p.key,copy(p.default));}
    const enabled=c.workflow.filter(n=>n.enabled),g=enabled.find(n=>n.algorithm==="alignment"),photo=enabled.find(n=>n.algorithm==="calibration"),stats=enabled.find(n=>n.algorithm==="statistics"),corr=enabled.find(n=>n.algorithm==="correlation");if(g)c.geometry=g.parameters;
    c.photometry=photo?copy(photo.parameters):{curve:"identity",lut_path:null,black_left:0,black_right:0,saturation_code:null,input_max:1023,gamma:1,input_semantics:"sensor_codes"};c.photometry.normalization=stats?.parameters.normalization||"none";
    if(corr){for(const k of["window","stride","min_mean","min_samples","lag_min","lag_max","lag_step","lag_list","roi_estimator"])c[k]=copy(corr.parameters[k]);this.controller.record.analysis=corr.parameters.analysis;this.controller.record.mapLag=corr.parameters.map_lag;}
    if(!c.workflow.some(n=>n.id===this.controller.record.selectedModule))this.controller.record.selectedModule=g?.id||c.workflow[0]?.id||null;
  }
  render(){this.sync();if(!this.controller.cfg)return;super.render();const select=document.getElementById("split-module-type"),previous=select.value;select.innerHTML=Object.values(this.specs).map(s=>`<option value="${esc(s.id)}">${esc(s.name)}</option>`).join("");if(this.specs[previous])select.value=previous;const position=document.getElementById("split-module-position"),old=position.value;position.innerHTML=this.nodes.map((n,i)=>`<option value="${i}">${i+1}. ${esc(this.specs[n.algorithm]?.name||n.algorithm)} 之前</option>`).join("")+`<option value="${this.nodes.length}">流程末尾</option>`;position.value=old!==""&&Number(old)<=this.nodes.length?old:this.nodes.length;}
}
