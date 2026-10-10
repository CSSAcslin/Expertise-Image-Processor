const copy = v => structuredClone(v);
const escape = v => String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const specs = {
  calibration:{name:"强度校准 / LUT",input:["sensor"],output:"sensor",stage:"preprocess"},
  alignment:{name:"双幅划分与固定配准",input:["sensor"],output:"pair",stage:"geometry"},
  affine:{name:"线性缩放与偏置",input:["sensor","pair","map"],defaults:{gain:1,offset:0,target:"both"}},
  gaussian:{name:"空间高斯平滑",input:["sensor","pair","map"],defaults:{sigma:1,radius:2,target:"both"}},
  curve:{name:"强度曲线",input:["sensor","pair","map"],defaults:{curve:"linear",input_max:1023,gamma:1,target:"both"}},
  statistics:{name:"全源均值与固定尺度",input:["pair"],output:"moments",stage:"preprocess"},
  correlation:{name:"标准二阶互关联 g²",input:["moments"],output:"map",stage:"trial"},
};
const typeName={sensor:"单幅传感器数据",pair:"已对齐双路",moments:"双路强度与全源统计",map:"系数图"};
const titles={gain:"倍率",offset:"偏置",sigma:"σ（像素）",radius:"半径（像素）",input_max:"固定输入尺度",gamma:"指数 γ",target:"作用幅面",curve:"曲线"};
const options={target:{both:"双路 / 当前图",left:"左路",right:"右路"},curve:{identity:"原值",linear:"线性归一",sqrt:"平方根",square:"平方",gamma:"幂函数",log:"对数"}};

export class FlowEditor {
  constructor(controller) {
    this.controller=controller;
    document.getElementById("split-module-add").onclick=()=>this.add();
    document.getElementById("split-flow-reset").onclick=()=>{const c=this.controller.cfg;if(c){delete c.workflow;this.sync();this.render();this.controller.changed();}};
  }
  sync() {
    const c=this.controller.cfg;if(!c?.geometry)return;
    c.workflow??=[
      {id:"calibration",algorithm:"calibration",enabled:true,parameters:{}},
      {id:"alignment",algorithm:"alignment",enabled:true,parameters:{}},
      {id:"statistics",algorithm:"statistics",enabled:true,parameters:{}},
      {id:"correlation",algorithm:"correlation",enabled:true,parameters:{}},
    ];
    for(const n of c.workflow) {
      if(n.algorithm==="calibration")n.parameters=copy(c.photometry);
      if(n.algorithm==="alignment")n.parameters=copy(c.geometry);
      if(n.algorithm==="statistics")n.parameters={normalization:c.photometry.normalization};
      if(n.algorithm==="correlation")n.parameters={window:c.window,min_mean:c.min_mean,min_samples:c.min_samples};
    }
  }
  validate() {
    let type="sensor";
    for(const n of this.controller.cfg?.workflow||[]) {
      if(!n.enabled)continue;
      const s=specs[n.algorithm];
      if(!s||!s.input.includes(type))return `「${s?.name||n.algorithm}」不能接收${typeName[type]}，请调整顺序。`;
      if(n.parameters.target&&n.parameters.target!=="both"&&type!=="pair")return "左 / 右路模块只能放在分幅之后、全源统计之前。";
      type=s.output||type;
    }
    return null;
  }
  add() {
    this.sync();const c=this.controller.cfg;if(!c?.workflow)return;
    if(c.workflow.length>=32){this.controller.bridge.report("处理流程最多 32 个模块",true);return;}
    const algorithm=document.getElementById("split-module-type").value;
    const index=Number(document.getElementById("split-module-position").value);
    c.workflow.splice(index,0,{id:crypto.randomUUID(),algorithm,enabled:true,parameters:copy(specs[algorithm].defaults)});
    this.render();this.controller.changed();
  }
  move(from,to) {
    const nodes=this.controller.cfg.workflow;if(to<0||to>=nodes.length)return;
    nodes.splice(to,0,nodes.splice(from,1)[0]);this.render();this.controller.changed();
  }
  render() {
    this.sync();const c=this.controller.cfg;if(!c?.workflow)return;
    const host=document.getElementById("split-flow-list");
    host.innerHTML=c.workflow.map((n,i)=>{
      const s=specs[n.algorithm],repeat=!!s.defaults;
      const fields=repeat?Object.entries(n.parameters).map(([key,value])=>options[key]?`<label>${titles[key]}<select data-param="${key}">${Object.entries(options[key]).map(([v,t])=>`<option value="${v}" ${v===value?"selected":""}>${t}</option>`).join("")}</select></label>`:`<label>${titles[key]}<input type="number" step="${key==="radius"?1:"any"}" data-param="${key}" value="${escape(value)}"/></label>`).join(""):"";
      const note=n.algorithm==="calibration"?`${c.photometry.lut_path?"自定义 LUT":c.photometry.curve||"identity"} · 左 / 右偏置 ${c.photometry.black_left} / ${c.photometry.black_right}`:n.algorithm==="alignment"?`偏移 (${c.geometry.dx}, ${c.geometry.dy}) · ${c.geometry.interpolation}`:n.algorithm==="statistics"?`全部输入帧 · ${c.photometry.normalization==="none"?"保留当前尺度":"固定 RMS"}`:`W=${c.window} · 最少 ${c.min_samples} 个有效样本`;
      return `<article class="flow-node ${n.enabled?"":"disabled"}" data-node="${i}" draggable="true"><div class="flow-node-title"><span class="flow-grip" title="拖动调整顺序">⠿</span><span class="mono">${i+1}</span><strong>${s.name}</strong><button data-move="-1" class="text-button" aria-label="上移模块" ${i===0?"disabled":""}>↑</button><button data-move="1" class="text-button" aria-label="下移模块" ${i===c.workflow.length-1?"disabled":""}>↓</button></div>${repeat?`<div class="flow-fields">${fields}</div><div class="flow-actions"><label><input type="checkbox" data-enable ${n.enabled?"checked":""}/>启用</label><button data-duplicate class="text-button">复制</button><button data-delete class="text-button">移除</button></div>`:`<p class="hint small-copy">${escape(note)}</p><button data-edit="${s.stage}" class="text-button">设置此模块参数</button>`}</article>`;
    }).join("");
    host.querySelectorAll("[data-node]").forEach(el=>{
      const index=Number(el.dataset.node),node=c.workflow[index];
      el.ondragstart=e=>{if(e.target.closest("input,select")){e.preventDefault();return;}this.drag=index;e.dataTransfer.effectAllowed="move";e.dataTransfer.setData("text/plain",String(index));};
      el.ondragover=e=>{e.preventDefault();el.classList.add("drop-target");};
      el.ondragleave=()=>el.classList.remove("drop-target");
      el.ondrop=e=>{e.preventDefault();if(this.drag!=null)this.move(this.drag,index);this.drag=null;};
      el.querySelectorAll("[data-move]").forEach(b=>b.onclick=()=>this.move(index,index+Number(b.dataset.move)));
      el.querySelectorAll("[data-param]").forEach(input=>input.onchange=()=>{node.parameters[input.dataset.param]=input.tagName==="SELECT"?input.value:Number(input.value);this.controller.changed();this.status();});
      if(el.querySelector("[data-edit]"))el.querySelector("[data-edit]").onclick=()=>this.controller.setStage(el.querySelector("[data-edit]").dataset.edit);
      if(el.querySelector("[data-enable]"))el.querySelector("[data-enable]").onchange=e=>{node.enabled=e.target.checked;this.render();this.controller.changed();};
      if(el.querySelector("[data-delete]"))el.querySelector("[data-delete]").onclick=()=>{c.workflow.splice(index,1);this.render();this.controller.changed();};
      if(el.querySelector("[data-duplicate]"))el.querySelector("[data-duplicate]").onclick=()=>{if(c.workflow.length>=32)return;c.workflow.splice(index+1,0,{...copy(node),id:crypto.randomUUID()});this.render();this.controller.changed();};
    });
    document.getElementById("split-module-position").innerHTML=c.workflow.map((n,i)=>`<option value="${i}" ${n.algorithm==="statistics"?"selected":""}>${i+1}. ${specs[n.algorithm].name} 之前</option>`).join("")+`<option value="${c.workflow.length}">流程末尾</option>`;
    this.status();
  }
  status() {
    const error=this.validate(),el=document.getElementById("split-flow-status");
    el.textContent=error||"按列表顺序执行；参数与顺序写入每次运行记录。";el.classList.toggle("error",!!error);
  }
}
