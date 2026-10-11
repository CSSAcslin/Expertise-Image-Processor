// Layout is a view preference only. Resizing never edits algorithm or ROI data.
const clamp=(n,min,max)=>Math.min(Math.max(n,min),Math.max(min,max));
const divider=6;
const defaults=()=>({library:null,tasks:null,compare:.5,splitCompare:.5,general:{width:null,share:.46,flow:false,parameters:false},split:{width:null,share:.46,flow:false,parameters:false}});

export class WorkbenchLayout {
  constructor(changed){
    this.root=document.querySelector(".workspace");this.changed=changed;this.state=defaults();
    this.addHandle(this.root,"library","vertical","调整数据来源栏宽度");
    this.addHandle(this.root,"dock","vertical","调整右侧处理面板宽度");
    this.addHandle(this.root,"tasks","horizontal","调整任务记录高度");
    this.addHandle(document.querySelector("#viewers"),"compare","vertical","调整原始数据与结果视窗宽度",document.querySelector("#result-pane"));
    this.addHandle(document.querySelector(".split-views"),"splitCompare","vertical","调整数据预览与处理结果视窗宽度",document.querySelector(".split-result-window"));
    document.querySelectorAll("[data-resize]").forEach(el=>this.bindHandle(el));
    document.querySelectorAll("[data-dock-toggle]").forEach(button=>{
      button.onclick=()=>{const mode=button.closest("[data-dock-stack]").dataset.dockStack,key=button.dataset.dockToggle;this.state[mode][key]=!this.state[mode][key];this.apply();this.changed();};
    });
    document.getElementById("reset-layout").onclick=()=>this.reset();
    new ResizeObserver(()=>this.apply()).observe(this.root);
    this.apply();
  }
  get mode(){return this.root.dataset.analysis==="split_correlation"?"split":"general";}
  addHandle(parent,key,orientation,label,before=null){
    const el=document.createElement("div");el.className="panel-resizer";el.dataset.resize=key;el.tabIndex=0;
    el.setAttribute("role","separator");el.setAttribute("aria-orientation",orientation);el.setAttribute("aria-label",label);
    parent.insertBefore(el,before);return el;
  }
  metrics(key){
    const width=this.root.clientWidth,height=this.root.clientHeight,compact=width<1200;
    const previewMin=Math.min(620,Math.max(320,width-520));
    const library=this.root.classList.contains("library-collapsed")?0:clamp(this.state.library??(compact?190:220),160,Math.min(400,width-previewMin-272));
    const dock=this.root.classList.contains("processing-collapsed")?0:clamp(this.state[this.mode].width??(compact?300:340),260,Math.min(520,width-library-previewMin-12));
    const taskHeight=clamp(this.state.tasks??(height<754?96:153),80,Math.min(360,height-380));
    if(key==="library")return {value:library,min:160,max:Math.min(400,width-dock-previewMin-12),sign:1};
    if(key==="dock")return {value:dock,min:260,max:Math.min(520,width-library-previewMin-12),sign:-1};
    if(key==="tasks")return {value:taskHeight,min:80,max:Math.min(360,height-380),sign:-1};
    const panel=key.endsWith("-panels"),mode=key.startsWith("split")?"split":"general";
    const host=panel?document.querySelector(`[data-dock-stack="${mode}"]`):document.querySelector(key==="compare"?"#viewers":".split-views");
    const total=panel?[...host.querySelectorAll(":scope > .dock-panel")].reduce((n,el)=>n+el.getBoundingClientRect().height,0):host.clientWidth-divider;
    const share=panel&&total?host.querySelector('[data-dock-panel="flow"]').getBoundingClientRect().height/total:this.state[mode].share;
    return {value:panel?share:this.state[key],min:panel?Math.min(.45,100/Math.max(1,total)):.2,max:panel?Math.max(.55,1-100/Math.max(1,total)):.8,sign:1,total};
  }
  setValue(key,value){
    if(key==="dock")this.state[this.mode].width=value;
    else if(key.endsWith("-panels"))this.state[key.startsWith("split")?"split":"general"].share=value;
    else this.state[key]=value;
    this.apply();
  }
  resetValue(key){
    if(key==="dock")this.state[this.mode].width=null;
    else if(key.endsWith("-panels"))this.state[key.startsWith("split")?"split":"general"].share=.46;
    else this.state[key]=["compare","splitCompare"].includes(key) ? .5 : null;
    this.apply();this.changed();
  }
  bindHandle(el){
    const key=el.dataset.resize,vertical=el.getAttribute("aria-orientation")==="vertical";
    el.title="拖动调整 · 方向键微调 · 双击恢复默认";
    el.onpointerdown=e=>{
      if(e.button!==0)return;e.preventDefault();el.focus();
      const start=vertical?e.clientX:e.clientY,m=this.metrics(key),pointer=e.pointerId;
      el.setPointerCapture(pointer);document.body.classList.add(vertical?"resizing-columns":"resizing-rows");
      el.onpointermove=event=>{if(event.pointerId!==pointer)return;const delta=((vertical?event.clientX:event.clientY)-start)*m.sign;this.setValue(key,clamp(m.value+delta/(m.total||1),m.min,m.max));};
      const end=()=>{el.onpointermove=null;el.onpointerup=null;el.onpointercancel=null;el.onlostpointercapture=null;document.body.classList.remove("resizing-columns","resizing-rows");if(el.hasPointerCapture(pointer))el.releasePointerCapture(pointer);this.changed();};
      el.onpointerup=end;el.onpointercancel=end;el.onlostpointercapture=end;
    };
    el.onkeydown=e=>{
      if(e.key==="Home"){e.preventDefault();this.resetValue(key);return;}
      const keys=vertical?["ArrowLeft","ArrowRight"]:["ArrowUp","ArrowDown"],index=keys.indexOf(e.key);if(index<0)return;
      e.preventDefault();const m=this.metrics(key),step=m.total ? .02 : (e.shiftKey?30:10);
      this.setValue(key,clamp(m.value+(index===0?-1:1)*step*m.sign,m.min,m.max));this.changed();
    };
    el.ondblclick=()=>this.resetValue(key);
  }
  openParameters(mode=this.mode){
    this.root.classList.remove("processing-collapsed");this.state[mode].parameters=false;
    this.apply();this.changed();document.dispatchEvent(new Event("workbench-layout"));
  }
  apply(){
    const lib=this.metrics("library").value,dock=this.metrics("dock").value,focused=this.root.classList.contains("library-collapsed")&&this.root.classList.contains("processing-collapsed"),tasks=focused?0:this.root.classList.contains("task-collapse")?43:this.metrics("tasks").value;
    for(const [key,value]of Object.entries({"library-track":lib,"library-divider":lib?divider:0,"dock-track":dock,"dock-divider":dock?divider:0,"task-track":tasks,"task-divider":focused||this.root.classList.contains("task-collapse")?0:divider}))this.root.style.setProperty(`--${key}`,`${value}px`);
    this.root.style.setProperty("--compare-left",this.state.compare);this.root.style.setProperty("--compare-right",1-this.state.compare);
    this.root.style.setProperty("--split-left",`${this.state.splitCompare}fr`);this.root.style.setProperty("--split-right",`${1-this.state.splitCompare}fr`);
    for(const mode of["general","split"]){
      const host=document.querySelector(`[data-dock-stack="${mode}"]`),s=this.state[mode];host.style.setProperty("--flow-share",s.share);host.style.setProperty("--parameter-share",1-s.share);
      host.classList.toggle("has-collapsed-panel",s.flow||s.parameters);
      for(const key of["flow","parameters"]){const panel=host.querySelector(`[data-dock-panel="${key}"]`),button=panel.querySelector("[data-dock-toggle]");panel.classList.toggle("is-collapsed",s[key]);button.setAttribute("aria-expanded",String(!s[key]));button.title=`${s[key]?"展开":"收起"}${key==="flow"?"模块流程":"参数设置"}`;}
      host.querySelector(":scope > .panel-resizer").hidden=s.flow||s.parameters;
    }
    const taskButton=document.getElementById("toggle-tasks"),collapsed=this.root.classList.contains("task-collapse");taskButton.setAttribute("aria-expanded",String(!collapsed));taskButton.setAttribute("aria-label",`${collapsed?"展开":"折叠"}任务记录`);
    document.querySelectorAll("[data-resize]").forEach(el=>{const m=this.metrics(el.dataset.resize),factor=m.total?100:1;for(const [a,v]of[["now",m.value],["min",m.min],["max",m.max]])el.setAttribute(`aria-value${a}`,String(Math.round(v*factor)));});
  }
  serialize(){return {...structuredClone(this.state),tasksCollapsed:this.root.classList.contains("task-collapse")};}
  restore(value){
    const s=defaults();if(!value||typeof value!=="object"){this.state=s;this.root.classList.remove("task-collapse");this.apply();return;}
    for(const key of["library","tasks","compare","splitCompare"])if(Number.isFinite(value[key])){const ratio=["compare","splitCompare"].includes(key);s[key]=clamp(value[key],ratio ? .2 : 80,ratio ? .8 : 520);}
    for(const mode of["general","split"]){const v=value[mode];if(!v||typeof v!=="object")continue;if(Number.isFinite(v.width))s[mode].width=clamp(v.width,260,520);if(Number.isFinite(v.share))s[mode].share=clamp(v.share,.1,.9);for(const key of["flow","parameters"])s[mode][key]=v[key]===true;}
    this.state=s;this.root.classList.toggle("task-collapse",value.tasksCollapsed===true);this.apply();
  }
  reset(){this.state=defaults();this.root.classList.remove("library-collapsed","processing-collapsed","task-collapse");this.apply();this.changed();document.dispatchEvent(new Event("workbench-layout"));}
}
