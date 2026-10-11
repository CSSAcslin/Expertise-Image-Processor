// Layout is a view preference only. Resizing never edits algorithm or ROI data.
const clamp=(n,min,max)=>Math.min(Math.max(n,min),Math.max(min,max));
const divider=6,rail=34,columnKeys=["flow","parameters"];
const columnDefault={flow:240,parameters:310},columnMin={flow:190,parameters:240},columnMax={flow:460,parameters:580};
const dockDefaults=()=>({flowWidth:null,parametersWidth:null,flow:false,parameters:false});
const defaults=()=>({schema:3,library:null,tasks:null,compare:.5,general:dockDefaults(),split:dockDefaults()});

// Fit visible columns without replacing their saved, expanded width preferences.
function fitColumns(keys,widths,budget){
  if(!keys.length)return {};
  if(keys.length===1){const key=keys[0];return {[key]:clamp(budget,columnMin[key],columnMax[key])};}
  const flow=clamp(budget*widths.flow/(widths.flow+widths.parameters),Math.max(columnMin.flow,budget-columnMax.parameters),Math.min(columnMax.flow,budget-columnMin.parameters));
  return {flow,parameters:budget-flow};
}

export class WorkbenchLayout {
  constructor(changed){
    this.root=document.querySelector(".workspace");this.changed=changed;this.state=defaults();
    this.addHandle(this.root,"library","vertical","调整数据来源栏宽度");
    this.addHandle(this.root,"dock","vertical","调整右侧处理面板宽度");
    this.addHandle(this.root,"tasks","horizontal","调整任务记录高度");
    this.addHandle(document.querySelector("#viewers"),"compare","vertical","调整原始数据与结果视窗宽度",document.querySelector("#result-pane"));
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
  columns(mode,available=Infinity){
    const s=this.state[mode],open=columnKeys.filter(key=>!s[key]),closed=2-open.length,gap=closed?1:divider;
    const preferred=Object.fromEntries(columnKeys.map(key=>[key,clamp(s[`${key}Width`]??columnDefault[key],columnMin[key],columnMax[key])]));
    const min=open.reduce((n,key)=>n+columnMin[key],closed*rail+gap),max=open.reduce((n,key)=>n+columnMax[key],closed*rail+gap);
    const total=clamp(open.reduce((n,key)=>n+preferred[key],closed*rail+gap),min,Math.min(max,available));
    const widths={flow:rail,parameters:rail,...fitColumns(open,preferred,total-closed*rail-gap)};
    return {open,gap,min,max,total,widths};
  }
  geometry(){
    const width=this.root.clientWidth,previewMin=Math.min(620,Math.max(320,width-900)),shown=!this.root.classList.contains("processing-collapsed");
    const libraryShown=!this.root.classList.contains("library-collapsed"),base=this.columns(this.mode);
    const library=libraryShown?clamp(this.state.library??(width<1200?190:220),160,Math.min(400,width-previewMin-(shown?base.min+divider:0)-divider)):0;
    const available=width-library-previewMin-(libraryShown?divider:0)-divider,columns=this.columns(this.mode,available);
    return {width,previewMin,library,available,columns,dock:shown?columns.total:0};
  }
  metrics(key){
    const g=this.geometry(),height=this.root.clientHeight;
    if(key==="library")return {value:g.library,min:160,max:Math.max(160,Math.min(400,g.width-g.dock-g.previewMin-divider-(g.dock?divider:0))),sign:1};
    if(key==="dock")return {value:g.dock,min:g.columns.min,max:Math.max(g.columns.min,Math.min(g.columns.max,g.available)),sign:-1};
    if(key==="tasks"){
      const max=Math.max(80,Math.min(g.width<=1100?120:360,height-380));
      return {value:clamp(this.state.tasks??(height<754?96:153),80,max),min:80,max,sign:-1};
    }
    if(key.endsWith("-panels")){
      const mode=key.startsWith("split")?"split":"general",c=this.columns(mode,g.available),total=c.widths.flow+c.widths.parameters;
      if(c.open.length<2)return {value:c.widths.flow,min:c.widths.flow,max:c.widths.flow,sign:1};
      return {value:c.widths.flow,min:Math.max(columnMin.flow,total-columnMax.parameters),max:Math.min(columnMax.flow,total-columnMin.parameters),sign:1};
    }
    const host=document.querySelector("#viewers");
    return {value:this.state[key],min:.2,max:.8,sign:1,total:Math.max(1,host.clientWidth-divider)};
  }
  setValue(key,value){
    if(key==="dock"){
      const c=this.geometry().columns,widths=fitColumns(c.open,c.widths,value-(2-c.open.length)*rail-c.gap);
      for(const name of c.open)this.state[this.mode][`${name}Width`]=widths[name];
    }
    else if(key.endsWith("-panels")){
      const mode=key.startsWith("split")?"split":"general",c=this.columns(mode,this.geometry().available);
      this.state[mode].flowWidth=value;this.state[mode].parametersWidth=c.widths.flow+c.widths.parameters-value;
    }
    else this.state[key]=value;
    this.apply();
  }
  resetValue(key){
    if(key==="dock"){this.state[this.mode].flowWidth=null;this.state[this.mode].parametersWidth=null;}
    else if(key.endsWith("-panels")){
      const mode=key.startsWith("split")?"split":"general",c=this.columns(mode,this.geometry().available),widths=fitColumns(columnKeys,columnDefault,c.total-c.gap);
      for(const name of columnKeys)this.state[mode][`${name}Width`]=widths[name];
    }
    else this.state[key]=key==="compare" ? .5 : null;
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
      if(e.key==="Home"){e.preventDefault();e.stopPropagation();this.resetValue(key);return;}
      const keys=vertical?["ArrowLeft","ArrowRight"]:["ArrowUp","ArrowDown"],index=keys.indexOf(e.key);if(index<0)return;
      e.preventDefault();e.stopPropagation();const m=this.metrics(key),step=m.total ? .02 : (e.shiftKey?30:10);
      this.setValue(key,clamp(m.value+(index===0?-1:1)*step*m.sign,m.min,m.max));this.changed();
    };
    el.ondblclick=()=>this.resetValue(key);
  }
  openParameters(mode=this.mode){
    this.root.classList.remove("processing-collapsed");this.state[mode].parameters=false;
    this.apply();this.changed();document.dispatchEvent(new Event("workbench-layout"));
  }
  apply(){
    const geometry=this.geometry(),lib=geometry.library,dock=geometry.dock,focused=this.root.classList.contains("library-collapsed")&&this.root.classList.contains("processing-collapsed"),tasks=focused?0:this.root.classList.contains("task-collapse")?43:this.metrics("tasks").value;
    for(const [key,value]of Object.entries({"library-track":lib,"library-divider":lib?divider:0,"dock-track":dock,"dock-divider":dock?divider:0,"task-track":tasks,"task-divider":focused||this.root.classList.contains("task-collapse")?0:divider}))this.root.style.setProperty(`--${key}`,`${value}px`);
    this.root.style.setProperty("--compare-left",this.state.compare);this.root.style.setProperty("--compare-right",1-this.state.compare);
    for(const mode of["general","split"]){
      const host=document.querySelector(`[data-dock-stack="${mode}"]`),s=this.state[mode],c=this.columns(mode,geometry.available);
      host.style.setProperty("--flow-track",`${c.widths.flow}px`);host.style.setProperty("--parameter-track",`${c.widths.parameters}px`);host.style.setProperty("--panel-divider",`${c.gap}px`);
      host.classList.toggle("both-collapsed",s.flow&&s.parameters);
      for(const key of columnKeys){const panel=host.querySelector(`[data-dock-panel="${key}"]`),button=panel.querySelector("[data-dock-toggle]");panel.classList.toggle("is-collapsed",s[key]);button.setAttribute("aria-expanded",String(!s[key]));button.title=`${s[key]?"展开":"收起"}${key==="flow"?"模块流程":"参数设置"}`;button.setAttribute("aria-label",button.title);button.querySelector(".dock-chevron").textContent=s[key]?"‹":"›";}
      host.querySelector(":scope > .panel-resizer").hidden=s.flow||s.parameters;
    }
    this.root.querySelector('[data-resize="dock"]').hidden=this.state[this.mode].flow&&this.state[this.mode].parameters;
    const taskButton=document.getElementById("toggle-tasks"),collapsed=this.root.classList.contains("task-collapse");taskButton.setAttribute("aria-expanded",String(!collapsed));taskButton.setAttribute("aria-label",`${collapsed?"展开":"折叠"}任务记录`);
    document.querySelectorAll("[data-resize]").forEach(el=>{const m=this.metrics(el.dataset.resize),factor=m.total?100:1;for(const [a,v]of[["now",m.value],["min",m.min],["max",m.max]])el.setAttribute(`aria-value${a}`,String(Math.round(v*factor)));});
  }
  serialize(){return {...structuredClone(this.state),tasksCollapsed:this.root.classList.contains("task-collapse")};}
  restore(value){
    const s=defaults();if(!value||typeof value!=="object"){this.state=s;this.root.classList.remove("task-collapse");this.apply();return;}
    for(const key of["library","tasks","compare"])if(Number.isFinite(value[key])){const ratio=key==="compare";s[key]=clamp(value[key],ratio ? .2 : 80,ratio ? .8 : 520);}
    // Old stacked panels saved one width and a height share; neither is a column width.
    for(const mode of["general","split"]){const v=value[mode];if(!v||typeof v!=="object")continue;for(const key of columnKeys){if([2,3].includes(value.schema)&&Number.isFinite(v[`${key}Width`]))s[mode][`${key}Width`]=clamp(v[`${key}Width`],columnMin[key],columnMax[key]);s[mode][key]=v[key]===true;}}
    this.state=s;this.root.classList.toggle("task-collapse",value.tasksCollapsed===true);this.apply();
  }
  reset(){this.state=defaults();this.root.classList.remove("library-collapsed","processing-collapsed","task-collapse");this.apply();this.changed();document.dispatchEvent(new Event("workbench-layout"));}
}
