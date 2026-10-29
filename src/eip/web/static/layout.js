import { svgIcon } from "./icons.js";
const clamp=(n,a,b)=>Math.min(Math.max(n,a),Math.max(a,b));
const initial=()=>({schema:6,libraryTab:"data",dataSplit:.4,dataSections:{sources:true,results:true},library:null,libraryHidden:false,flowHeight:innerHeight<820?220:245,tasks:128,tasksCollapsed:true,compare:.5,deviceHeight:286,deviceCollapsed:false,windows:Object.fromEntries(["flow","inspector","input","result"].map(k=>[k,{mode:"dock",hidden:false,bounds:null}]))});
export class WorkbenchLayout{
 constructor(changed){this.root=document.querySelector(".workspace");this.changed=changed;this.state=initial();this.z=400;this.root.classList.toggle("library-collapsed",this.state.libraryHidden);this.root.classList.add("task-collapse");document.querySelectorAll("[data-window]").forEach(el=>{const key=el.dataset.window;el.querySelectorAll("[data-window-action]").forEach(b=>b.onclick=()=>this.action(key,b.dataset.windowAction));this.bindDrag(el);new ResizeObserver(()=>{if(this.state.windows[key].mode==="float"){const r=el.getBoundingClientRect();const previous=this.state.windows[key].bounds;if(!previous||Math.abs(previous.width-r.width)>1||Math.abs(previous.height-r.height)>1){this.state.windows[key].bounds={x:r.x,y:r.y,width:r.width,height:r.height};this.changed();}}}).observe(el);});document.querySelectorAll("[data-show-window]").forEach(b=>b.onclick=()=>this.show(b.dataset.showWindow));document.getElementById("reset-layout").onclick=()=>this.reset();this.addFlowResizer();this.addDockResizers();this.bindDevicePanel();this.bindDataPanel();window.addEventListener("resize",()=>this.apply());this.apply();}
 bindDevicePanel(){
  const panel=document.getElementById("device-monitor"),toggle=document.getElementById("toggle-device-monitor"),handle=document.createElement("div");
  handle.className="device-divider";handle.dataset.resize="devices";handle.tabIndex=0;handle.setAttribute("role","separator");handle.setAttribute("aria-orientation","horizontal");handle.setAttribute("aria-label","调整设备状态区域高度");handle.title="拖动或方向键调整设备状态高度；双击恢复";panel.prepend(handle);
  this.deviceMobileOpen=false;
  const launcher=()=>innerWidth<1280||this.root.classList.contains("processing-collapsed");
  const resize=value=>{this.state.deviceHeight=clamp(value,246,Math.max(246,this.root.clientHeight*.6));this.apply();};
  toggle.onclick=()=>{if(launcher())this.deviceMobileOpen=!this.deviceMobileOpen;else this.state.deviceCollapsed=!this.state.deviceCollapsed;this.apply();this.changed();};
  document.getElementById("open-device-monitor-toolbar").onclick=document.getElementById("open-device-monitor").onclick=()=>{if(launcher())this.deviceMobileOpen=true;else this.state.deviceCollapsed=false;this.apply();this.changed();document.querySelector(".window-menu")?.removeAttribute("open");};
  handle.onpointerdown=e=>{
   if(e.button!==0)return;e.preventDefault();handle.setPointerCapture(e.pointerId);
   const start=e.clientY,height=panel.getBoundingClientRect().height;this.root.dataset.resizing="rows";
   const move=v=>resize(height+start-v.clientY);
   const end=v=>{if(v.pointerId!==e.pointerId)return;handle.removeEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])handle.removeEventListener(type,end);delete this.root.dataset.resizing;if(handle.hasPointerCapture(e.pointerId))handle.releasePointerCapture(e.pointerId);this.changed();};
   handle.addEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])handle.addEventListener(type,end);
  };
  handle.onkeydown=e=>{if(!["ArrowUp","ArrowDown","Home"].includes(e.key))return;e.preventDefault();resize(e.key==="Home"?286:panel.getBoundingClientRect().height+(e.key==="ArrowUp"?10:-10));this.changed();};
  handle.ondblclick=()=>{resize(286);this.changed();};
 }
 bindDataPanel(){
  for(const key of ["sources","results"])document.getElementById("toggle-data-"+key).onclick=()=>{this.state.dataSections[key]=!this.state.dataSections[key];this.apply();this.changed();};
  document.getElementById("open-data-management").onclick=()=>this.openLibrary("data");
  const panel=document.getElementById("library-data"),results=panel.querySelector('[data-data-group="results"]'),handle=document.createElement("div");
  handle.className="data-group-divider";handle.dataset.resize="data-groups";handle.tabIndex=0;handle.setAttribute("role","separator");handle.setAttribute("aria-orientation","horizontal");handle.setAttribute("aria-label","调整输入与处理区域高度");handle.title="拖动调整输入 / 处理高度；双击恢复";results.before(handle);
  const resize=ratio=>{this.state.dataSplit=ratio;this.applyDataSections();};
  handle.onpointerdown=e=>{
   if(e.button!==0)return;e.preventDefault();handle.setPointerCapture(e.pointerId);
   const start=e.clientY,height=panel.querySelector('[data-data-group="sources"]').getBoundingClientRect().height,total=panel.clientHeight-handle.offsetHeight;
   this.root.dataset.resizing="rows";
   const move=v=>resize((height+v.clientY-start)/Math.max(1,total));
   const end=v=>{if(v.pointerId!==e.pointerId)return;handle.removeEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])handle.removeEventListener(type,end);delete this.root.dataset.resizing;if(handle.hasPointerCapture(e.pointerId))handle.releasePointerCapture(e.pointerId);this.changed();};
   handle.addEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])handle.addEventListener(type,end);
  };
  handle.onkeydown=e=>{if(!["ArrowUp","ArrowDown","Home","End"].includes(e.key))return;e.preventDefault();resize(e.key==="Home"?0:e.key==="End"?1:this.state.dataSplit+(e.key==="ArrowUp"?-.025:.025));this.changed();};
  handle.ondblclick=()=>{resize(.4);this.changed();};
 }
 applyDataSections(){
  for(const key of ["sources","results"]){
   const expanded=this.state.dataSections[key],button=document.getElementById("toggle-data-"+key);
   document.getElementById("library-"+key).hidden=!expanded;
   button.setAttribute("aria-expanded",String(expanded));
   document.querySelector('[data-data-group="'+key+'"]').classList.toggle("section-collapsed",!expanded);
  }
  for(const key of ["data","modules"]){
   const active=this.state.libraryTab===key,button=document.querySelector('[data-library-tab="'+key+'"]');
   document.getElementById("library-"+key).hidden=!active;
   button.classList.toggle("selected",active);button.setAttribute("aria-selected",String(active));button.tabIndex=active?0:-1;
  }
  const panel=document.getElementById("library-data"),handle=panel.querySelector(".data-group-divider");
  handle.hidden=!this.state.dataSections.sources||!this.state.dataSections.results;
  const total=panel.clientHeight-handle.offsetHeight,minimum=total>0?Math.min(120,total*.35)/total:.15,maximum=total>0?1-Math.min(200,total*.4)/total:.85;
  const ratio=clamp(Number.isFinite(this.state.dataSplit)?this.state.dataSplit:.4,minimum,maximum);this.state.dataSplit=ratio;
  panel.style.setProperty("--data-input-share",ratio);panel.style.setProperty("--data-result-share",1-ratio);
  handle.setAttribute("aria-valuemin",Math.round(minimum*100));handle.setAttribute("aria-valuemax",Math.round(maximum*100));handle.setAttribute("aria-valuenow",Math.round(ratio*100));
 }
 openLibrary(tab="data"){
  this.state.libraryTab=tab==="modules"?"modules":"data";
  this.root.classList.remove("library-collapsed");this.apply();this.changed();
  document.querySelector(".window-menu")?.removeAttribute("open");
 }
 openData(key="sources"){
  if(!["sources","results"].includes(key))key="sources";
  this.state.dataSections[key]=true;this.openLibrary("data");
  document.getElementById("toggle-data-"+key).focus({preventScroll:true});
 }
 get mode(){return "general";}
 action(key,action){const s=this.state.windows[key];if(action==="hide")s.hidden=true;else if(action==="float"){s.mode="float";s.hidden=false;s.bounds??={x:Math.max(15,innerWidth*.22),y:100,width:Math.min(key==="inspector"?350:620,innerWidth-30),height:Math.min(460,innerHeight-180)};}else if(action==="dock")s.mode="dock";else if(action==="maximize"){s.mode=s.mode==="max"?(s.previous||"dock"):(s.previous=s.mode,"max");s.hidden=false;}this.apply();this.changed();}
 show(key){this.state.windows[key].hidden=false;const el=document.querySelector(`[data-window="${key}"]`);el?.classList.remove("responsive-hidden");if(key==="inspector"){this.root.classList.remove("processing-collapsed");document.body.classList.remove("browsing-results");}if(["input","result"].includes(key)){const views=document.getElementById("viewers");if(views.dataset.mode!==(key==="input"?"input":"result"))views.dataset.mode="compare";}this.apply();this.changed();document.querySelector(".window-menu")?.removeAttribute("open");}
 openParameters(){this.show("inspector");}
 bindDrag(el){const head=el.querySelector(".window-heading");head?.addEventListener("pointerdown",e=>{if(e.button!==0||e.target.closest("button,input,select,summary")||this.state.windows[el.dataset.window].mode!=="float")return;const box=el.getBoundingClientRect(),start={x:e.clientX,y:e.clientY},key=el.dataset.window;head.setPointerCapture(e.pointerId);el.style.zIndex=String(++this.z);const move=v=>{const x=clamp(box.x+v.clientX-start.x,0,innerWidth-100),y=clamp(box.y+v.clientY-start.y,61,innerHeight-110);el.style.left=x+"px";el.style.top=y+"px";this.state.windows[key].bounds={x,y,width:box.width,height:box.height};};const end=()=>{head.removeEventListener("pointermove",move);head.removeEventListener("pointerup",end);head.removeEventListener("pointercancel",end);this.changed();};head.addEventListener("pointermove",move);head.addEventListener("pointerup",end);head.addEventListener("pointercancel",end);});el.addEventListener("pointerdown",()=>{if(el.classList.contains("is-floating"))el.style.zIndex=String(++this.z);});}
 addFlowResizer(){const flow=document.querySelector('[data-window="flow"]'),el=document.createElement("div");el.className="panel-resizer";el.dataset.resize="flow-height";el.tabIndex=0;el.setAttribute("role","separator");el.setAttribute("aria-orientation","horizontal");el.setAttribute("aria-label","调整流程窗口高度");flow.append(el);el.onpointerdown=e=>{if(e.button!==0)return;e.preventDefault();el.setPointerCapture(e.pointerId);const y=e.clientY,h=flow.clientHeight;el.onpointermove=v=>{this.state.flowHeight=clamp(h+v.clientY-y,150,this.dockLimits().flow);this.apply();};el.onpointerup=()=>{el.onpointermove=null;this.changed();};};el.onkeydown=e=>{if(["ArrowUp","ArrowDown"].includes(e.key)){e.preventDefault();this.state.flowHeight=clamp(this.state.flowHeight+(e.key==="ArrowUp"?-10:10),150,this.dockLimits().flow);this.apply();this.changed();}};el.ondblclick=()=>{this.state.flowHeight=initial().flowHeight;this.apply();this.changed();};}
 dockLimits(){
  const inspector=this.state.windows.inspector,flow=this.state.windows.flow;
  const monitorDocked=innerWidth>=1280&&!this.root.classList.contains("processing-collapsed")||this.deviceMobileOpen&&innerWidth>=960;
  const inspectorWidth=monitorDocked||innerWidth>=1280&&inspector.mode==="dock"&&!inspector.hidden&&!this.root.classList.contains("processing-collapsed")?(innerWidth<1400?300:330):0;
  const runHeight=innerHeight<820?48:58,previewFloor=innerHeight<820?290:350;
  const taskFloor=this.root.classList.contains("task-collapse")?28:96;
  const flowMax=Math.max(150,this.root.clientHeight-runHeight-taskFloor-previewFloor);
  const flowHeight=flow.mode==="dock"&&!flow.hidden?clamp(this.state.flowHeight,150,flowMax):0;
  return {flow:flowMax,library:Math.max(244,Math.min(420,this.root.clientWidth-inspectorWidth-420)),tasks:Math.max(96,Math.min(160,this.root.clientHeight-flowHeight-runHeight-previewFloor))};
 }
 addDockResizers(){
  for(const[key,selector,orientation,label]of [["library",".library","vertical","调整左侧栏宽度"],["tasks",".tasks","horizontal","调整任务记录高度"]]){
   const panel=document.querySelector(selector),el=document.createElement("div");
   el.className="panel-resizer dock-resizer";el.dataset.resize=key;el.tabIndex=0;el.setAttribute("role","separator");el.setAttribute("aria-orientation",orientation);el.setAttribute("aria-label",label);el.title=label+"；拖动或方向键调整，双击恢复";panel.append(el);
   const resize=value=>{
    if(key==="library")this.state.library=clamp(value,244,this.dockLimits().library);
    else{this.root.classList.toggle("task-collapse",value<70);if(value>=70)this.state.tasks=clamp(value,96,this.dockLimits().tasks);}
    this.apply();
   };
   el.onpointerdown=e=>{
    if(e.button!==0)return;e.preventDefault();el.setPointerCapture(e.pointerId);
    const coordinate=key==="library"?e.clientX:e.clientY,size=key==="library"?panel.getBoundingClientRect().width:panel.getBoundingClientRect().height;
    this.root.dataset.resizing=key==="library"?"columns":"rows";
    const move=v=>resize(size+(key==="library"?v.clientX-coordinate:coordinate-v.clientY));
    const end=v=>{if(v.pointerId!==e.pointerId)return;el.removeEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])el.removeEventListener(type,end);delete this.root.dataset.resizing;if(el.hasPointerCapture(e.pointerId))el.releasePointerCapture(e.pointerId);this.changed();};
    el.addEventListener("pointermove",move);for(const type of["pointerup","pointercancel","lostpointercapture"])el.addEventListener(type,end);
   };
   el.onkeydown=e=>{
    const positive=key==="library"?"ArrowRight":"ArrowUp",negative=key==="library"?"ArrowLeft":"ArrowDown";
    if(![positive,negative].includes(e.key))return;e.preventDefault();
    const current=key==="library"?panel.getBoundingClientRect().width:panel.getBoundingClientRect().height;
    resize(key==="tasks"&&current<=28&&e.key===positive?96:key==="tasks"&&current<=96&&e.key===negative?28:current+(e.key===positive?10:-10));this.changed();
   };
   el.ondblclick=()=>{if(key==="library")this.state.library=null;else{this.state.tasks=128;this.root.classList.add("task-collapse");}this.apply();this.changed();};
  }
 }
 apply(){
  const narrow=innerWidth<1280,libraryHidden=this.root.classList.contains("library-collapsed"),processingHidden=this.root.classList.contains("processing-collapsed"),inspector=this.state.windows.inspector,flow=this.state.windows.flow,flowDocked=flow.mode==="dock"&&!flow.hidden,inspectorDocked=!narrow&&inspector.mode==="dock"&&!inspector.hidden&&!processingHidden;
  const panel=document.getElementById("device-monitor"),launcher=narrow||processingHidden,deviceCollapsed=launcher?!this.deviceMobileOpen:this.state.deviceCollapsed;
  const deviceBottom=launcher&&this.deviceMobileOpen&&innerWidth<960,deviceDocked=!launcher||this.deviceMobileOpen&&!deviceBottom;
  const deviceHeight=deviceCollapsed?32:clamp(this.state.deviceHeight||286,246,Math.max(246,this.root.clientHeight*.6));
  panel.classList.toggle("is-collapsed",deviceCollapsed);panel.classList.toggle("is-launcher",launcher&&!this.deviceMobileOpen);panel.classList.toggle("is-bottom",deviceBottom);
  this.root.classList.toggle("device-bottom-open",deviceBottom);this.root.style.setProperty("--device-track",deviceHeight+"px");this.root.style.setProperty("--device-inspector-space",deviceDocked?deviceHeight+"px":"0px");
  document.getElementById("device-monitor-body").hidden=deviceCollapsed;
  const deviceToggle=document.getElementById("toggle-device-monitor"),deviceLabel=deviceCollapsed?"展开设备状态":"折叠设备状态";deviceToggle.setAttribute("aria-expanded",String(!deviceCollapsed));deviceToggle.setAttribute("aria-label",deviceLabel);deviceToggle.title=deviceLabel;
  const deviceSeparator=panel.querySelector(".device-divider");deviceSeparator.setAttribute("aria-valuemin",246);deviceSeparator.setAttribute("aria-valuemax",Math.round(Math.max(246,this.root.clientHeight*.6)));deviceSeparator.setAttribute("aria-valuenow",Math.round(deviceHeight));
  const limits=this.dockLimits(),libraryWidth=clamp(this.state.library||(narrow?260:280),244,limits.library),taskHeight=this.root.classList.contains("task-collapse")?28:clamp(this.state.tasks,96,limits.tasks);
  const tracks={library:libraryHidden?0:libraryWidth,inspector:inspectorDocked||deviceDocked?(innerWidth<1400?300:330):0,flow:flowDocked&&!deviceBottom?clamp(this.state.flowHeight,150,limits.flow):0,run:innerHeight<820?48:58,task:taskHeight};
  for(const[key,value]of Object.entries(tracks)){this.root.style.setProperty("--"+key+"-track",value+"px");document.body.style.setProperty("--"+key+"-track",value+"px");}
  this.applyDataSections();document.querySelector(".library").style.display=libraryHidden?"none":"flex";
  for(const[key,value,min,max]of[["library",libraryWidth,244,limits.library],["tasks",taskHeight,28,limits.tasks]]){
   const separator=this.root.querySelector(`[data-resize="${key}"]`);separator.setAttribute("aria-valuemin",min);separator.setAttribute("aria-valuemax",max);separator.setAttribute("aria-valuenow",Math.round(value));
  }
 for(const[key,s]of Object.entries(this.state.windows)){const el=document.querySelector(`[data-window="${key}"]`);if(!el)continue;el.classList.toggle("is-hidden",s.hidden||key==="inspector"&&processingHidden);el.classList.toggle("is-floating",s.mode==="float");el.classList.toggle("is-maximized",s.mode==="max");if(key==="inspector"){if(narrow&&!this.wasNarrow)el.classList.add("responsive-hidden");else if(!narrow)el.classList.remove("responsive-hidden");}for(const prop of["left","top","width","height"])el.style.removeProperty(prop);if(s.mode==="float"){const b=s.bounds||{x:100,y:100,width:550,height:400};el.style.left=clamp(b.x,0,innerWidth-100)+"px";el.style.top=clamp(b.y,61,innerHeight-110)+"px";el.style.width=clamp(b.width,300,innerWidth-20)+"px";el.style.height=clamp(b.height,210,innerHeight-130)+"px";}const button=el.querySelector('[data-window-action="float"],[data-window-action="dock"]');if(button){button.dataset.windowAction=s.mode==="dock"?"float":"dock";button.setAttribute("aria-label",s.mode==="dock"?"浮动窗口":"停靠窗口");button.title=s.mode==="dock"?"浮动窗口":"停靠窗口";button.innerHTML=svgIcon(s.mode==="dock"?"float":"dock");}}
 this.wasNarrow=narrow;const toggle=document.getElementById("toggle-tasks");toggle.setAttribute("aria-label",this.root.classList.contains("task-collapse")?"展开任务":"折叠任务");toggle.setAttribute("aria-expanded",String(!this.root.classList.contains("task-collapse")));requestAnimationFrame(()=>document.dispatchEvent(new Event("workbench-windows-resized")));document.dispatchEvent(new Event("device-monitor-layout"));}
 serialize(){return {...structuredClone(this.state),libraryHidden:this.root.classList.contains("library-collapsed"),tasksCollapsed:this.root.classList.contains("task-collapse")};}
 restore(value){
  if([4,5,6].includes(value?.schema)){
   const defaults=initial(),saved=structuredClone(value);
   this.state={...defaults,...saved,schema:6,dataSections:{...defaults.dataSections,...saved.dataSections},windows:{...defaults.windows,...saved.windows}};
   delete this.state.dataHeight;
   if(value.schema!==6){
    // Older drafts describe a different panel hierarchy; adopt the new default geometry.
    this.state.libraryTab="data";this.state.libraryHidden=false;this.state.library=null;
    this.state.flowHeight=defaults.flowHeight;this.state.tasksCollapsed=true;this.state.tasks=defaults.tasks;
   }
   if(!["data","modules"].includes(this.state.libraryTab))this.state.libraryTab="data";
   this.root.classList.toggle("library-collapsed",this.state.libraryHidden);this.root.classList.toggle("task-collapse",this.state.tasksCollapsed);
  }
  this.apply();
 }
 reset(){this.state=initial();this.wasNarrow=false;this.deviceMobileOpen=false;this.root.classList.toggle("library-collapsed",this.state.libraryHidden);this.root.classList.remove("processing-collapsed");this.root.classList.add("task-collapse");this.apply();this.changed();}
}
