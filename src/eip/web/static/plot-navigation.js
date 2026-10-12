// Display-only horizontal navigation, shared by all one-dimensional result plots.
const instances=new WeakMap(),clamp=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
// Keep adjacent tick labels distinct when zooming into a narrow interval.
export const plotTick=(value,span)=>Number(value).toPrecision(clamp(Math.ceil(Math.log10(Math.max(Math.abs(value),1)/span))+2,4,15));
export const plotNavigation=host=>{
 if(!instances.has(host))instances.set(host,new PlotNavigation(host));
 return instances.get(host);
};
export const fitPlot=host=>{const nav=instances.get(host);if(!nav?.visible)return false;nav.fit();return true;};
export const plotZoom=host=>{const nav=instances.get(host);return nav?.visible?(nav.full[1]-nav.full[0])/(nav.range[1]-nav.range[0]):null;};

class PlotNavigation{
 constructor(host){
  this.host=host;
  host.classList.add("interactive-plot");
  host.title="滚轮缩放横轴 · 左键拖动左右浏览 · 双击恢复全范围";
  host.addEventListener("wheel",e=>this.wheel(e),{passive:false});
  host.addEventListener("pointerdown",e=>{
   if(e.button!==0||!this.visible)return;
   const p=this.point(e);if(!p||p.x<this.box.left||p.x>this.box.right||p.y<this.box.top||p.y>this.box.bottom)return;
   this.suppressClick=false;this.pointer=null;
   this.drag={id:e.pointerId,x:p.x,clientX:e.clientX,min:this.range[0],span:this.range[1]-this.range[0],moved:false};
   host.setPointerCapture(e.pointerId);host.classList.add("plot-dragging");e.preventDefault();
  });
  host.addEventListener("pointermove",e=>{
   if(!this.visible)return;
   this.pointer={clientX:e.clientX,clientY:e.clientY};
   if(this.drag?.id===e.pointerId){
    const p=this.point(e);if(!p)return;
    this.drag.moved ||= Math.abs(e.clientX-this.drag.clientX)>3;
    if(this.drag.moved)this.move(this.drag.min-(p.x-this.drag.x)/(this.box.right-this.box.left)*this.drag.span,this.drag.span);
   }else this.hover?.(this.pointer);
  });
  const end=e=>{if(this.drag?.id!==e.pointerId)return;this.suppressClick=this.drag.moved;this.drag=null;host.classList.remove("plot-dragging");if(host.hasPointerCapture(e.pointerId))host.releasePointerCapture(e.pointerId);};
  for(const type of["pointerup","pointercancel","lostpointercapture"])host.addEventListener(type,end);
  host.addEventListener("click",e=>{if(this.suppressClick){this.suppressClick=false;e.preventDefault();e.stopImmediatePropagation();}},true);
  host.addEventListener("click",e=>{if(this.visible)this.select?.(e);});
  host.addEventListener("dblclick",e=>{if(this.visible){e.preventDefault();e.stopPropagation();this.fit();}});
  host.addEventListener("pointerleave",()=>{if(!this.drag)this.pointer=null;});
 }
 get visible(){return !this.host.hidden&&!!this.host.querySelector("svg")&&!!this.box;}
 configure(key,full,redraw){
  if(this.key!==key||this.full?.some((v,i)=>v!==full[i])){
   this.range=[...full];this.pointer=null;this.drag=null;this.host.classList.remove("plot-dragging");
  }
  this.key=key;this.full=full;this.redraw=redraw;
  return this.range;
 }
 painted(box,hover,select){
  this.box=box;this.hover=hover;this.select=select;
  const svg=this.host.querySelector("svg");
  svg.dataset.xMin=this.range[0];svg.dataset.xMax=this.range[1];
  if(this.pointer)this.hover?.(this.pointer);
  this.host.dispatchEvent(new Event("plotviewchange"));
 }
 point(e){const matrix=this.host.querySelector("svg")?.getScreenCTM();return matrix?new DOMPoint(e.clientX,e.clientY).matrixTransform(matrix.inverse()):null;}
 move(min,span){
  min=clamp(min,this.full[0],this.full[1]-span);this.range=[min,min+span];
  if(!this.frame)this.frame=requestAnimationFrame(()=>{this.frame=null;if(this.visible)this.redraw();});
 }
 wheel(e){
  if(!this.visible)return;
  const p=this.point(e);if(!p)return;
  e.preventDefault();if(this.drag)return;
  const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?this.host.clientHeight:1);
  const fullSpan=this.full[1]-this.full[0],oldSpan=this.range[1]-this.range[0];
  const span=clamp(oldSpan*Math.exp(clamp(delta,-1000,1000)*.0015),fullSpan/1000,fullSpan);
  const anchor=clamp((p.x-this.box.left)/(this.box.right-this.box.left),0,1);
  this.pointer={clientX:e.clientX,clientY:e.clientY};
  this.move(this.range[0]+anchor*(oldSpan-span),span);
 }
 fit(){this.pointer=null;this.move(this.full[0],this.full[1]-this.full[0]);}
}
