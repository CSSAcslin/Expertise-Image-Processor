import { api } from "./api.js";
import { svgIcon } from "./icons.js";
import { plotNavigation, plotTick } from "./plot-navigation.js";
const $=id=>document.getElementById(id),esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const fmt=v=>v==null?"—":Number(v).toPrecision(7),key=e=>JSON.stringify([e.job_id,e.source,e.side_output,e.column||0,e.index||0]);
const colors=["#91d7c5","#e8b87d","#baadf2","#77bde8","#f394ad","#d6d775","#d5a4e8","#84c09a"];
export class CurveComparison{
 constructor(owner){
  this.owner=owner;this.entries=[];this.revision=0;this.serial=0;
  this.bar=document.createElement("div");this.bar.id="curve-toolbar";this.bar.hidden=true;
  this.bar.innerHTML=`<button id="curve-list-toggle" class="quiet small" aria-expanded="false">${svgIcon("layers")}<span id="curve-count">曲线</span></button><button id="curve-add" class="quiet small">${svgIcon("plus")}添加结果</button><select id="curve-axis" aria-label="比较横轴"><option value="samples">样本 / τ</option><option value="time">真实时间 / 时差</option></select><button id="curve-fit" class="icon-button" title="适应曲线" aria-label="适应曲线">${svgIcon("reset")}</button><button id="curve-export" class="icon-button" title="导出完整比较包" aria-label="导出完整比较包">${svgIcon("download")}</button>`;
  const surface=$("result-pane").querySelector(".canvas-surface");surface.before(this.bar);
  this.panel=document.createElement("section");this.panel.id="curve-list";this.panel.hidden=true;this.panel.setAttribute("aria-label","比较曲线清单");surface.before(this.panel);
  const dialog=document.createElement("dialog");dialog.id="curve-add-dialog";dialog.innerHTML=`<div class="dialog-heading"><h2>添加比较结果</h2><button id="curve-add-close" class="icon-button" aria-label="关闭添加曲线">${svgIcon("close")}</button></div><div class="dialog-body"><label>已保存的一维结果<select id="curve-candidate"></select></label><p id="curve-candidate-info" class="module-note"></p><div class="two-fields"><label>分量（从 1 起）<input id="curve-column" type="number" min="1" value="1"/></label><label class="checkbox-line"><input id="curve-all-columns" type="checkbox"/>加入全部分量</label></div><p id="curve-add-error" class="result-inline-warning" role="alert" hidden></p><button id="curve-add-confirm" class="primary">加入比较</button></div>`;document.body.append(dialog);
  const safe=fn=>async()=>{try{await fn();}catch(e){owner.bridge.report(e.message,true);}};
  $("curve-list-toggle").onclick=()=>{this.panel.hidden=!this.panel.hidden;this.expanded=!this.panel.hidden;$("curve-list-toggle").setAttribute("aria-expanded",String(this.expanded));owner.bridge.changed?.();};
  $("curve-add").onclick=safe(()=>this.choose());$("curve-add-close").onclick=()=>dialog.close();
  $("curve-candidate").onchange=()=>this.candidateInfo();$("curve-all-columns").onchange=()=>{$("curve-column").disabled=$("curve-all-columns").checked||this.mode==="component";};
  $("curve-add-confirm").onclick=async()=>{const button=$("curve-add-confirm");button.disabled=true;try{await this.addCandidate();dialog.close();}catch(e){$("curve-add-error").textContent=e.message;$("curve-add-error").hidden=false;}finally{button.disabled=false;}};
  $("curve-axis").onchange=safe(async()=>{const previous=this.x;try{await this.fetch(this.entries,$("curve-axis").value);$("sequence-x").value=this.x;}catch(e){$("curve-axis").value=previous;throw e;}});
  $("curve-fit").onclick=()=>plotNavigation($("split-curve")).fit();$("curve-export").onclick=safe(()=>this.export());
  this.resize=new ResizeObserver(()=>{this.adapt();if(this.data&&!this.bar.hidden&&$("split-curve").dataset.plotOwner==="comparison")this.paint();});this.resize.observe($("split-curve"));
 }
 adapt(){const pane=$("result-pane"),compact=pane.clientHeight<330;if(compact&&!this.compact){this.expanded=false;this.panel.hidden=true;$("curve-list-toggle").setAttribute("aria-expanded","false");}this.compact=compact;pane.classList.toggle("curve-compact",compact);this.panel.style.top=(this.bar.offsetTop+this.bar.offsetHeight)+"px";}
 hide(){this.revision++;this.bar.hidden=this.panel.hidden=true;}
 removeJob(id){const keep=this.entries.map((e,i)=>e.job_id!==id?i:null).filter(i=>i!==null);if(keep.length===this.entries.length)return;this.revision++;this.entries=keep.map(i=>this.entries[i]);if(this.data)this.data.curves=keep.map(i=>this.data.curves[i]);this.ensureFocus();this.renderList();this.paint();}
 snapshot(){return this.data&&!this.bar.hidden?{items:this.entries,focus:this.focus,expanded:this.expanded,x:this.x,mode:this.mode}:null;}
 async restore(saved){if(!saved?.items?.length)return;this.mode=saved.mode||"sequence";this.focus=saved.focus;this.expanded=!!saved.expanded;await this.fetch(saved.items,saved.x||"samples");}
 async open(data){
  const o=this.owner,c=data.contract,mode=data.profile?"component":"sequence",base=JSON.stringify([o.id,o.source,o.sideOutput,mode,mode==="component"?o.index:null]);
  if(!o.id){this.hide();o.plot(data);return;}
  this.mode=mode;this.owner.plotData=null;this.base=base;
  let entries=this.lastBase===base?this.entries:[];
  if(!entries.length){const columns=mode==="component"?[0]:(data.column_indices||[0]).slice(0,32);entries=columns.map(column=>({job_id:o.id,source:o.source,side_output:o.sideOutput,column,index:o.index||0,visible:true}));this.focus=null;this.expanded=entries.length>1;}
  try{await this.fetch(entries,$("sequence-x").value);this.lastBase=base;}catch(e){if(this.lastBase!==base){this.hide();$("split-curve").hidden=true;$("split-result-empty").hidden=false;$("split-result-empty").textContent=e.message;}throw e;}
 }
 async fetch(entries,x=this.x||"samples"){
  const revision=++this.revision,mode=this.mode;
  const data=await api("curves/compare",{items:entries,mode,x});if(revision!==this.revision)return;
  this.entries=entries.map((e,i)=>({...e,name:data.curves[i].name,visible:e.visible!==false,color:/^#[0-9a-f]{6}$/i.test(e.color)?e.color:colors[this.serial++%colors.length]}));
  this.x=x;this.data=data;this.bar.hidden=false;this.panel.hidden=!this.expanded;$("curve-list-toggle").setAttribute("aria-expanded",String(!!this.expanded));$("curve-axis").value=x;$("curve-axis").hidden=mode==="component";
  this.ensureFocus();this.renderList();this.paint();this.owner.bridge.changed?.();
 }
 ensureFocus(){if(!this.entries.some(e=>e.visible&&key(e)===this.focus))this.focus=this.entries.find(e=>e.visible)?key(this.entries.find(e=>e.visible)):null;}
 renderList(){
  $("curve-count").textContent=`曲线 ${this.entries.filter(e=>e.visible).length}/${this.entries.length}`;
  this.panel.innerHTML=`<div class="curve-list-heading"><span>显示 / 名称</span><span>焦点</span><span>光标值</span><span></span></div>`+this.entries.map((e,i)=>`<div class="curve-row ${key(e)===this.focus?"is-focus":""}" data-curve-row="${i}"><label class="curve-name"><input data-curve-visible="${i}" type="checkbox" ${e.visible?"checked":""} aria-label="显示 ${esc(e.name)}"/><input data-curve-color="${i}" type="color" value="${e.color}" aria-label="曲线颜色"/><input data-curve-name="${i}" value="${esc(e.name)}" maxlength="240" aria-label="曲线名称" title="${esc(e.name)}${this.data.curves[i].temporary?" · 临时试算":""}"/></label><input data-curve-focus="${i}" type="radio" name="curve-focus" ${key(e)===this.focus?"checked":""} ${e.visible?"":"disabled"} aria-label="取值焦点 ${esc(e.name)}"/><output data-curve-value="${i}">—</output><button data-curve-remove="${i}" class="icon-button" title="从比较中移除；保留结果文件" aria-label="移除比较曲线 ${esc(e.name)}">${svgIcon("close")}</button></div>`).join("");
  this.panel.querySelectorAll("[data-curve-visible]").forEach(el=>el.onchange=()=>{this.entries[+el.dataset.curveVisible].visible=el.checked;this.ensureFocus();this.renderList();this.paint();this.owner.bridge.changed?.();});
  this.panel.querySelectorAll("[data-curve-focus]").forEach(el=>el.onchange=()=>{this.focus=key(this.entries[+el.dataset.curveFocus]);this.renderList();this.paint();this.owner.bridge.changed?.();});
  this.panel.querySelectorAll("[data-curve-color]").forEach(el=>el.oninput=()=>{this.entries[+el.dataset.curveColor].color=el.value;this.paint();this.owner.bridge.changed?.();});
  this.panel.querySelectorAll("[data-curve-name]").forEach(el=>el.onchange=()=>{const i=+el.dataset.curveName;this.entries[i].name=el.value.trim()||this.data.curves[i].name;this.renderList();this.paint();this.owner.bridge.changed?.();});
  this.panel.querySelectorAll("[data-curve-remove]").forEach(el=>el.onclick=()=>{const i=+el.dataset.curveRemove;this.entries.splice(i,1);this.data.curves.splice(i,1);this.ensureFocus();this.renderList();this.paint();this.owner.bridge.changed?.();});
 }
 async choose(){this.candidates=await api("curves/candidates");$("curve-candidate").innerHTML=this.candidates.map((c,i)=>`<option value="${i}">${esc(c.name)} · ${esc(c.contract.label)}${c.temporary?" · 临时":""}</option>`).join("");$("curve-add-error").hidden=true;$("curve-add-confirm").disabled=!this.candidates.length;this.candidateInfo();$("curve-add-dialog").showModal();}
 candidateInfo(){const c=this.candidates?.[Number($("curve-candidate").value)],n=c?Math.max(1,c.contract.shape.reduce((a,b)=>a*b,1)):1;$("curve-candidate-info").textContent=c?`${c.contract.label} · ${c.contract.meaning} · ${this.mode==="component"?"比较当前样本的截面 / 分量":"比较沿样本轴的时序"}`:"暂无一维结果；请先提取点或区域时序";$("curve-column").max=n;$("curve-column").value=1;$("curve-all-columns").checked=this.mode!=="component"&&n<=4;$("curve-all-columns").disabled=this.mode==="component"||n>32;$("curve-column").disabled=this.mode==="component"||$("curve-all-columns").checked;}
 async addCandidate(){
  const candidate=this.candidates[Number($("curve-candidate").value)];if(!candidate)throw Error("请选择结果");if(!$("curve-column").reportValidity())throw Error("请选择有效分量");
  const n=Math.max(1,candidate.contract.shape.reduce((a,b)=>a*b,1)),columns=this.mode==="component"?[0]:$("curve-all-columns").checked?Array.from({length:n},(_,i)=>i):[Number($("curve-column").value)-1];
  const additions=columns.map(column=>({job_id:candidate.job_id,source:candidate.source,side_output:candidate.side_output,column,index:this.owner.index||0,visible:true})).filter(e=>!this.entries.some(old=>key(old)===key(e)));
  if(!additions.length)throw Error("这些曲线已在比较中");if(this.entries.length+additions.length>32)throw Error("最多同时比较 32 条曲线，请选择部分分量");
  await this.fetch([...this.entries,...additions]);this.expanded=true;this.panel.hidden=false;$("curve-list-toggle").setAttribute("aria-expanded","true");
 }
 paint(){
  if(!this.data||this.bar.hidden)return;const host=$("split-curve"),{xs}=this.data,visible=this.entries.map((e,i)=>({...e,i})).filter(e=>e.visible),focus=visible.find(e=>key(e)===this.focus),W=Math.max(200,host.clientWidth-10),H=Math.max(100,host.clientHeight-10),L=W<400?54:67,R=18,T=24,B=40;
  let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(const x of xs){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);}for(const e of visible)for(const v of this.data.curves[e.i].values)if(v!=null){ymin=Math.min(ymin,v);ymax=Math.max(ymax,v);}
  const histogram=this.data.kind==="B";if(histogram&&this.data.edges){xmin=this.data.edges[0];xmax=this.data.edges.at(-1);}
  if(!Number.isFinite(xmin)){xmin=0;xmax=1;}if(xmin===xmax){xmin-=.5;xmax+=.5;}if(!Number.isFinite(ymin)){ymin=0;ymax=1;}if(histogram){ymin=0;ymax=Math.max(1,Math.ceil(ymax*1.08));}else if(ymin===ymax){const pad=Math.max(1e-6,Math.abs(ymin)*.02);ymin-=pad;ymax+=pad;}else{const pad=(ymax-ymin)*.05;ymin-=pad;ymax+=pad;}
  host.dataset.plotOwner="comparison";host.hidden=false;const nav=plotNavigation(host);
  [xmin,xmax]=nav.configure(JSON.stringify(["comparison",this.base,this.x,this.data.coordinate_digest]),[xmin,xmax],()=>this.paint(),[ymin,ymax]);[ymin,ymax]=nav.yRange;
  const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>H-B-(y-ymin)/(ymax-ymin)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="多组一维数据比较"><rect width="${W}" height="${H}" fill="#101923"/><defs><clipPath id="comparison-clip"><rect x="${L}" y="${T}" width="${Math.max(1,W-L-R)}" height="${Math.max(1,H-T-B)}"/></clipPath></defs>`;
  for(let k=0;k<4;k++){const y=ymin+(ymax-ymin)*k/3,x=xmin+(xmax-xmin)*k/3;svg+=`<path d="M${L},${Y(y)}H${W-R}" stroke="#2a394a"/><text class="plot-y-tick" x="${L-7}" y="${Y(y)+4}" text-anchor="end" fill="#a5bad0" font-size="10">${plotTick(y,ymax-ymin)}</text><text class="plot-x-tick" x="${X(x)}" y="${H-B+18}" text-anchor="middle" fill="#a5bad0" font-size="10">${plotTick(x,xmax-xmin)}</text>`;}
  if(histogram)svg+='<text x="'+L+'" y="14" fill="#bdd0e1" font-size="11">频数</text>';
  svg+='<g clip-path="url(#comparison-clip)">';
  for(const e of visible){let d="",open=false;const values=this.data.curves[e.i].values;for(let i=0;i<xs.length;i++){const v=values[i];if(v==null){open=false;continue;}d+=`${open?"L":"M"}${X(xs[i]).toFixed(2)},${Y(v).toFixed(2)} `;open=true;if(xs.length===1)svg+=`<circle cx="${X(xs[i])}" cy="${Y(v)}" r="3" fill="${e.color}"/>`;}
   if(histogram&&visible.length===1&&this.data.edges){for(let i=0;i<xs.length;i++){const bin=this.data.indices[i],v=values[i];if(v==null)continue;const width=X(this.data.edges[bin+1])-X(this.data.edges[bin]);svg+=`<rect class="histogram-bin" data-bin="${bin}" data-frequency="${v}" x="${X(this.data.edges[bin])}" y="${Y(v)}" width="${Math.max(0,width-1)}" height="${Math.max(0,Y(0)-Y(v))}" fill="${e.color}"><title>强度 [${this.data.edges[bin]}, ${this.data.edges[bin+1]}] · 频数 ${v}</title></rect>`;}}
   else svg+=`<path class="comparison-series" data-curve="${e.i}" d="${d}" fill="none" stroke="${e.color}" stroke-width="${e.i===focus?.i?2.5:1.4}" opacity="${e.i===focus?.i?1:.8}"><title>${esc(e.name)}</title></path>`;}
  svg+=`<path id="comparison-cursor" stroke="#b3c6d8" stroke-dasharray="3 3"/><circle id="comparison-focus-point" r="3.5" fill="${focus?.color||"#fff"}" visibility="hidden"/></g><text x="${(L+W-R)/2}" y="${H-4}" text-anchor="middle" fill="#bdd0e1" font-size="11">${esc(this.data.x_label)}</text>${!visible.length?`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#a5bad0" font-size="12">添加结果并勾选需要显示的曲线</text>`:""}</svg>`;
  host.innerHTML=svg;
  const inspect=(i)=>{this.cursor=i;this.panel.querySelectorAll("[data-curve-value]").forEach(el=>el.textContent=fmt(this.data.curves[+el.dataset.curveValue]?.values[i]));const cursor=$("comparison-cursor");cursor.setAttribute("d",`M${X(xs[i])},${T}V${H-B}`);const point=$("comparison-focus-point"),v=focus?this.data.curves[focus.i].values[i]:null;if(v!=null){point.setAttribute("cx",X(xs[i]));point.setAttribute("cy",Y(v));point.setAttribute("visibility","visible");}else point.setAttribute("visibility","hidden");$("sequence-readout").textContent=`${this.data.x_label} ${fmt(xs[i])}${focus?` · 焦点 ${focus.name}：${fmt(v)}`:" · 未选择显示曲线"}${this.data.downsampled?" · 显示抽样，导出保留完整数据":""}${this.data.warnings.length?"\n"+this.data.warnings.join("；"):""}`;};
  const nearest=e=>{const p=nav.point(e);if(!p)return null;const x=xmin+(p.x-L)/(W-L-R)*(xmax-xmin);let i=0;for(let j=1;j<xs.length;j++)if(Math.abs(xs[j]-x)<Math.abs(xs[i]-x))i=j;return {i,p};};
  nav.painted({left:L,right:W-R,top:T,bottom:H-B},e=>{const p=nearest(e);if(p)inspect(p.i);},e=>{const point=nearest(e);if(!point)return;const values=visible.filter(s=>this.data.curves[s.i].values[point.i]!=null).sort((a,b)=>Math.abs(Y(this.data.curves[a.i].values[point.i])-point.p.y)-Math.abs(Y(this.data.curves[b.i].values[point.i])-point.p.y));if(values.length){this.focus=key(values[0]);this.renderList();this.paint();this.owner.bridge.changed?.();}});
  if(xs.length)inspect(Math.max(0,Math.min(xs.length-1,this.cursor||0)));
 }
 async export(){if(!this.entries.length)throw Error("请先添加曲线");const nav=plotNavigation($("split-curve")),response=await api("curves/export",{items:this.entries,mode:this.mode,x:this.x,view:{focus:this.focus,x_range:nav.range,y_range:nav.yRange}}),link=document.createElement("a");link.href=response.url;link.download=response.filename;document.body.append(link);link.click();link.remove();}
}
