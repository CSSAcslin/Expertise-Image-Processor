import { CurveComparison } from "./curve-comparison.js";
import { api,download } from "./api.js";
import { plotNavigation, plotTick } from "./plot-navigation.js";
const $=id=>document.getElementById(id),esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]),format=v=>v==null?"未知":Number(v).toPrecision(7),extrema=values=>values.reduce((b,v)=>v==null||!Number.isFinite(v)?b:[Math.min(b[0],v),Math.max(b[1],v)],[Infinity,-Infinity]);
export class SequenceResults{
 constructor(bridge){this.bridge=bridge;this.generation=0;this.sideOutput=null;this.resize=new ResizeObserver(()=>{if(this.plotData&&!$("split-curve").hidden)this.plot(this.plotData);});this.resize.observe($("split-curve"));const wrap=fn=>async()=>{try{await fn();}catch(e){this.stop();bridge.report(e.message,true);}};
  $("sequence-source").onchange=wrap(async()=>{this.stop();this.source=Number($("sequence-source").value);this.sideOutput=this.preferredSideOutput();this.index=0;$("sequence-channel").value="0";this.artifactOptions();await this.show();});$("result-artifact").onchange=wrap(async()=>{this.stop();this.sideOutput=$("result-artifact").value||null;this.index=0;$("sequence-channel").value="0";await this.show();});$("sequence-index").onchange=wrap(async()=>{this.index=Number($("sequence-index").value);await this.show();});for(const id of["sequence-channel","sequence-x"])$(id).onchange=wrap(()=>this.show());$("sequence-play").onclick=wrap(()=>this.play());
  $("sequence-promote").onclick=wrap(async()=>{if(!this.sideOutput)return bridge.promote(this.id,this.source);const opened=await api(`jobs/${this.id}/sequence/promote`,{source:this.source,side_output:this.sideOutput,target:"input"});await bridge.promoted?.(opened,"input");bridge.report("截取数据已加入输入来源");});$("artifact-to-result").onclick=wrap(async()=>{const opened=await api(`jobs/${this.id}/sequence/promote`,{source:this.source,side_output:this.sideOutput,target:"result"});await bridge.promoted?.(opened,"result");bridge.report("已加入处理结果");});$("artifact-export").onclick=wrap(()=>download(this.id,$("artifact-export-format").value,this.source,this.sideOutput));
  const fields=document.createElement("div");fields.className="two-fields";fields.innerHTML='<label>信号视图<select id="sequence-plot-axis"><option value="sequence">沿样本 / 扫描轴</option><option value="component">当前截面 / 分量</option></select></label><label>显示分量<select id="sequence-series"></select></label>';$("sequence-x").parentElement.parentElement.after(fields);$("sequence-plot-axis").onchange=wrap(()=>this.show());$("sequence-series").onchange=()=>{if(this.plotData)this.plot(this.plotData);};this.comparison=new CurveComparison(this);
 }
 stop(){this.playing=false;clearTimeout(this.timer);$("sequence-play").textContent="播放结果";}
 hide(){this.comparison?.hide();this.stop();this.generation++;this.plotRevision=(this.plotRevision||0)+1;this.plotData=null;$("sequence-controls").hidden=true;$("sequence-nav").hidden=true;}
 get item(){const parent=this.meta?.results[this.source];return this.sideOutput?parent?.side_outputs?.find(s=>s.id===this.sideOutput):parent;}
 preferredSideOutput(){const parent=this.meta.results[this.source],last=parent.modules?.nodes?.filter(n=>n.enabled).at(-1);return ["distribution","paired_trace"].includes(last?.algorithm)?(parent.side_outputs||[]).find(s=>s.node_id===last.id&&(s.contract?.semantic==="distribution"||last.algorithm==="paired_trace")&&s.status==="succeeded")?.id||null:null;}
 artifactOptions(){const parent=this.meta.results[this.source];$("result-artifact").innerHTML=`<option value="">流程主输出 · ${esc(parent.contract.full_axes)}</option>`+(parent.side_outputs||[]).map(s=>`<option value="${esc(s.id)}" ${s.status!=="succeeded"?"disabled":""}>${esc(s.name||s.port||s.id)}${s.temporary?" · 临时":""}${s.status!=="succeeded"?" · "+esc(s.status):""}</option>`).join("");$("result-artifact").value=this.sideOutput||"";}
 async load(id,source=0,sideOutput=undefined){this.stop();const generation=++this.generation,meta=await api(`jobs/${id}/sequence`);if(generation!==this.generation)return;this.id=id;this.source=source;this.index=0;$("sequence-channel").value="0";this.meta=meta;if(this.meta.results[this.source]?.status!=="succeeded")this.source=this.meta.results.findIndex(r=>r.status==="succeeded");if(this.source<0)throw Error("该运行没有完整结果");this.sideOutput=sideOutput===undefined?this.preferredSideOutput():sideOutput;$("sequence-source").innerHTML=this.meta.results.map((r,i)=>`<option value="${i}" ${r.status!=="succeeded"?"disabled":""}>${esc(r.data_spec?.name||r.source?.path||i+1)}</option>`).join("");$("sequence-source").value=this.source;this.artifactOptions();if(["T","Q","W"].includes(this.item?.contract.sequence_axis))$("sequence-plot-axis").value="sequence";this.bridge.reset?.();await this.show(true);}
 async inspect(data,name){this.hide();this.id=null;this.source=0;this.sideOutput=null;this.index=data.index||0;this.meta={name,mode:"inspection",results:[{contract:data.contract,mappings:data.mappings,node_id:data.node_id}]};$("result-label").textContent=name;$("result-meta").textContent=data.contract.label;$("result-meta").title=data.contract.meaning||data.contract.semantic;$("split-result-empty").hidden=true;$("split-result-controls").hidden=true;$("split-results-tools").hidden=true;$("result-tools-empty").hidden=false;$("result-tools-empty").textContent="阶段检查仅显示当前样本；保存和导出请使用试算或正式运行。";$("sequence-x").value="samples";$("sequence-plot-axis").value=data.contract.shape.length?"component":"sequence";$("sequence-series").innerHTML='<option value="0">当前样本</option>';this.bridge.reset?.();this.bridge.signals();this.plot({...data,temporary:false});this.bridge.shown(true);}
 async show(reveal=false){const generation=++this.generation,id=this.id,item=this.item,c=item?.contract;if(!c)throw Error(item?.error||"此端口没有完整结果");if(c.sequence_axis==="W")$("sequence-x").value="samples";$("sequence-x").querySelector('[value="time"]').disabled=c.sequence_axis==="W";const histogram=c.semantic==="distribution"&&c.axes==="B";this.index=Math.max(0,Math.min(c.length-1,this.index||0));const data=await api(`jobs/${id}/sequence`,{source:this.source,side_output:this.sideOutput,index:this.index,channel:Number($("sequence-channel").value),profile:histogram||!["T","Q","W"].includes(c.sequence_axis)||(c.shape.reduce((a,b)=>a*b,1)>1&&$("sequence-plot-axis").value==="component"),display:this.bridge.display?.()});if(generation!==this.generation||id!==this.id)return;
  $("sequence-controls").hidden=false;$("split-results-tools").hidden=true;$("result-tools-empty").hidden=true;$("split-result-controls").hidden=true;$("split-result-empty").hidden=true;$("split-result-title").hidden=true;$("result-label").textContent=item.name||this.meta.name;$("result-meta").textContent=`${c.label}${data.temporary?" · 临时试算":""}`;$("result-meta").title=c.meaning||c.semantic;
  const isTemporal=["T","Q","W"].includes(c.sequence_axis),vector=c.shape.reduce((a,b)=>a*b,1)>1,component=histogram||!isTemporal||vector&&$("sequence-plot-axis").value==="component";$("sequence-plot-axis").value=component?"component":"sequence";$("sequence-plot-axis").querySelector('[value="sequence"]').disabled=!isTemporal;$("sequence-plot-axis").parentElement.hidden=data.kind!=="signals"||!vector||histogram;$("sequence-series").parentElement.hidden=data.kind!=="signals"||component||!vector;$("sequence-x").parentElement.hidden=data.kind!=="signals"||component||!isTemporal;$("sequence-channel-field").hidden=data.kind!=="image"||!c.axes.startsWith("P");$("sequence-nav").hidden=!(data.kind==="image"||component&&c.length>1);$("sequence-index").max=c.length-1;$("sequence-index").value=this.index;$("sequence-play").hidden=c.sequence_axis!=="T"||c.length<2;$("sequence-nav").querySelector("label").firstChild.textContent=c.sequence_axis==="W"?"窗长项":c.sequence_axis==="Q"?"延迟项":"结果样本";
  const columns=data.column_indices||[0],old=$("sequence-series").value;$("sequence-series").innerHTML=columns.slice(0,4096).map((v,i)=>`<option value="${i}">分量 ${v+1}</option>`).join("");if(Number(old)<columns.length)$("sequence-series").value=old;
  $("sequence-promote").disabled=$("artifact-to-result").disabled=this.meta.mode!=="run";const formats=$("artifact-export-format");for(const option of formats.options){if(["png","svg"].includes(option.value))option.textContent=(histogram?"首样本直方图 ":"曲线 ")+option.value.toUpperCase();option.disabled=["png","svg"].includes(option.value)?data.kind!=="signals":option.value==="image"?data.kind!=="image":option.value==="export"?!item.export_file:false;}if(formats.selectedOptions[0]?.disabled)formats.value="array";$("artifact-provenance").textContent=this.sideOutput?`${item.node_id?.slice(0,8)} · ${item.port||"data"} · ${item.contract.label}${data.temporary?" · 临时结果，正式运行后可加入来源":""}`:`流程主输出 · ${c.meaning||c.semantic}`;
  if(data.kind==="image"){this.comparison.hide();this.plotData=null;$("split-curve").hidden=true;await this.bridge.image(data.frame);if(generation!==this.generation||id!==this.id)return;const r=data.reference,q=r.correlation;$("sequence-readout").textContent=`${c.sequence_axis==="Q"?"延迟项":"样本"} ${this.index+1}/${c.length} · 来源帧 ${r.index==null?"合成 / 未知":r.index+1} · ${r.time==null?"时间未知":format(r.time)+" s"}${q?` · W=${q.window} · τ=${q.lag_samples} · 真实时差 ${format(q.lag_seconds)} s`:""}`;}else{this.bridge.signals();await this.comparison.open(data);$("sequence-series").parentElement.hidden=true;}if(generation!==this.generation||id!==this.id)return;this.bridge.shown(reveal);this.bridge.changed?.();
 }
 mappings(){const item=this.item,m=item?.mappings||item?.modules?.mappings||item?.upstream_modules?.mappings||{};return Array.isArray(m)?m:Object.entries(m).map(([node_id,value])=>({node_id,...value}));}
 plot(data){
  const c=data.contract,histogram=c.semantic==="distribution"&&c.axes==="B",isLag=c.sequence_axis==="Q",isWindow=c.sequence_axis==="W";
  const component=histogram||!["T","Q","W"].includes(c.sequence_axis)||(c.shape.reduce((a,b)=>a*b,1)>1&&$("sequence-plot-axis").value==="component");
  const physical=!component&&$("sequence-x").value==="time",columnIndices=data.column_indices||data.values[0]?.map((_,i)=>i)||[];
  const maps=this.mappings().filter(v=>c.axes.includes(v?.axis)&&(!histogram||v.edges?.length===c.shape[0]+1));
  const mainNode=this.item?.modules?.nodes?.filter(n=>n.enabled&&n.algorithm==="distribution"&&n.parameters.output==="main").at(-1);
  const mapping=maps.find(v=>v.node_id===(this.item?.node_id||mainNode?.id))||maps.at(-1),edges=histogram?mapping?.edges:null;
  let xs,series,xLabel,readIndex;
  if(component){
   const row=Math.min(data.values.length-1,data.indices.indexOf(this.index)>=0?data.indices.indexOf(this.index):0),values=data.values[row]||[];
   xs=columnIndices.map(i=>mapping?.coordinates?.[i]??(mapping?.edges?.[i]!=null?(mapping.edges[i]+mapping.edges[i+1])/2:histogram?i+.5:i+1));series=[values];
   xLabel=histogram?(edges?"强度":"箱索引（强度边界缺失）"):mapping?.axis==="R"?"剖面距离 / px":"分量索引";
   readIndex=i=>{
    if(!histogram)return `${xLabel} ${format(xs[i])} · 数值 ${format(values[i])}`;
    const bin=columnIndices[i],left=edges?.[bin]??bin,right=edges?.[bin+1]??bin+1,reference=data.references?.[row];
    return `样本 ${data.indices[row]+1}${reference?.index==null?"":` · 来源帧 ${reference.index+1}`} · ${xLabel} [${format(left)}, ${format(right)}${bin===c.shape[0]-1?"]":")"} · 频数 ${format(values[i])}`;
   };
  }else{
   xs=data.references.map((r,i)=>isWindow?r.correlation?.window:physical?(isLag?r.correlation?.lag_seconds:r.time):(isLag?r.correlation?.lag_samples:data.indices[i]+1));
   if(xs.some(x=>x==null))throw Error("真实时间存在未知值，请切换样本 / τ 横轴");
   const selected=Number($("sequence-series").value)||0,count=data.values[0]?.length||1,indices=count>1?[selected]:[0];
   series=indices.map(k=>data.values.map(row=>row[k]));xLabel=isWindow?"平均窗长 W / 上游样本数":physical?isLag?"实际配对时差 / s":"真实时间 / s":isLag?"有符号 τ / 上游样本":"编辑后样本号";
   readIndex=i=>{const r=data.references[i],q=r.correlation;return `${isWindow?`W=${q?.window} · τ=${q?.lag_samples} · 中心样本 ${(q?.center_sample??0)+1}`:isLag?`τ=${q?.lag_samples} · 真实时差 ${format(q?.lag_seconds)} s`:`样本 ${data.indices[i]+1} · ${r.time==null?"时间未知":format(r.time)+" s"}`} · ${series.map(s=>format(s[i])).join(" / ")}`;};
  }
  let[xmin,xmax]=histogram?[edges?.[0]??0,edges?.at(-1)??c.shape[0]]:extrema(xs),[ymin,ymax]=extrema(series.flat());
  if(!Number.isFinite(ymin)){ymin=0;ymax=1;}if(!Number.isFinite(xmin)){xmin=0;xmax=1;}if(xmin===xmax){xmin-=.5;xmax+=.5;}
  if(histogram){ymin=0;ymax=Math.max(1,Math.ceil(ymax*1.08));}else if(ymin===ymax){const p=Math.max(1e-6,Math.abs(ymin)*.02);ymin-=p;ymax+=p;}
  const host=$("split-curve"),W=Math.max(250,host.clientWidth-10),H=Math.max(150,host.clientHeight-10),L=W<400?52:64,R=18,T=25,B=46;
  host.dataset.plotOwner="sequence";
  const navigation=plotNavigation(host),key=JSON.stringify(["sequence",this.id,this.source,this.sideOutput,this.plotRevision,c.axes,component,physical]);
  // Index changes retain the view; data/port/axis changes start with the full domain.
  [xmin,xmax]=navigation.configure(key,[xmin,xmax],()=>{if(this.plotData)this.plot(this.plotData);},[ymin,ymax]);
  [ymin,ymax]=navigation.yRange;
  const X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>H-B-(y-ymin)/(ymax-ymin)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${histogram?"强度—频数直方图":esc(xLabel)+"数值曲线"}"><rect width="${W}" height="${H}" fill="#101923"/>`;
  const ticks=[...new Set(Array.from({length:4},(_,k)=>{const v=ymin+(ymax-ymin)*k/3;return histogram&&ymax-ymin>=3?Math.round(v):v;}))].filter(y=>y>=ymin&&y<=ymax);
  for(const y of ticks)svg+=`<path d="M${L},${Y(y)}H${W-R}" stroke="#2a394a"/><text class="plot-y-tick" x="${L-7}" y="${Y(y)+4}" text-anchor="end" fill="#a5bad0" font-size="10">${histogram&&Number.isInteger(y)?y:plotTick(y,ymax-ymin)}</text>`;
  for(let k=0;k<4;k++){const x=xmin+(xmax-xmin)*k/3;svg+=`<text class="plot-x-tick" x="${X(x)}" y="${H-B+19}" text-anchor="middle" fill="#a5bad0" font-size="10">${plotTick(x,xmax-xmin)}</text>`;}
  svg+=`<defs><clipPath id="sequence-plot-clip"><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}"/></clipPath></defs>`;
  if(histogram)svg+=`<text x="${L}" y="14" fill="#bdd0e1" font-size="11">频数</text>`;
  svg+='<g clip-path="url(#sequence-plot-clip)">';
  if(histogram){
   for(let i=0;i<xs.length;i++){
    const value=series[0][i];if(value==null||!Number.isFinite(value))continue;
    const bin=columnIndices[i],left=edges?.[bin]??bin,right=edges?.[bin+1]??bin+1,width=X(right)-X(left),gap=Math.min(1,width*.12);
    svg+=`<rect class="histogram-bin" data-bin="${bin}" data-frequency="${value}" x="${X(left)+gap/2}" y="${Y(value)}" width="${Math.max(0,width-gap)}" height="${Math.max(0,Y(0)-Y(value))}" fill="#91d7c5"><title>${esc(readIndex(i))}</title></rect>`;
   }
  }else{
   const colors=["#91d7c5","#e8b87d","#baadf2"];
   series.forEach((values,k)=>{
    let path="",open=false;for(let i=0;i<xs.length;i++){const y=values[i];if(y==null){open=false;continue;}path+=`${open?"L":"M"}${X(xs[i]).toFixed(2)},${Y(y).toFixed(2)} `;open=true;if(xs.length===1)svg+=`<circle cx="${X(xs[i])}" cy="${Y(y)}" r="3" fill="${colors[k%3]}"/>`;}
    svg+=`<path d="${path}" fill="none" stroke="${colors[k%3]}" stroke-width="1.7"/>`;
   });
  }
  svg+=`</g><text x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle" fill="#bdd0e1" font-size="11">${esc(xLabel)}</text></svg>`;host.hidden=false;host.innerHTML=svg;this.plotData=data;
  const inspect=i=>{if(xs.length)$("sequence-readout").textContent=readIndex(i)+(data.downsampled?" · 显示抽样；导出包含完整数据":"")+(data.temporary?" · 临时试算":"");};inspect(Math.max(0,xs.findIndex(x=>x>=xmin&&x<=xmax)));
  navigation.painted({left:L,right:W-R,top:T,bottom:H-B},e=>{const p=navigation.point(e);if(!p)return;const v=xmin+(p.x-L)/(W-L-R)*(xmax-xmin);let nearest=0;for(let i=1;i<xs.length;i++)if(Math.abs(xs[i]-v)<Math.abs(xs[nearest]-v))nearest=i;inspect(nearest);});
 }
 async play(){if(this.playing){this.stop();return;}if(this.item.contract.sequence_axis!=="T")return;this.playing=true;$("sequence-play").textContent="暂停";const next=async()=>{if(!this.playing)return;if(this.index+1>=this.item.contract.length){this.stop();return;}this.index++;try{await this.show();}catch(e){this.stop();this.bridge.report(e.message,true);return;}if(this.playing)this.timer=setTimeout(next,90);};await next();}
}
