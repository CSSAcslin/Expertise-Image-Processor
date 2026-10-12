// Historical split-result reader only. New workbench flows use the common ModuleEditor.
import { api,downloadSplit } from "./api.js";
import { detailHTML,bindRename } from "./job-detail.js";
import { plotNavigation, plotTick } from "./plot-navigation.js";
const $=id=>document.getElementById(id),clone=v=>structuredClone(v),esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const number=n=>n==null?"未知":Number(n).toPrecision(6),coefficient=n=>n==null?"无效":Number(n).toPrecision(10);
const label={statistics:"时序统计",trial:"系数图试算",curve:"延迟扫描",run:"固定延迟图"};
export class SplitController{
 constructor(bridge){this.bridge=bridge;this.viewer=bridge.viewer;this.resultViewer=bridge.resultViewer;this.scene=this.viewer.scene;this.resultScene=this.resultViewer.scene;this.mode="general";this.records={};this.jobs=[];this.requested=new Map();this.views={};this.resultAutoFit=true;
  $("split-result-select").onchange=this.wrap(()=>this.loadResult($("split-result-select").value));$("split-result-time").onchange=this.wrap(()=>{this.timeIndex=Number($("split-result-time").value);return this.showSlice();});$("split-result-lag").onchange=this.wrap(()=>{this.lagIndex=Number($("split-result-lag").value);return this.resultData?.run.array_shape?this.showSlice():this.showCurve();});$("split-result-play").onclick=this.wrap(()=>this.playResult());$("split-show-map").onclick=this.wrap(()=>this.showSlice());$("split-show-curve").onclick=()=>this.showCurve();$("split-curve-axis").onchange=()=>this.showCurve();for(const id of["split-display-low","split-display-high"])$(id).onchange=this.wrap(()=>this.resultData?.run.array_shape?this.showSlice():null);
  $("split-result-rename").onclick=this.wrap(async()=>{if(!this.resultId)throw Error("请先选择结果");await api(`jobs/${this.resultId}/rename`,{name:$("split-result-name").value});await this.bridge.poll();await this.loadResult(this.resultId);});document.querySelectorAll("[data-split-export]").forEach(b=>b.onclick=this.wrap(()=>downloadSplit(this.resultId,b.dataset.splitExport,this.timeIndex||0,this.lagIndex||0)));$("split-promote").onclick=this.wrap(async()=>{const opened=await api(`split/jobs/${this.resultId}/promote`,{lag_index:this.lagIndex||0});await this.bridge.promote(opened);});$("split-restore").onclick=this.wrap(async()=>{const r=await api("split/restore",{path:$("split-restore-path").value.trim()});await this.bridge.poll();await this.loadResult(r.id);});
  new ResizeObserver(()=>{if(this.resultView==="curve"&&$("split-curve").dataset.plotOwner==="split")this.showCurve(true);}).observe($("split-curve"));
 }
 wrap(fn){return async(...args)=>{try{await fn(...args);}catch(e){this.stop();this.bridge.report(e.message,true);}};}
 get context(){return this.bridge.context();}get sourceId(){return this.context.source?.id;}get record(){return this.records[this.sourceId]||null;}get cfg(){return this.record?.config;}async contextChanged(){}async setMode(){this.mode="general";document.body.dataset.analysis="general";}async showRaw(s){if(s)await this.viewer.set(s);}inputChanged(){}draw(){this.viewer.draw();}drawResult(){this.resultViewer.draw();}previewTransform(){return this.scene;}
 serialize(){return {records:clone(this.records),resultId:this.resultId||null,resultViews:clone(this.views)};}restore(v){if(v?.records)this.records=clone(v.records);this.views=clone(v?.resultViews||{});}signature(){return "history";}
 updateJobs(jobs){this.jobs=jobs;const list=jobs.filter(j=>j.mode==="split"&&j.status==="succeeded");$("split-result-select").innerHTML='<option value="">选择历史结果</option>'+list.map(j=>`<option value="${esc(j.id)}">${esc(j.name)}</option>`).join("");$("split-result-select").value=this.resultId||"";}
 async completed(job){if(job.error)this.bridge.report(job.error,true);}
 async loadResult(id){if(!id)return;this.stop();this.resultId=id;const data=await api(`split/jobs/${id}`);if(this.resultId!==id)return;this.resultData=data;const saved=this.views[id]||{};this.timeIndex=Math.min((data.run.array_shape?.[0]||1)-1,saved.timeIndex||0);this.lagIndex=Math.min(data.run.lags.length-1,saved.lagIndex??Math.max(0,data.run.lags.indexOf(0)));$("split-curve-axis").value=saved.axis||"frame";$("sequence-controls").hidden=true;$("sequence-nav").hidden=true;$("split-results-tools").hidden=false;$("result-tools-empty").hidden=true;$("split-result-controls").hidden=false;$("split-result-title").hidden=false;$("split-result-title").textContent=data.run.display_name||label[data.run.task];$("split-result-name").value=data.run.display_name||id;$("split-result-select").value=id;$("split-result-summary").textContent=`历史记录 · ${label[data.run.task]} · W=${data.run.config.window} · ${data.directory}`;$("split-result-time").max=(data.run.array_shape?.[0]||1)-1;$("split-result-time").value=this.timeIndex;$("split-result-time").disabled=!data.run.array_shape;$("split-result-play").disabled=!data.run.array_shape||data.run.array_shape[0]<2;$("split-result-lag").innerHTML=data.run.lags.map((q,i)=>`<option value="${i}">${q} 样本</option>`).join("");$("split-result-lag").value=this.lagIndex;$("split-result-nav").hidden=data.run.task==="statistics";$("split-show-map").hidden=!data.run.array_shape;$("split-show-curve").hidden=!data.curve_result;$("split-promote").disabled=!data.run.array_shape;document.querySelectorAll("[data-split-export]").forEach(b=>{b.disabled=["csv","png","svg","curve_json"].includes(b.dataset.splitExport)?!data.curve_result:["image","windows"].includes(b.dataset.splitExport)?!data.run.array_shape:false;});if(data.curve_result)this.showCurve();else if(data.run.array_shape)await this.showSlice();else{const stat=await api(`split/jobs/${id}/statistics`,{view:"mean_left",index:0});this.resultView="statistics";$("split-curve").hidden=true;this.bridge.resultShown(stat.frame);await this.resultViewer.set(stat.frame);this.fitResult();}this.bridge.changed();}
 async showSlice(){if(!this.resultData?.run.array_shape)throw Error("该记录无矩阵结果");const id=this.resultId,ti=this.timeIndex||0,qi=this.lagIndex||0,data=await api(`split/jobs/${id}/slice`,{time_index:ti,lag_index:qi,display:{low:Number($("split-display-low").value),high:Number($("split-display-high").value),gamma:1,colormap:"heat"}});if(id!==this.resultId||ti!==this.timeIndex||qi!==this.lagIndex)return;this.resultView="result";$("split-curve").hidden=true;$("split-result-empty").hidden=true;this.resultScene.width=data.frame.base_shape[1];this.resultScene.height=data.frame.base_shape[0];this.bridge.resultShown(data.frame);await this.resultViewer.set(data.frame);this.fitResult();const r=data.window_ref;$("split-result-caption").textContent=`中心 ${r.center_index+1} · τ=${r.lag_frames} · 有效像素 ${data.valid_pixels}`;$("split-result-note").textContent=`实际时差 ${number(r.lag_seconds)} s · 有效样本 ${data.count_min}～${data.count_max}`;this.saveResultView();}
 saveResultView(){if(this.resultId)this.views[this.resultId]={timeIndex:this.timeIndex||0,lagIndex:this.lagIndex||0,axis:$("split-curve-axis").value};this.bridge.changed();}
   showCurve(viewOnly=false) {
    const data=this.resultData?.curve_result;
    if(!data) {this.bridge.report("选定中心帧和点 / ROI，点击“固定帧曲线”计算该结果",true);return;}
    if(!viewOnly)this.bridge.resultShown(null);this.resultView="curve";$("split-curve").hidden=false;$("split-result-empty").hidden=true;$("split-curve").dataset.plotOwner="split";
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
    const navigation=plotNavigation($("split-curve"));
    [xmin,xmax]=navigation.configure(JSON.stringify(["split",this.resultId,time,factor]),[xmin,xmax],()=>this.showCurve(true),[lo,hi]);
    [lo,hi]=navigation.yRange;
    const X=x=>72+(x-xmin)/(xmax-xmin)*620,Y=y=>350-(y-lo)/(hi-lo)*306;
    const colors=["#87dfc5","#e6b386","#9db4f3","#dc9dd9","#82cadd"];
    const parts=['<svg viewBox="0 0 720 425" role="img" aria-label="g² 随正负延迟变化曲线">'];
    for(let i=0;i<=5;i++) {const x=xmin+(xmax-xmin)*i/5;parts.push(`<line x1="${X(x)}" y1="44" x2="${X(x)}" y2="350" stroke="#25384a"/><text class="plot-x-tick" x="${X(x)}" y="372" text-anchor="middle" fill="#9ab0c8" font-size="11">${plotTick(x,xmax-xmin)}</text>`);}
    for(let i=0;i<=4;i++) {const y=lo+(hi-lo)*i/4;parts.push(`<line x1="72" y1="${Y(y)}" x2="692" y2="${Y(y)}" stroke="#25384a"/><text class="plot-y-tick" x="63" y="${Y(y)+4}" text-anchor="end" fill="#9ab0c8" font-size="11">${plotTick(y,hi-lo)}</text>`);}
    if(xmin<=0&&xmax>=0) parts.push(`<line x1="${X(0)}" y1="44" x2="${X(0)}" y2="350" stroke="#7597ad" stroke-dasharray="4 5"/>`);
    if(reference!=null&&reference>=lo&&reference<=hi)parts.push(`<line x1="72" y1="${Y(reference)}" x2="692" y2="${Y(reference)}" stroke="#7597ad" stroke-dasharray="4 5"/>`);
    for(const [i,c] of curves.entries()) {
      let path="",connected=false;
      c.g2.forEach((v,j)=>{if(v==null){connected=false;return;}path+=`${connected?"L":"M"}${X(xs[j])},${Y(v)} `;connected=true;});
      parts.push(`<path d="${path}" stroke="${colors[i%colors.length]}" fill="none" stroke-width="1.7" clip-path="url(#split-plot-clip)"/>`);
      if(i<7||curves.length===8&&i===7) parts.push(`<text x="${80+(i%4)*150}" y="${18+Math.floor(i/4)*15}" fill="${colors[i%colors.length]}" font-size="10">${esc(c.name)}${c.estimator==="pixel_mean"?" · 像素平均":c.estimator==="intensity_sum"?" · 总强度":""}</text>`);
    }
    if(curves.length>8) parts.push(`<text x="530" y="33" fill="#aac2d8" font-size="10">共 ${curves.length} 条，数值见右侧</text>`);
    const selected=Math.max(0,Math.min(refs.length-1,this.lagIndex||0));
    parts.push(`<defs><clipPath id="split-plot-clip"><rect x="72" y="44" width="620" height="306"/></clipPath></defs><text x="382" y="405" text-anchor="middle" fill="#aac2d8" font-size="12">${time?`实际延迟 τ (${factor===1e6?"μs":"s"})`:"有符号帧延迟 q（右路 i+q）"}</text><text x="22" y="198" fill="#aac2d8" font-size="13">g²</text><line id="split-curve-cursor" x1="${X(xs[selected])}" x2="${X(xs[selected])}" y1="44" y2="350" stroke="#c7ddec" clip-path="url(#split-plot-clip)"/></svg>`);
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
    const select=e=>{this.lagIndex=hover(e);$("split-result-lag").value=this.lagIndex;this.saveResultView();};
    $("split-result-caption").textContent=`固定中心帧 ${refs[0].center_index+1} · W=${data.config.window} · ${refs.length} 个有符号延迟${data.result_semantics==="processed_g2"?" · g² 后处理":""}`;
    const ref=refs[selected];
    $("split-result-note").textContent=`左窗 ${ref.left_start+1}～${ref.left_end} · 右窗 ${ref.right_start+1}～${ref.right_end} · 实际时差 ${number(ref.lag_seconds)} s`;
    $("split-curve-readout").textContent=`${curves.map(c=>c.name+" · "+(c.estimator==="pixel_mean"?"逐像素 g² 空间平均":c.estimator==="intensity_sum"?"先汇总强度再算 g²":"单像素 g²")).join("\n")}\n选定 q=${ref.lag_frames}：${curves.map(c=>coefficient(c.g2[selected])).join(" / ")}\n曲线使用完整 W 帧窗口，鼠标移动查看数值。`;
    navigation.painted({left:72,right:692,top:44,bottom:350},hover,select);
    if(!viewOnly)this.saveResultView();
  }

 stop(){clearTimeout(this.playTimer);this.playing=false;$("split-result-play").textContent="播放结果";}
 async playResult(){if(this.playing){this.stop();return;}this.playing=true;$("split-result-play").textContent="暂停";const next=async()=>{if(!this.playing)return;if(this.timeIndex+1>=this.resultData.run.array_shape[0]){this.stop();return;}this.timeIndex++;$("split-result-time").value=this.timeIndex;try{await this.showSlice();}catch(e){this.stop();this.bridge.report(e.message,true);return;}if(this.playing)this.playTimer=setTimeout(next,90);};await next();}
 fitResult(){if(!this.resultViewer.snapshot)return;this.resultScene.width=this.resultViewer.snapshot.base_shape[1];this.resultScene.height=this.resultViewer.snapshot.base_shape[0];this.resultScene.zoom=1;this.resultScene.panX=this.resultScene.panY=0;this.resultViewer.draw();}
 pixel(snapshot,x,y){const box=$("split-result-pixel");box.textContent=`x ${x} · y ${y}`;clearTimeout(this.pixelTimer);this.pixelTimer=setTimeout(async()=>{try{const data=await api(`frames/${snapshot.token}/pixel?x=${x}&y=${y}`);box.textContent=`x ${x} · y ${y} · ${Array.isArray(data.value)?data.value.join(" / "):data.value}`;}catch{}},130);}
 async jobDetail(job){$("job-dialog").showModal();$("job-detail").innerHTML=detailHTML(job)+(job.status==="succeeded"?'<button id="split-job-open" class="primary">查看结果</button>':"");bindRename(job,async()=>{await this.bridge.poll();await this.jobDetail(await api(`jobs/${job.id}`));},this.bridge.report);$("split-job-open")?.addEventListener("click",this.wrap(async()=>{await this.loadResult(job.id);$("job-dialog").close();}));}
}
