import { api, download } from "./api.js";
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
const format=v=>v==null?"未知":Number(v).toPrecision(9);

export class SequenceResults{
  constructor(bridge){
    this.bridge=bridge;this.generation=0;
    $("result-pane").append($("sequence-controls"));
    this.resize=new ResizeObserver(()=>{if(this.plotData&&!$("split-curve").hidden){try{this.plot(this.plotData);}catch(error){bridge.report(error.message,true);}}});this.resize.observe($("split-curve"));
    const wrap=fn=>async()=>{try{await fn();}catch(e){this.stop();bridge.report(e.message,true);}};
    $("sequence-source").onchange=wrap(async()=>{this.source=Number($("sequence-source").value);this.index=0;await this.show();});
    $("sequence-index").onchange=wrap(async()=>{this.index=Number($("sequence-index").value);await this.show();});
    $("sequence-channel").onchange=wrap(()=>this.show());
    $("sequence-x").onchange=wrap(()=>this.show());
    $("sequence-play").onclick=wrap(()=>this.play());
    $("sequence-promote").onclick=wrap(()=>bridge.promote(this.id,this.source));
    document.querySelectorAll("[data-sequence-export]").forEach(b=>b.onclick=wrap(()=>download(this.id,b.dataset.sequenceExport,this.source)));
  }
  stop(){this.playing=false;clearTimeout(this.timer);$("sequence-play").textContent="播放结果";}
  hide(){this.stop();this.generation++;this.plotData=null;$("sequence-controls").hidden=true;}
  async load(id,source=0){
    this.stop();this.id=id;this.source=source;this.index=0;this.meta=await api(`jobs/${id}/sequence`);if(this.meta.results[this.source]?.status!=="succeeded")this.source=this.meta.results.findIndex(r=>r.status==="succeeded");
    $("sequence-source").innerHTML=this.meta.results.map((r,i)=>`<option value="${i}" ${r.status!=="succeeded"?"disabled":""}>${i+1}. ${esc(r.data_spec?.name||r.source.path)}</option>`).join("");$("sequence-source").value=this.source;
    await this.show();
  }
  async show(){
    const generation=++this.generation,id=this.id,item=this.meta.results[this.source],c=item.contract;if(!c)throw Error(item.error||"此来源没有完整结果");
    this.index=Math.max(0,Math.min(c.length-1,this.index||0));
    const data=await api(`jobs/${id}/sequence`,{source:this.source,index:this.index,channel:Number($("sequence-channel").value),display:this.bridge.display?.()});if(generation!==this.generation||id!==this.id)return;
    $("sequence-controls").hidden=false;$("split-result-controls").hidden=true;$("split-result-empty").hidden=true;$("split-result-title").hidden=true;
    $("result-label").textContent=this.meta.name;$("result-meta").textContent=`${c.label} · ${c.semantic}`;
    $("sequence-nav").hidden=data.kind!=="image";$("sequence-x").parentElement.hidden=data.kind!=="signals";
    $("sequence-channel-field").hidden=data.kind!=="image"||!c.axes.startsWith("P");
    $("sequence-index").max=c.length-1;$("sequence-index").value=this.index;
    for(const b of document.querySelectorAll("[data-sequence-export]")){const f=b.dataset.sequenceExport;b.hidden=["png","svg"].includes(f)?data.kind!=="signals":f==="image"?data.kind!=="image":false;}
    $("sequence-promote").disabled=this.meta.mode!=="run";
    if(data.kind==="image"){
      this.plotData=null;
      $("split-curve").hidden=true;await this.bridge.image(data.frame);const r=data.reference,q=r.correlation;
      $("sequence-readout").textContent=`样本 ${this.index+1}/${c.length} · 来源帧 ${r.index==null?"合成 / 未知":r.index+1} · 时间 ${format(r.time)} s${q?` · τ=${q.lag_samples} 当前输入样本 · 实际时差 ${format(q.lag_seconds)} s（${format(q.lag_seconds_min)}～${format(q.lag_seconds_max)}）`:""}`;
    }else{this.bridge.signals();this.plot(data);}
    this.bridge.shown();
  }
  plot(data){
    const c=data.contract,isLag=c.sequence_axis==="Q",physical=$("sequence-x").value==="time";
    const xs=data.references.map((r,i)=>physical?(isLag?r.correlation?.lag_seconds:r.time):(isLag?r.correlation?.lag_samples:data.indices[i]+1));
    if(xs.some(x=>x==null)){if(physical)throw Error("存在未知真实时间；请选择样本 / τ 横轴");}
    const ys=data.values.flat().filter(v=>v!=null),lo=ys.length?Math.min(...ys):0,hi=ys.length?Math.max(...ys):1;
    let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=lo,ymax=hi;if(xmin===xmax){xmin-=.5;xmax+=.5;}if(ymin===ymax){ymin-=Math.max(1e-6,Math.abs(ymin)*.01);ymax+=Math.max(1e-6,Math.abs(ymax)*.01);}
    const host=$("split-curve"),W=Math.max(260,host.clientWidth-20),H=Math.max(240,host.clientHeight-20),L=68,R=40,T=52,B=65,X=x=>L+(x-xmin)/(xmax-xmin)*(W-L-R),Y=y=>H-B-(y-ymin)/(ymax-ymin)*(H-T-B),colors=["#67d2c6","#e9b577","#a8adf0","#ed8a9b"];
    const count=data.values[0]?.length||1;
    let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${isLag?"有符号延迟关联曲线":"时间信号曲线"}"><rect width="${W}" height="${H}" fill="#111a24"/>`;
    for(let k=0;k<5;k++){const y=ymin+(ymax-ymin)*k/4;svg+=`<path d="M${L},${Y(y)}H${W-R}" stroke="#273544"/><text x="${L-8}" y="${Y(y)+4}" text-anchor="end" fill="#a7b9c9" font-size="11">${Number(y).toPrecision(5)}</text>`;}
    const ticks=W<430?3:5;for(let k=0;k<ticks;k++){const x=xmin+(xmax-xmin)*k/(ticks-1);svg+=`<text x="${X(x)}" y="${H-B+22}" text-anchor="middle" fill="#a7b9c9" font-size="11">${Number(x).toPrecision(5)}</text>`;}
    for(let channel=0;channel<count;channel++){let path="",open=false;for(let j=0;j<xs.length;j++){const y=data.values[j][channel];if(y==null){open=false;continue;}path+=`${open?"L":"M"}${X(xs[j]).toFixed(2)},${Y(y).toFixed(2)} `;if(xs.length===1)svg+=`<circle cx="${X(xs[j])}" cy="${Y(y)}" r="3" fill="${colors[channel%colors.length]}"/>`;open=true;}svg+=`<path d="${path}" fill="none" stroke="${colors[channel%colors.length]}" stroke-width="1.8"/>`;}
    svg+=`<text x="${W/2}" y="${H-12}" text-anchor="middle" fill="#bdccda" font-size="13">${physical?isLag?"实际配对时差 / s":"真实时间 / s":isLag?"有符号 τ / 当前上游样本":"编辑后样本号"}</text></svg>`;
    host.hidden=false;host.innerHTML=svg;this.plotData=data;
    const inspect=index=>{const r=data.references[index],q=r.correlation;$("sequence-readout").textContent=`${isLag?`τ=${q?.lag_samples} · 实际时差 ${format(q?.lag_seconds)} s（${format(q?.lag_seconds_min)}～${format(q?.lag_seconds_max)}）`:`样本 ${data.indices[index]+1} · 时间 ${format(r.time)} s`}\n${data.values[index].map((v,k)=>`${count>1?"通道 "+(k+1)+"：":""}${format(v)}`).join(" · ")}${data.downsampled?`\n显示抽样 ${data.indices.length}/${data.total}；导出包含完整数据。`:""}`;};inspect(0);
    host.onpointermove=e=>{const rect=host.querySelector("svg").getBoundingClientRect(),v=xmin+((e.clientX-rect.left)/rect.width*W-L)/(W-L-R)*(xmax-xmin);let nearest=0;for(let i=1;i<xs.length;i++)if(Math.abs(xs[i]-v)<Math.abs(xs[nearest]-v))nearest=i;inspect(nearest);};
  }
  async play(){if(this.playing){this.stop();return;}this.playing=true;$("sequence-play").textContent="暂停";const next=async()=>{if(!this.playing)return;if(this.index+1>=this.meta.results[this.source].contract.length){this.stop();return;}this.index++;try{await this.show();}catch(e){this.stop();this.bridge.report(e.message,true);return;}if(this.playing)this.timer=setTimeout(next,90);};await next();}
}
