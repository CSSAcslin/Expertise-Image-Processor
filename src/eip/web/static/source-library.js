import { svgIcon } from "./icons.js";
const $=id=>document.getElementById(id);
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
export function setupSourceLibrary(bridge){
  const selected=new Set(); let key="";
  const guard=fn=>async(...args)=>{try{return await fn(...args);}catch(e){bridge.report(e.message,true);}};
  const inputs=()=>bridge.sources().filter(s=>s.role==="input");
  const rows=()=>{const query=$("source-search").value.trim().toLocaleLowerCase();return inputs().filter(s=>(s.display_name||s.meta.name).toLocaleLowerCase().includes(query)||s.source.path.toLocaleLowerCase().includes(query));};
  const blockers=id=>bridge.jobs().filter(j=>j.source_ids?.includes(id)&&["queued","running","cancelling"].includes(j.status));
  function selection(){
    const visible=rows(), count=visible.filter(s=>selected.has(s.id)).length;
    $("source-select-all").checked=!!visible.length&&count===visible.length;
    $("source-select-all").indeterminate=count>0&&count<visible.length;
    $("source-select-all").disabled=!visible.length;
    $("source-selection-count").textContent=selected.size?selected.size+" 已选":"";
    $("source-batch-remove").disabled=!selected.size;
    $("source-batch-run").disabled=!selected.size;
    for(const row of $("source-list").querySelectorAll("[data-source]")){
      const checked=selected.has(row.dataset.source);row.classList.toggle("batch-selected",checked);
      row.querySelector("[data-source-check]").checked=checked;
    }
  }
  function update(){
    const all=inputs(), available=new Set(all.map(s=>s.id));for(const id of selected)if(!available.has(id))selected.delete(id);
    $("source-count").textContent=all.length;
    const list=$("source-list"), data=rows(), fingerprint=JSON.stringify([bridge.activeSource?.(),data.map(s=>[s.id,s.display_name,blockers(s.id).map(j=>[j.id,j.name])])]);
    if(key!==fingerprint){
      key=fingerprint;const scroll=list.scrollTop, focus=document.activeElement?.closest("[data-source]")?.dataset.source;
      list.innerHTML=data.map(s=>{
        const name=s.display_name||s.meta.name,busy=blockers(s.id), reason=busy.length?"任务正在使用："+busy.map(j=>j.name).join("、"):"移除来源引用（保留文件）";
        const kind=s.lineage?.length?"派生输入":s.meta.kind==="video"?"视频":s.meta.kind==="sequence"?"序列":s.meta.kind==="dataset"?"数据集":"图像";
        return `<article class="source-card ${bridge.activeSource?.()===s.id?"selected":""}" data-source="${esc(s.id)}" tabindex="0" role="button" aria-label="预览 ${esc(name)}"><input type="checkbox" data-source-check="${esc(s.id)}" aria-label="选择 ${esc(name)}"/><img src="${s.thumbnail}" alt=""/><div class="source-copy"><strong title="${esc(s.source.path)}">${esc(name)}</strong><small>${kind} · ${esc(s.meta.contract?.label||s.meta.shape.join(" × "))} · ${esc(s.meta.dtype)}</small></div><button class="icon-button source-info" data-info="${esc(s.id)}" title="属性与来源" aria-label="属性与来源">${svgIcon("info")}</button><button class="icon-button source-remove" data-remove="${esc(s.id)}" title="${esc(reason)}" aria-label="${esc(reason)}" ${busy.length?"disabled":""}>${svgIcon("trash")}</button></article>`;
      }).join("")||'<div class="library-empty">'+(all.length?"无匹配来源":"导入图像、视频或序列")+"</div>";
      list.scrollTop=scroll;
      for(const row of list.querySelectorAll("[data-source]")){
        row.onclick=guard(e=>{if(e.target.closest("button,input"))return;return bridge.selectSource(row.dataset.source);});
        row.onkeydown=guard(e=>{
          if(e.target!==row)return;
          if(["Enter"," "].includes(e.key)){e.preventDefault();e.stopPropagation();return bridge.selectSource(row.dataset.source);}
          if(["ArrowUp","ArrowDown","Home","End"].includes(e.key)){e.preventDefault();e.stopPropagation();const nodes=[...list.querySelectorAll("[data-source]")],i=nodes.indexOf(row),next=e.key==="Home"?nodes[0]:e.key==="End"?nodes.at(-1):nodes[Math.max(0,Math.min(nodes.length-1,i+(e.key==="ArrowDown"?1:-1)))];next.focus();return bridge.selectSource(next.dataset.source);}
        });
        row.querySelector("[data-source-check]").onchange=e=>{const id=row.dataset.source;e.target.checked?selected.add(id):selected.delete(id);selection();};
        row.querySelector("[data-info]").onclick=guard(e=>{e.stopPropagation();return bridge.sourceDetail(row.dataset.source);});
        row.querySelector("[data-remove]").onclick=guard(e=>{e.stopPropagation();return bridge.removeReference(row.dataset.source);});
      }
      [...list.querySelectorAll("[data-source]")].find(row=>row.dataset.source===focus)?.focus({preventScroll:true});
    }
    selection();
  }
  $("source-search").oninput=update;
  $("source-search").onkeydown=e=>{if(e.key==="Escape"){e.preventDefault();$("source-search").value="";update();}};
  $("source-select-all").onchange=e=>{for(const row of rows())e.target.checked?selected.add(row.id):selected.delete(row.id);selection();};
  $("source-batch-remove").onclick=guard(async()=>{
    const button=$("source-batch-remove");button.classList.add("is-loading");
    try{const result=await bridge.removeSources([...selected]);for(const id of result.removed)selected.delete(id);update();bridge.report(`已移除 ${result.removed.length} 个来源${result.failed.length?"；"+result.failed.map(f=>f.error).join("；"):""}`,!!result.failed.length);$("source-selection-count").title=result.failed.map(f=>f.error).join("\n");}finally{button.classList.remove("is-loading");}
  });
  $("source-batch-run").onclick=guard(()=>bridge.runSources([...selected]));
  return {update,selectedIds:()=>[...selected]};
}
