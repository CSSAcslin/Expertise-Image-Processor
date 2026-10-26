import { api } from "./api.js";
import { setupResultLibrary } from "./result-library.js";
import { setupSourceLibrary } from "./source-library.js";
const $=id=>document.getElementById(id);
const mib=v=>v==null?"未知":(v/1048576).toFixed(1)+" MiB";
export function setupManagement(bridge){
 const guard=fn=>async(...args)=>{try{return await fn(...args);}catch(e){bridge.report(e.message,true);}};
 const runtimeDetails=document.createElement("details"),runtimeInfo=document.createElement("pre");
 runtimeDetails.className="resource-runtime";runtimeDetails.innerHTML="<summary>运行环境</summary>";
 runtimeInfo.className="resource-readout";runtimeDetails.append(runtimeInfo);$("resource-readout").after(runtimeDetails);
 $("resource-refresh").textContent="重新检测";$("resource-refresh").title="重新检测 GPU 并刷新内存占用";
 async function resources(fill=false,refresh=false){
  const data=await api(refresh?"resources/refresh":"resources",refresh?{}:undefined),s=data.settings;
  if(fill)for(const k of Object.keys(s))$("resource-"+k).value=s[k];
  const rss=data.processes.filter(p=>p.resident_bytes!=null).reduce((sum,p)=>sum+p.resident_bytes,0);
  $("resource-readout").textContent=`当前进程内存 ${mib(rss)} · 可用系统内存 ${mib(data.system.available_bytes)}\n处理预算 ${data.processing_mb} MiB（任务 ${data.job_mb} / 检查 ${data.inspection_mb}）\n原值缓存 ${mib(data.frame_cache_bytes)} / ${s.preview_mb/2} MiB（${data.frame_cache_count} 帧）\n${data.processes.map(p=>p.name+" "+mib(p.resident_bytes)).join(" · ")}`;
  $("resource-device").textContent=data.devices.gpu?"CUDA · "+data.devices.gpu_name:"CPU 可用 · "+data.devices.reason;
  runtimeDetails.hidden=!data.runtime;
  if(data.runtime)runtimeInfo.textContent=`Python ${data.runtime.python_version}\n${data.runtime.python_executable}\n程序位置\n${data.runtime.code_directory}`;
  $("resource-save").disabled=data.busy;
  $("resource-save").title=data.busy?"等待或取消任务 / 阶段检查后保存":"保存至当前工作区";
 }
 $("open-run-settings").addEventListener("click",guard(()=>resources(true)));
 $("resource-refresh").onclick=guard(async()=>{const button=$("resource-refresh");button.disabled=true;button.textContent="检测中…";try{await resources(false,true);}finally{button.disabled=false;button.textContent="重新检测";}});
 $("resource-save").onclick=guard(async()=>{
  const settings={};for(const k of["memory_mb","preview_mb","cache_mb","gpu_mb","backend"]){const el=$("resource-"+k);if(!el.reportValidity())return;settings[k]=k==="backend"?el.value:Number(el.value);}
  await api("resources",settings);await resources();bridge.report("统一资源预算已保存，后续检查与运行使用新设置");
 });
 $("resource-clear").onclick=guard(async()=>{await api("resources/clear",{});await resources();bridge.report("空闲缓存已释放，重新定位即可恢复原值查询");});
 $("resource-inherit-flow").onclick=()=>{bridge.inheritFlow();bridge.report("当前流程已继承统一设备与内存预算，可撤销");};
 const results=setupResultLibrary(bridge),inputs=setupSourceLibrary(bridge);
 return {...results,inputs,resources,update:()=>{inputs.update();results.update();}};
}
