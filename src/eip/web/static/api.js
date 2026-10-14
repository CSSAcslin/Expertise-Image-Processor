let token = "";
export function session(value) {
  token = value;
}
export async function api(path, value) {
  const response = await fetch("/api/" + path, {
    method: value === undefined ? "GET" : "POST",
    headers: {
      "X-EIP-Token": token,
      ...(value === undefined ? {} : { "Content-Type": "application/json" }),
    },
    ...(value === undefined ? {} : { body: JSON.stringify(value) }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "请求失败");
  return data;
}
function saveDownload(data){
  const link=document.createElement("a");link.href=data.url;link.download=data.filename;document.body.append(link);link.click();link.remove();
}
export async function download(id,file,source=0,sideOutput=null){
  saveDownload(await api(`jobs/${id}/download-link`,{file,source,side_output:sideOutput}));
}
export async function downloadSplit(id,file,timeIndex=0,lagIndex=0){
  saveDownload(await api(`split/jobs/${encodeURIComponent(id)}/download-link`,{file,time_index:timeIndex,lag_index:lagIndex}));
}
