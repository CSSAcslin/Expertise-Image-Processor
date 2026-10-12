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
export async function download(id, file, source = 0, sideOutput = null) {
  const response = await fetch(
    `/api/jobs/${id}/download?file=${encodeURIComponent(file)}&source=${source}${sideOutput?"&side_output="+encodeURIComponent(sideOutput):""}`,
    { headers: { "X-EIP-Token": token } },
  );
  if (!response.ok) throw new Error((await response.json()).detail);
  const blob = await response.blob(),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `${id}-${source + 1}.${{ manifest: "json", csv: "csv", image: "tif",array:"npy",references:"jsonl",png:"png",svg:"svg",export:"dat" }[file]||file}`;
  const disposition=response.headers.get("Content-Disposition")||"";const named=disposition.match(/filename\*=UTF-8\x27\x27([^;]+)/i)||disposition.match(/filename="([^"]+)"/i);if(named){try{a.download=decodeURIComponent(named[1]);}catch{}}
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export async function downloadSplit(id,file,timeIndex=0,lagIndex=0) {
  const response = await fetch(`/api/split/jobs/${encodeURIComponent(id)}/download?file=${encodeURIComponent(file)}&time_index=${timeIndex}&lag_index=${lagIndex}`,{headers:{"X-EIP-Token":token}});
  if(!response.ok) throw new Error((await response.json()).detail);
  const url = URL.createObjectURL(await response.blob()),a=document.createElement("a");
  a.href=url;
  a.download=`${id}-${{manifest:"run.json",csv:"curves.csv",png:"curves.png",svg:"curves.svg",curve_json:"curves.json",windows:"windows.csv",image:`t${timeIndex}-q${lagIndex}.tif`}[file]}`;
  a.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
}
