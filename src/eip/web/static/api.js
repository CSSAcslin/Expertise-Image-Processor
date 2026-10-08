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
export async function download(id, file, source = 0) {
  const response = await fetch(
    `/api/jobs/${id}/download?file=${file}&source=${source}`,
    { headers: { "X-EIP-Token": token } },
  );
  if (!response.ok) throw new Error((await response.json()).detail);
  const blob = await response.blob(),
    url = URL.createObjectURL(blob),
    a = document.createElement("a");
  a.href = url;
  a.download = `${id}-${source + 1}.${{ manifest: "json", csv: "csv", image: "tif" }[file]}`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
