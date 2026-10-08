// ROI is always expressed in the current input array's integer pixel coordinates.
export function validROI(roi, width, height) {
  return (
    !!roi &&
    [roi.x, roi.y, roi.width, roi.height].every(Number.isInteger) &&
    roi.x >= 0 &&
    roi.y >= 0 &&
    roi.width >= 1 &&
    roi.height >= 1 &&
    roi.x + roi.width <= width &&
    roi.y + roi.height <= height
  );
}
export function createROI(a, b, width, height) {
  const x1 = Math.max(0, Math.min(width, Math.round(a.x))),
    y1 = Math.max(0, Math.min(height, Math.round(a.y)));
  const x2 = Math.max(0, Math.min(width, Math.round(b.x))),
    y2 = Math.max(0, Math.min(height, Math.round(b.y)));
  const roi = {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  };
  return validROI(roi, width, height) ? roi : null;
}
export function moveROI(roi, dx, dy, width, height) {
  return {
    ...roi,
    x: Math.max(0, Math.min(width - roi.width, roi.x + Math.round(dx))),
    y: Math.max(0, Math.min(height - roi.height, roi.y + Math.round(dy))),
  };
}
export function resizeROI(roi, handle, dx, dy, width, height) {
  let left = roi.x,
    top = roi.y,
    right = roi.x + roi.width,
    bottom = roi.y + roi.height;
  if (handle.includes("w"))
    left = Math.max(0, Math.min(right - 1, left + Math.round(dx)));
  if (handle.includes("e"))
    right = Math.max(left + 1, Math.min(width, right + Math.round(dx)));
  if (handle.includes("n"))
    top = Math.max(0, Math.min(bottom - 1, top + Math.round(dy)));
  if (handle.includes("s"))
    bottom = Math.max(top + 1, Math.min(height, bottom + Math.round(dy)));
  return { x: left, y: top, width: right - left, height: bottom - top };
}
export function handles(roi) {
  const x = roi.x,
    y = roi.y,
    r = x + roi.width,
    b = y + roi.height,
    cx = (x + r) / 2,
    cy = (y + b) / 2;
  return {
    nw: { x, y },
    n: { x: cx, y },
    ne: { x: r, y },
    e: { x: r, y: cy },
    se: { x: r, y: b },
    s: { x: cx, y: b },
    sw: { x, y: b },
    w: { x, y: cy },
  };
}
export function hitROI(roi, point, tolerance) {
  if (!roi) return null;
  for (const [name, position] of Object.entries(handles(roi)))
    if (
      Math.abs(point.x - position.x) <= tolerance &&
      Math.abs(point.y - position.y) <= tolerance
    )
      return name;
  return point.x >= roi.x &&
    point.x <= roi.x + roi.width &&
    point.y >= roi.y &&
    point.y <= roi.y + roi.height
    ? "move"
    : null;
}
