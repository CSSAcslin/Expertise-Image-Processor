import { svgIcon } from "./icons.js";

// Keep each control beside the geometry operation it configures. Parameters
// still use the module editor's normal binding, history and persistence.
export function alignmentParameters(fields, node, renderField, coordinatesOpen = false) {
  const field = (key, label) => {
    const parameter = fields.find(p => p.key === key);
    return renderField({ ...parameter, label: label || parameter.label }, node);
  };
  const midline = fields.find(p => p.key === "midline");
  const endpoint = (index, label) => renderField({
    ...midline, key: `midline.${index}`, kind: "float", label,
    default: midline.default[index], minimum: 0,
  }, node);
  const region = (side, label) => `<div class="alignment-coordinate-group" role="group" aria-label="${label}精确坐标">
    <h4>${label}</h4><div class="flow-fields">
      ${field(`${side}.x`, "X / px")}${field(`${side}.y`, "Y / px")}
      ${field(`${side}.width`, "宽 / px")}${field(`${side}.height`, "高 / px")}
    </div></div>`;
  const canvasButton = (kind, label, icon, title) =>
    `<button type="button" class="quiet" data-canvas-tool="${kind}" title="${title}">${svgIcon(icon)}${label}</button>`;
  const action = (kind, label, icon, title) =>
    `<button type="button" class="quiet" data-geometry="${kind}" title="${title}" ${kind === "apply" ? "disabled" : ""}>${svgIcon(icon)}${label}</button>`;

  return `<div class="alignment-settings" aria-label="双幅配准参数">
    <section class="alignment-section" aria-label="分幅范围">
      <h3>分幅范围</h3>
      <div class="alignment-button-grid">
        ${canvasButton("alignment_midline", "调整中线", "line", "在数据视窗拖动中线，划分左右幅")}
        ${action("suggest", "建议分幅", "channel", "根据当前中线和当前帧建议左右有效区")}
        ${canvasButton("alignment_left", "左幅有效区", "crop", "在数据视窗绘制或修改左幅有效区")}
        ${canvasButton("alignment_right", "右幅有效区", "crop", "在数据视窗绘制或修改右幅有效区")}
      </div>
      <details class="alignment-coordinates" data-alignment-coordinates ${coordinatesOpen ? "open" : ""}>
        <summary>精确坐标</summary>
        <div class="alignment-coordinate-group" role="group" aria-label="中线坐标"><h4>中线</h4><div class="flow-fields">
          ${endpoint(0, "顶部 X / px")}${endpoint(1, "底部 X / px")}
        </div></div>
        ${region("left", "左幅")}${region("right", "右幅")}
      </details>
    </section>
    <section class="alignment-section" aria-label="固定配准">
      <h3>固定配准</h3>
      <div class="flow-fields alignment-offsets">
        ${field("dx", "水平 Δx / px")}${field("dy", "垂直 Δy / px")}
      </div>
      <div class="flow-fields alignment-orientation">
        ${field("mirror_x", "水平镜像")}${field("mirror_y", "垂直镜像")}
      </div>
      <div class="flow-fields alignment-sampling">${field("interpolation", "插值")}</div>
      <div class="alignment-button-grid">
        ${action("calibrate", "自动平移", "align", "基于当前帧估计右路平移，生成建议后手动应用")}
        ${action("apply", "应用建议", "check", "将自动平移建议写入当前模块，再查看叠加")}
      </div>
      <output class="alignment-proposal" data-alignment-proposal aria-live="polite" hidden></output>
    </section>
    <section class="alignment-section" aria-label="检查配准">
      <h3>检查配准</h3>
      ${action("overlay", "检查叠加", "layers", "使用当前配准参数检查当前帧的左右叠加")}
    </section>
  </div>`;
}
