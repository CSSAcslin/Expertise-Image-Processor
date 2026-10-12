import { createROI, moveROI, resizeROI, hitROI, handles } from "./geometry.js";

export class Viewer {
  constructor(canvas, scene, callbacks) {
    this.canvas = canvas;
    this.scene = scene;
    this.callbacks = callbacks;
    this.snapshot = null;
    this.image = null;
    new ResizeObserver(() => callbacks.redraw()).observe(canvas);
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const p = this.point(e),
          old = this.scale();
        const native = this.native(p);
        scene.zoom = Math.max(
          0.08,
          Math.min(30, scene.zoom * Math.exp(-e.deltaY * 0.0015)),
        );
        const next = this.scale();
        scene.panX += (native.x - scene.width / 2) * (old - next);
        scene.panY += (native.y - scene.height / 2) * (old - next);
        callbacks.redraw(this);
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.snapshot || (e.button !== 0 && e.button !== 1)) return;
      if(scene.tool==="point"&&e.button===0){const p=this.native(this.point(e));if(p.x>=0&&p.y>=0&&p.x<scene.width&&p.y<scene.height){scene.annotation={kind:"point",x:Math.floor(p.x),y:Math.floor(p.y)};callbacks.geometry?.(scene.annotation);callbacks.redraw(this);}return;}
      canvas.setPointerCapture(e.pointerId);
      const point = this.point(e);
      this.drag = {
        point,
        native: this.native(point),
        x: scene.panX,
        y: scene.panY,
        initialROI: structuredClone(scene.roi),
        action:
          scene.tool === "line" && !e.altKey && e.button === 0 ? "line" :
          scene.tool === "roi" && !e.altKey && e.button === 0
            ? hitROI(scene.roi, this.native(point), 7 / this.scale()) ||
              "create"
            : "pan",
      };
      if(this.drag.action==="line"){const p=this.drag.native;scene.annotation={kind:"line",x0:p.x,y0:p.y,x1:p.x,y1:p.y};}
      if (this.drag.action === "create") {
        const n = this.drag.native;
        if (n.x < 0 || n.y < 0 || n.x > scene.width || n.y > scene.height) {
          this.drag = null;
          return;
        }
        scene.roi = null;
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      const p = this.point(e),
        n = this.native(p);
      if (this.drag) {
        const drag = this.drag;
        if(drag.action==="line"){scene.annotation={kind:"line",x0:drag.native.x,y0:drag.native.y,x1:Math.max(0,Math.min(scene.width-1,n.x)),y1:Math.max(0,Math.min(scene.height-1,n.y))};}
        else if (drag.action === "create")
          scene.roi = createROI(drag.native, n, scene.width, scene.height);
        else if (drag.action === "move")
          scene.roi = moveROI(
            drag.initialROI,
            n.x - drag.native.x,
            n.y - drag.native.y,
            scene.width,
            scene.height,
          );
        else if (drag.action !== "pan")
          scene.roi = resizeROI(
            drag.initialROI,
            drag.action,
            n.x - drag.native.x,
            n.y - drag.native.y,
            scene.width,
            scene.height,
          );
        else {
          scene.panX = this.drag.x + p.x - this.drag.point.x;
          scene.panY = this.drag.y + p.y - this.drag.point.y;
        }
        if (drag.action !== "pan") callbacks.roiPreview?.();
        callbacks.redraw(this);
      } else if (this.snapshot) {
        const hit =
          scene.tool === "roi" ? hitROI(scene.roi, n, 7 / this.scale()) : null;
        canvas.style.cursor =
          scene.tool === "roi"
            ? {
                n: "ns-resize",
                s: "ns-resize",
                e: "ew-resize",
                w: "ew-resize",
                nw: "nwse-resize",
                se: "nwse-resize",
                ne: "nesw-resize",
                sw: "nesw-resize",
                move: "move",
              }[hit] || "crosshair"
            : "grab";
        callbacks.pixel(this.snapshot, Math.floor(n.x), Math.floor(n.y));
      }
    });
    canvas.addEventListener("pointerup", () => {
      if (this.drag && this.drag.action !== "pan") {
        if (this.drag.action === "create" && !scene.roi)
          scene.roi = this.drag.initialROI;
        if(this.drag.action==="line")callbacks.geometry?.(scene.annotation);else callbacks.roi(scene.roi);
      }
      this.drag = null;
    });
    canvas.addEventListener("pointercancel", () => {
      if (this.drag) {
        scene.roi = this.drag.initialROI;
        callbacks.roiPreview?.();
        callbacks.redraw(this);
      }
      this.drag = null;
    });
    canvas.addEventListener("dblclick", () => {
      if (scene.tool === "pan") callbacks.fit();
    });
  }
  point(event) {
    const r = this.canvas.getBoundingClientRect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  }
  scale() {
    const r = this.canvas.getBoundingClientRect();
    return Math.max(
      0.001,
      Math.min(
        (r.width - 40) / this.scene.width,
        (r.height - 50) / this.scene.height,
      ) * this.scene.zoom,
    );
  }
  native(p) {
    const r = this.canvas.getBoundingClientRect(),
      s = this.scale();
    return {
      x: (p.x - r.width / 2 - this.scene.panX) / s + this.scene.width / 2,
      y: (p.y - r.height / 2 - this.scene.panY) / s + this.scene.height / 2,
    };
  }
  async set(snapshot) {
    this.snapshot = snapshot;
    const img = new Image();
    img.src = snapshot.image;
    await img.decode();
    if (this.snapshot === snapshot) {
      this.image = img;
      this.draw();
    }
  }
  clear() {
    this.image = this.snapshot = null;
    this.draw();
  }
  draw() {
    const c = this.canvas,
      r = c.getBoundingClientRect(),
      dpr = window.devicePixelRatio || 1;
    c.width = Math.round(r.width * dpr);
    c.height = Math.round(r.height * dpr);
    const ctx = c.getContext("2d");
    ctx.scale(dpr, dpr);
    if (!this.image || !this.snapshot || !r.width || !r.height) return;
    const s = this.scale(),
      scene = this.scene,
      snap = this.snapshot;
    ctx.translate(r.width / 2 + scene.panX, r.height / 2 + scene.panY);
    ctx.scale(s, s);
    ctx.translate(-scene.width / 2, -scene.height / 2);
    ctx.imageSmoothingEnabled = s < 1;
    ctx.drawImage(
      this.image,
      snap.origin[0],
      snap.origin[1],
      snap.shape[1],
      snap.shape[0],
    );
    if(scene.annotation){const a=scene.annotation;ctx.strokeStyle="#e2c58b";ctx.fillStyle="#e2c58b";ctx.lineWidth=1.5/s;ctx.beginPath();if(a.kind==="point"){ctx.moveTo(a.x-8/s,a.y);ctx.lineTo(a.x+8/s,a.y);ctx.moveTo(a.x,a.y-8/s);ctx.lineTo(a.x,a.y+8/s);ctx.stroke();ctx.beginPath();ctx.arc(a.x,a.y,3/s,0,Math.PI*2);ctx.stroke();}else{ctx.moveTo(a.x0,a.y0);ctx.lineTo(a.x1,a.y1);ctx.stroke();for(const p of[{x:a.x0,y:a.y0},{x:a.x1,y:a.y1}])ctx.fillRect(p.x-3/s,p.y-3/s,6/s,6/s);}}
    if (scene.roi) {
      const q = scene.roi;
      ctx.strokeStyle = "#8ae6ce";
      ctx.lineWidth = 1.4 / s;
      ctx.fillStyle = "#82e1c518";
      ctx.fillRect(q.x, q.y, q.width, q.height);
      ctx.strokeRect(q.x, q.y, q.width, q.height);
      if (scene.tool === "roi") {
        ctx.fillStyle = "#91ead1";
        ctx.strokeStyle = "#163f32";
        ctx.lineWidth = 1 / s;
        for (const p of Object.values(handles(q))) {
          ctx.fillRect(p.x - 3 / s, p.y - 3 / s, 6 / s, 6 / s);
          ctx.strokeRect(p.x - 3 / s, p.y - 3 / s, 6 / s, 6 / s);
        }
      }
    }
  }
}
