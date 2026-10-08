export class Viewer {
  constructor(canvas, scene, callbacks) {
    this.canvas = canvas;
    this.scene = scene;
    this.callbacks = callbacks;
    this.snapshot = null;
    this.image = null;
    new ResizeObserver(() => this.draw()).observe(canvas);
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
        callbacks.redraw();
      },
      { passive: false },
    );
    canvas.addEventListener("pointerdown", (e) => {
      if (!this.snapshot) return;
      canvas.setPointerCapture(e.pointerId);
      const point = this.point(e);
      this.drag = {
        point,
        native: this.native(point),
        x: scene.panX,
        y: scene.panY,
      };
      if (scene.tool === "roi") {
        this.draft = this.drag.native;
        scene.roi = null;
      }
    });
    canvas.addEventListener("pointermove", (e) => {
      const p = this.point(e),
        n = this.native(p);
      if (this.drag) {
        if (scene.tool === "roi") {
          const x1 = Math.max(
              0,
              Math.min(scene.width - 1, Math.floor(this.drag.native.x)),
            ),
            y1 = Math.max(
              0,
              Math.min(scene.height - 1, Math.floor(this.drag.native.y)),
            );
          const x2 = Math.max(0, Math.min(scene.width, Math.round(n.x))),
            y2 = Math.max(0, Math.min(scene.height, Math.round(n.y)));
          scene.roi = {
            x: Math.min(x1, x2),
            y: Math.min(y1, y2),
            width: Math.abs(x2 - x1),
            height: Math.abs(y2 - y1),
          };
        } else {
          scene.panX = this.drag.x + p.x - this.drag.point.x;
          scene.panY = this.drag.y + p.y - this.drag.point.y;
        }
        callbacks.redraw();
      } else if (this.snapshot)
        callbacks.pixel(this.snapshot, Math.floor(n.x), Math.floor(n.y));
    });
    canvas.addEventListener("pointerup", () => {
      if (this.drag && scene.tool === "roi") {
        if (!scene.roi?.width || !scene.roi?.height) scene.roi = null;
        callbacks.roi(scene.roi);
      }
      this.drag = null;
    });
    canvas.addEventListener("pointercancel", () => {
      this.drag = null;
    });
    canvas.addEventListener("dblclick", () => callbacks.fit());
  }
  point(event) {
    const r = this.canvas.getBoundingClientRect();
    return { x: event.clientX - r.left, y: event.clientY - r.top };
  }
  scale() {
    const r = this.canvas.getBoundingClientRect();
    return (
      Math.min(
        (r.width - 40) / this.scene.width,
        (r.height - 50) / this.scene.height,
      ) * this.scene.zoom
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
    if (scene.roi) {
      const q = scene.roi;
      ctx.strokeStyle = "#8ae6ce";
      ctx.lineWidth = 1.4 / s;
      ctx.fillStyle = "#82e1c518";
      ctx.fillRect(q.x, q.y, q.width, q.height);
      ctx.strokeRect(q.x, q.y, q.width, q.height);
    }
  }
}
