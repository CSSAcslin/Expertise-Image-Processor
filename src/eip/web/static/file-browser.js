import { api } from "./api.js";
const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );

export class FileBrowser {
  constructor({ icon, submit, report }) {
    this.icon = icon;
    this.submit = submit;
    this.report = report;
    this.entries = [];
    this.selected = [];
    this.generation = 0;
    this.mode = "data";
    this.importMode = "files";
    $("browse-go").onclick = () => this.act(() => this.navigate());
    $("browse-path").onkeydown = (e) => {
      if (e.key === "Enter") this.act(() => this.navigate());
    };
    $("file-filter").oninput = () => this.render();
    $("file-select-all").onchange = () => {
      this.selected = $("file-select-all").checked
        ? this.visible()
            .filter((e) => !e.directory)
            .map((e) => e.path)
        : [];
      this.render();
    };
    $("clear-files").onclick = () => {
      this.selected = [];
      this.render();
    };
    for (const mode of ["files", "sequence"])
      $("import-mode-" + mode).onclick = () => {
        this.importMode = mode;
        this.selected = [];
        this.render();
      };
  }
  async act(fn) {
    try {
      await fn();
    } catch (error) {
      this.report(error.message, true);
    }
  }
  visible() {
    const filter = $("file-filter").value.trim().toLocaleLowerCase();
    return this.entries.filter((e) =>
      e.name.toLocaleLowerCase().includes(filter),
    );
  }
  async open(mode, initial, fallback) {
    this.mode = mode;
    this.importMode = "files";
    this.selected = [];
    $("file-filter").value = "";
    $("interval").value = "";
    $("import-report").hidden = true;
    const titles = {
      data: "导入本地数据",
      plugin: "接入 Python 算法",
      load: "打开项目",
      result: "打开历史结果",
      "result-directory": "打开历史导出文件夹",
      save: "保存项目",
      output: "选择输出目录",
    };
    $("dialog-title").textContent = titles[mode];
    $("filename-label").hidden = mode !== "save";
    $("import-modes").hidden = mode !== "data";
    $("dialog-note").textContent =
      mode === "data"
        ? "目录单击进入；文件支持 Ctrl / Shift 多选。原始文件保持在本机。"
        : mode === "plugin"
          ? "仅选择可信的算法描述 JSON；注册只读取描述，运行时才执行代码。"
          : mode === "result-directory" ? "进入历史导出文件夹后，点击使用当前导出目录。只找回引用，不复制文件。" : mode === "result" ? "选择结果目录中的 run.json。" : "项目记录引用路径及参数，不复制原始数据。";
    try {
      await this.browse(initial);
    } catch {
      await this.browse(fallback);
    }
    $("file-dialog").showModal();
  }
  async browse(path) {
    const generation = ++this.generation;
    $("file-list").setAttribute("aria-busy", "true");
    try {
      const data = await api(
        `browse?path=${encodeURIComponent(path)}&mode=${["output","result-directory"].includes(this.mode) ? "directory" : this.mode === "data" ? "data" : "json"}`,
      );
      if (generation !== this.generation) return;
      this.path = data.path;
      this.parent = data.parent;
      this.entries = data.entries;
      this.selected = [];
      this.anchor = null;
      $("browse-path").value = data.path;
      $("file-filter").value = "";
      $("drives").innerHTML = data.drives
        .map(
          (d) =>
            `<button class="quiet" data-drive="${esc(d)}">${esc(d)}</button>`,
        )
        .join("");
      $("drives")
        .querySelectorAll("button")
        .forEach(
          (b) =>
            (b.onclick = () => this.act(() => this.browse(b.dataset.drive))),
        );
      $("browse-up").onclick = () => this.act(() => this.browse(data.parent));
      this.render();
      if (data.truncated)
        this.report(
          "目录项超过 1000，当前显示前 1000 项；可输入完整文件路径。",
        );
    } finally {
      if (generation === this.generation)
        $("file-list").setAttribute("aria-busy", "false");
    }
  }
  async navigate() {
    const info = await api(
      `path?path=${encodeURIComponent($("browse-path").value.trim())}`,
    );
    if (info.directory) {
      await this.browse(info.path);
      return;
    }
    if (["output","save","result-directory"].includes(this.mode))
      throw Error("此操作需要目录路径");
    if (this.mode === "data" && !info.supported_data)
      throw Error("当前仅支持图像、视频和图像序列");
    if (this.mode !== "data" && info.suffix !== ".json")
      throw Error("请选择 JSON 文件");
    if (this.mode === "data") this.importMode = "files";
    this.selected = [info.path];
    $("browse-path").value = info.path;
    this.render();
  }
  render(rows = true) {
    const multiple = this.mode === "data" && this.importMode === "files",
      files = this.visible().filter((e) => !e.directory);
    $("file-select-all").checked =
      files.length > 0 && files.every((e) => this.selected.includes(e.path));
    $("file-select-all").indeterminate =
      files.some((e) => this.selected.includes(e.path)) &&
      !$("file-select-all").checked;
    $("selection-tools").hidden = !multiple;
    $("interval-label").hidden =
      this.mode !== "data" || this.importMode !== "sequence";
    $("import-mode-files").classList.toggle(
      "selected",
      this.importMode === "files",
    );
    $("import-mode-sequence").classList.toggle(
      "selected",
      this.importMode === "sequence",
    );
    const text = {
      plugin: "注册算法",
      load: "打开项目",
      save: "保存项目",
      output: "使用当前目录",
      result: "打开运行记录",
      "result-directory": "使用当前导出目录",
    };
    $("dialog-submit").textContent =
      this.mode === "data"
        ? this.importMode === "files"
          ? `导入选中文件${this.selected.length ? "（" + this.selected.length + "）" : ""}`
          : "导入当前目录序列"
        : text[this.mode];
    $("dialog-submit").disabled =
      ["data", "plugin", "load", "result"].includes(this.mode) &&
      !(this.mode === "data" && this.importMode === "sequence") &&
      !this.selected.length;
    $("selection-info").textContent = this.selected.length
      ? `已选择 ${this.selected.length} 个文件：${this.selected[0]}`
      : "当前目录：" + (this.path || "");
    $("file-count").textContent = `${this.visible().length} 项`;
    if (!rows) {
      $("file-list")
        .querySelectorAll(".file-row")
        .forEach((el) => {
          const selected = this.selected.includes(el.dataset.path);
          el.classList.toggle("selected", selected);
          el.setAttribute("aria-selected", String(selected));
        });
      return;
    }
    $("file-list").innerHTML =
      this.visible()
        .map(
          (e) =>
            `<div class="file-row ${this.selected.includes(e.path) ? "selected" : ""}" role="option" aria-selected="${this.selected.includes(e.path)}" tabindex="0" data-path="${esc(e.path)}" data-directory="${e.directory}">${this.icon(e.directory ? "folder" : e.name.match(/\.(mp4|avi|mov|mkv|webm|m4v|mpg|mpeg|cine)$/i) ? "video" : "file")}<span>${esc(e.name)}</span><small>${e.directory ? "目录" : (e.size / 1024 / 1024).toFixed(2) + " MB"}</small></div>`,
        )
        .join("") ||
      '<div class="library-empty">当前目录或筛选条件下没有可用文件</div>';
    $("file-list")
      .querySelectorAll(".file-row")
      .forEach((el) => {
        el.onclick = (event) => {
          if (el.dataset.directory === "true") {
            this.act(() => this.browse(el.dataset.path));
            return;
          }
          if (this.mode === "data" && this.importMode === "sequence") return;
          const path = el.dataset.path,
            visible = files.map((f) => f.path);
          if (
            multiple &&
            event.shiftKey &&
            this.anchor &&
            visible.includes(this.anchor)
          ) {
            const a = visible.indexOf(this.anchor),
              b = visible.indexOf(path);
            this.selected = [
              ...new Set([
                ...this.selected,
                ...visible.slice(Math.min(a, b), Math.max(a, b) + 1),
              ]),
            ];
          } else if (multiple && (event.ctrlKey || event.metaKey))
            this.selected = this.selected.includes(path)
              ? this.selected.filter((p) => p !== path)
              : [...this.selected, path];
          else this.selected = [path];
          this.anchor = path;
          this.render(false);
        };
        el.ondblclick = () => {
          if (
            el.dataset.directory !== "true" &&
            !(this.mode === "data" && this.importMode === "sequence")
          )
            this.act(this.submit);
        };
        el.onkeydown = (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            el.click();
          }
        };
      });
  }
  requests() {
    if (this.importMode === "sequence") {
      if (!$("interval").reportValidity())
        throw Error("序列间隔必须是正数，未知时请留空");
      return [
        {
          path: this.path,
          kind: "sequence",
          interval: $("interval").value ? Number($("interval").value) : null,
        },
      ];
    }
    if (!this.selected.length) throw Error("请选择要导入的文件");
    return this.selected.map((path) => ({
      path,
      kind: "auto",
      interval: null,
    }));
  }
  showReport(results) {
    $("import-report").hidden = false;
    $("import-report").textContent = results
      .map(
        (r) => `${r.ok ? "✓" : "×"} ${r.path}${r.error ? "：" + r.error : ""}`,
      )
      .join("\n");
  }
}
