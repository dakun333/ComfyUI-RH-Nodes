import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

function viewUrl(item, filename) {
    // ComfyUI /view deliberately strips directories from filename. Keep them in subfolder.
    const parts = filename.replaceAll("\\", "/").split("/");
    const basename = parts.pop();
    const subfolder = [item.subfolder || "", ...parts].filter(Boolean).join("/");
    const params = new URLSearchParams({ filename: basename, subfolder, type: "output" });
    return api.apiURL(`/view?${params}`);
}

function showVideo(node, item) {
    if (!item?.filename) return;
    if (!node._vrmbg3Preview) {
        const host = document.createElement("div");
        host.className = "vrmbg3-video-preview";
        Object.assign(host.style, { display: "flex", flexDirection: "column", width: "100%", height: "100%", gap: "6px", background: "#202124", color: "#eee", pointerEvents: "auto", borderRadius: "6px", overflow: "hidden" });
        const label = document.createElement("div");
        label.textContent = "棋盘预览 · 保存的 MOV 保留真实 Alpha";
        Object.assign(label.style, {font: "12px sans-serif", padding: "6px"});
        const video = document.createElement("video");
        Object.assign(video, {controls: true, loop: true, muted: true, playsInline: true, preload: "metadata"});
        video.setAttribute("aria-label", "VRMBG3 棋盘视频预览");
        Object.assign(video.style, {width: "100%", minHeight: "0", flex: "1", objectFit: "contain", background: "#111"});
        const download = document.createElement("a");
        download.textContent = "下载透明 MOV";
        download.setAttribute("download", "transparent.mov");
        Object.assign(download.style, {color: "#9ccdff", padding: "6px", font: "12px sans-serif"});
        host.append(label, video, download);
        const widget = node.addDOMWidget("vrmbg3_video_preview", "vrmbg3_video", host, {serialize: false, hideOnZoom: false});
        widget.computeSize = (width) => [width, Math.min(520, Math.max(280, (width-20) * (item.height / item.width) + 62))];
        widget.serializeValue = () => undefined;
        widget.onRemove = () => {video.pause(); video.removeAttribute("src"); video.load(); host.remove();};
        node._vrmbg3Preview = {host, video, download, widget};
    }
    const { video, download, widget } = node._vrmbg3Preview;
    const url = viewUrl(item, item.filename);
    if (video.getAttribute("src") !== url) {video.src = url; video.load();}
    download.href = viewUrl(item, item.mov_filename);
    widget.computeSize = (width) => [width, Math.min(520, Math.max(280, (width-20) * (item.height / item.width) + 62))];
    requestAnimationFrame(() => {
        const size = node.computeSize();
        node.setSize([Math.max(350, node.size[0], size[0]), Math.max(size[1], 460)]);
        node.graph?.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "VRMBG3.V6.MovPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VRMBG3V6SaveVideo") return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            const result = onExecuted?.apply(this, arguments);
            const item = message?.vrmbg3_preview?.[0];
            if (item) { this.properties.vrmbg3_last_preview = item; showVideo(this, item); }
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = onConfigure?.apply(this, arguments);
            if (this.properties?.vrmbg3_last_preview) requestAnimationFrame(() => showVideo(this, this.properties.vrmbg3_last_preview));
            return result;
        };
        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function() {
            this._vrmbg3Preview?.widget.onRemove?.();
            return onRemoved?.apply(this, arguments);
        };
    }
});
