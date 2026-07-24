"""
CLIP + FAISS fabric pattern matching GUI.
Network needed only for first model download (~300MB).
"""
import os, time, traceback, hashlib
import numpy as np
import faiss
import gradio as gr
from PIL import Image

# ============================================================
# CLIP Model (lazy load)
# ============================================================
_model = None
_preprocess = None
_device = None

def get_clip():
    global _model, _preprocess, _device
    if _model is None:
        import open_clip
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-B/32 on {_device}...")
        _model, _, _preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        _model = _model.to(_device).eval()
        print("CLIP loaded.")
    return _model, _preprocess

def extract_embedding(pil_img, model, preprocess):
    import torch
    img_tensor = preprocess(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype(np.float32)

# ============================================================
# Fabric Index
# ============================================================

class FabricIndex:
    def __init__(self):
        self.names = []
        self.images = {}
        self.embeddings = None
        self.index = None

    @property
    def cache_dir(self):
        return os.path.join(self._fabric_dir, "..", ".fabric_cache")

    def build(self, fabric_dir, force=False):
        self._fabric_dir = os.path.abspath(fabric_dir)

        # Try cache first
        cache_emb = os.path.join(self.cache_dir, "embeddings.npy")
        cache_names = os.path.join(self.cache_dir, "names.txt")
        cache_summary = os.path.join(self.cache_dir, "summary.txt")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Compute a hash of the fabric dir to detect changes
        files = sorted(os.listdir(fabric_dir))
        fnames_hash = hashlib.md5("".join(files).encode()).hexdigest()[:8]

        if not force and os.path.exists(cache_emb) and os.path.exists(cache_names):
            with open(cache_summary) as f:
                cached_hash = f.read().strip()
            if cached_hash == fnames_hash:
                print(f"Loading cached index ({len(files)} fabrics)...")
                self.embeddings = np.load(cache_emb)
                with open(cache_names) as f:
                    self.names = [ln.strip() for ln in f if ln.strip()]
                # Load PIL images for display
                for fname in self.names:
                    try:
                        self.images[fname] = Image.open(
                            os.path.join(fabric_dir, fname)).convert("RGB")
                    except Exception:
                        pass
                self._build_faiss()
                print(f"Cached index ready: {len(self.names)} fabrics.")
                return self

        # Build from scratch
        model, preprocess = get_clip()
        print(f"Extracting CLIP embeddings ({len(files)} fabrics)...")

        vectors = []
        for i, f in enumerate(files):
            path = os.path.join(fabric_dir, f)
            try:
                pil = Image.open(path).convert("RGB")
                emb = extract_embedding(pil, model, preprocess)
                vectors.append(emb)
                self.names.append(f)
                self.images[f] = pil
            except Exception:
                continue
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(files)}")

        self.embeddings = np.array(vectors, dtype=np.float32)

        # Save cache
        np.save(cache_emb, self.embeddings)
        with open(cache_names, "w") as f:
            for name in self.names:
                f.write(name + "\n")
        with open(cache_summary, "w") as f:
            f.write(fnames_hash)

        self._build_faiss()
        print(f"Index ready: {len(self.names)} fabrics (cached).")
        return self

    def _build_faiss(self):
        n, d = self.embeddings.shape
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.embeddings)

    def search(self, pil_img, k=20):
        model, preprocess = get_clip()
        q = extract_embedding(pil_img, model, preprocess)
        sims, idxs = self.index.search(q.reshape(1, -1), k)
        results = []
        for sim, i in zip(sims[0], idxs[0]):
            if 0 <= i < len(self.names):
                results.append((self.names[i], float(sim), self.images[self.names[i]]))
        return results

# ============================================================
# Gradio GUI
# ============================================================

def build_ui(index: FabricIndex):
    css = """
    .progress-wrap { margin: 12px 0; }
    .progress-bar { height: 6px; background: #e0e0e0; border-radius: 3px; overflow: hidden; }
    .progress-fill { height: 100%; background: #111; border-radius: 3px; transition: width 0.3s; }
    .result-summary { font-size: 13px; color: #666; padding: 8px 0; border-bottom: 1px solid #eee; margin-bottom: 12px; }
    footer { display: none !important; }
    """

    with gr.Blocks(css=css, theme=gr.themes.Monochrome(), title="Fabric Pattern Matcher") as app:
        gr.Markdown("## 面料花型匹配")

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                input_img = gr.Image(type="pil", label="拖拽或点击上传照片", height=320)
            with gr.Column(scale=3):
                with gr.Row():
                    top_k = gr.Slider(3, 50, value=15, step=1, label="返回数量",
                                      info="控制显示多少条匹配结果")
                with gr.Row():
                    btn = gr.Button("开始检索", variant="primary", size="lg")
                    clear_btn = gr.Button("清空", size="lg")

                progress = gr.HTML(
                    '<div style="color:#999;padding:20px 0">'
                    f'花型库已就绪 · {len(index.names)} 张 · CLIP ViT-B/32'
                    '</div>'
                )

        gr.Markdown("---")
        gallery = gr.Gallery(label="匹配结果", columns=5, rows=3, height=520, object_fit="contain",
                             show_label=False)
        detail = gr.Textbox(label="", lines=10, max_lines=20, visible=True, show_label=False,
                            placeholder="检索结果将显示在这里...")

        def _confidence_label(sim):
            if sim > 0.85: return "极高"
            if sim > 0.75: return "高"
            if sim > 0.65: return "中"
            if sim > 0.50: return "较低"
            return "低"

        def _conf_color(sim):
            if sim > 0.85: return "#2e7d32"
            if sim > 0.75: return "#4caf50"
            if sim > 0.65: return "#ff9800"
            if sim > 0.50: return "#f57c00"
            return "#d32f2f"

        def _conf_bar(sim):
            pct = min(int(sim * 100), 100)
            color = _conf_color(sim)
            return (f'<div style="display:flex;align-items:center;gap:8px">'
                    f'<span style="font-size:12px;color:{color};font-weight:600;min-width:32px">{pct}%</span>'
                    f'<div style="flex:1;height:4px;background:#e0e0e0;border-radius:2px">'
                    f'<div style="width:{pct}%;height:100%;background:{color};border-radius:2px"></div>'
                    f'</div></div>')

        def on_search(img, k):
            if img is None:
                return [], "", progress_html("请先上传照片", "#999")

            yield [], "", progress_html("正在提取特征...", "#999")

            t0 = time.time()
            try:
                results = index.search(img, int(k))
            except Exception as e:
                traceback.print_exc()
                return [], "", progress_html(f"错误: {e}", "#d32f2f")

            yield [], "", progress_html("正在排序结果...", "#999")

            elapsed = time.time() - t0

            gallery_items = []
            top_sim = results[0][1] if results else 0
            if top_sim > 0.85:
                summary_color = "#2e7d32"
                summary_text = "高置信度匹配"
            elif top_sim > 0.65:
                summary_color = "#ff9800"
                summary_text = "中等置信度"
            else:
                summary_color = "#d32f2f"
                summary_text = "置信度较低，请确认"

            detail_lines = [
                f'<div class="result-summary">'
                f'<b style="color:{summary_color}">{summary_text}</b> · '
                f'Top-1: <b>{results[0][0] if results else "-"}</b> · '
                f'耗时 {elapsed*1000:.0f}ms'
                f'</div>'
            ]

            for i, (name, sim, pil) in enumerate(results):
                label = _confidence_label(sim)
                bar = _conf_bar(sim)
                gallery_items.append((pil, f"#{i+1} {label}"))
                detail_lines.append(
                    f'<div style="padding:4px 0">'
                    f'<span style="font-weight:600">#{i+1}</span> '
                    f'<span style="font-family:monospace;font-size:13px">{name}</span> '
                    f'{bar}</div>'
                )

            yield gallery_items, "\n".join(detail_lines), progress_html(
                f"检索完成 · {elapsed*1000:.0f}ms · Top-1: {results[0][0] if results else '-'}",
                summary_color
            )

        def progress_html(msg, color):
            return f'<div style="color:{color};padding:20px 0;font-size:14px">{msg}</div>'

        def on_clear():
            return None, "", progress_html(
                f'花型库已就绪 · {len(index.names)} 张 · CLIP ViT-B/32', "#999")

        btn.click(on_search, inputs=[input_img, top_k], outputs=[gallery, detail, progress])
        clear_btn.click(on_clear, outputs=[input_img, detail, progress])

    return app


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fabric-dir", default="dir/fabric")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Fabric Pattern Matcher GUI")
    print("=" * 60)

    index = FabricIndex()
    index.build(args.fabric_dir)
    app = build_ui(index)
    app.launch(server_port=args.port, share=args.share, inbrowser=True)
