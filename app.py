"""
Fabric Pattern Matcher - CLIP + FAISS + Gradio GUI.
"""
import os, time, traceback, hashlib, html
import numpy as np
import faiss
import gradio as gr
from PIL import Image

# ============================================================
# CLIP
# ============================================================
_model = None
_preprocess = None
_device = None

def get_clip():
    global _model, _preprocess, _device
    if _model is None:
        import open_clip; import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading CLIP ViT-B/32 on {_device}...")
        _model, _, _preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        _model = _model.to(_device).eval()
        print("CLIP loaded.")
    return _model, _preprocess

def extract_embedding(pil_img, model, preprocess):
    import torch
    t = preprocess(pil_img).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = model.encode_image(t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype(np.float32)

# ============================================================
# FAISS Index
# ============================================================
class FabricIndex:
    def __init__(self):
        self.names = []; self.images = {}; self.embeddings = None; self.index = None

    @property
    def cache_dir(self):
        return os.path.join(self._fabric_dir, "..", ".fabric_cache")

    def build(self, fabric_dir, force=False):
        self._fabric_dir = os.path.abspath(fabric_dir)
        cache_emb = os.path.join(self.cache_dir, "embeddings.npy")
        cache_names = os.path.join(self.cache_dir, "names.txt")
        cache_hash = os.path.join(self.cache_dir, "hash.txt")
        os.makedirs(self.cache_dir, exist_ok=True)

        files = sorted(os.listdir(fabric_dir))
        h = hashlib.md5("".join(files).encode()).hexdigest()[:8]

        if not force and os.path.exists(cache_emb) and os.path.exists(cache_hash):
            with open(cache_hash) as f:
                if f.read().strip() == h:
                    print(f"Loading cache ({len(files)} fabrics)...")
                    self.embeddings = np.load(cache_emb)
                    self.names = [ln.strip() for ln in open(cache_names) if ln.strip()]
                    for n in self.names:
                        try: self.images[n] = Image.open(os.path.join(fabric_dir, n)).convert("RGB")
                        except: pass
                    self._build_faiss()
                    print(f"Ready: {len(self.names)} fabrics.")
                    return self

        model, preprocess = get_clip()
        print(f"Extracting embeddings ({len(files)} fabrics)...")
        vectors = []
        for i, f in enumerate(files):
            try:
                pil = Image.open(os.path.join(fabric_dir, f)).convert("RGB")
                vectors.append(extract_embedding(pil, model, preprocess))
                self.names.append(f); self.images[f] = pil
            except: pass
            if (i+1) % 100 == 0: print(f"  {i+1}/{len(files)}")

        self.embeddings = np.array(vectors, dtype=np.float32)
        np.save(cache_emb, self.embeddings)
        open(cache_names,"w").write("\n".join(self.names))
        open(cache_hash,"w").write(h)
        self._build_faiss()
        print(f"Ready: {len(self.names)} fabrics (cached).")
        return self

    def _build_faiss(self):
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, pil_img, k=20):
        model, preprocess = get_clip()
        q = extract_embedding(pil_img, model, preprocess)
        sims, idxs = self.index.search(q.reshape(1,-1), k)
        return [(self.names[i], float(s), self.images[self.names[i]])
                for s, i in zip(sims[0], idxs[0]) if 0 <= i < len(self.names)]

# ============================================================
# GUI
# ============================================================
def build_ui(index):
    css = """
    footer { display: none !important; }
    .result-summary { font-size: 13px; color: #666; padding: 8px 0; border-bottom: 1px solid #eee; margin-bottom: 12px; }
    """
    with gr.Blocks(css=css, theme=gr.themes.Monochrome(), title="Fabric Pattern Matcher") as app:
        gr.Markdown("## Fabric Pattern Matcher")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.ImageEditor(
                    type="pil", label="Upload photo, then crop fabric area",
                    canvas_size=(800, 800), transforms=["crop"], brush=False, layers=False,
                )
            with gr.Column(scale=1):
                gr.Markdown(f"**Library**: {len(index.names)} fabrics | **Model**: CLIP ViT-B/32\n\n"
                            "1. Upload photo on the left\n"
                            "2. Click scissors icon (top-right)\n"
                            "3. Drag handles to crop fabric area\n"
                            "4. Click Search")
                top_k = gr.Slider(3, 50, value=15, step=1, label="Results count")
                with gr.Row():
                    btn = gr.Button("Search", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="lg")
                progress = gr.HTML('<div style="color:#999;padding:10px 0;font-size:13px">Ready</div>')

        gr.Markdown("---")
        gallery = gr.Gallery(label="Results", columns=5, rows="auto", height=640,
                             object_fit="contain", show_label=False)
        detail = gr.HTML()

        def conf_label(s):
            if s > 0.85: return "Very High"
            if s > 0.75: return "High"
            if s > 0.65: return "Medium"
            if s > 0.50: return "Low"
            return "Very Low"

        def conf_color(s):
            if s > 0.85: return "#2e7d32"
            if s > 0.75: return "#4caf50"
            if s > 0.65: return "#ff9800"
            if s > 0.50: return "#f57c00"
            return "#d32f2f"

        def conf_bar(s):
            pct = min(int(s*100), 100); c = conf_color(s)
            return (f'<span style="color:{c};font-weight:600">{pct}%</span> '
                    f'<span style="display:inline-block;width:{pct}px;height:4px;background:{c};border-radius:2px"></span>')

        def on_search(img, k):
            if img is None: return [], "", status_html("Upload a photo first", "#999")
            if isinstance(img, dict):
                img = img.get("composite") or img.get("background") or img
            if img is None: return [], "", status_html("Upload a photo first", "#999")

            yield [], "", status_html("Extracting features...", "#999")
            t0 = time.time()
            try:
                results = index.search(img, int(k))
            except Exception as e:
                traceback.print_exc()
                return [], "", status_html(f"Error: {e}", "#d32f2f")

            elapsed = time.time() - t0
            top_n = html.escape(results[0][0]) if results else "-"
            top_s = results[0][1] if results else 0

            if top_s > 0.85: sc, st = "#2e7d32", "High confidence"
            elif top_s > 0.65: sc, st = "#ff9800", "Medium confidence"
            else: sc, st = "#d32f2f", "Low confidence"

            items = []
            lines = [f'<div style="padding-bottom:8px"><b style="color:{sc}">{st}</b> '
                     f'Top-1: <b>{top_n}</b> - {elapsed*1000:.0f}ms</div>']
            for i, (name, sim, pil) in enumerate(results):
                items.append((pil, f"#{i+1} {conf_label(sim)} ({sim:.3f})"))
                lines.append(f'<div style="padding:2px 0">'
                             f'<b>#{i+1}</b> {html.escape(name)} {conf_bar(sim)}</div>')

            yield items, "\n".join(lines), status_html(
                f"Done - {elapsed*1000:.0f}ms - Top-1: {top_n} ({top_s:.3f})", sc)

        def status_html(msg, color):
            return f'<div style="color:{color};padding:10px 0;font-size:13px">{msg}</div>'

        def on_clear():
            return None, "", status_html("Ready", "#999")

        btn.click(on_search, inputs=[input_img, top_k], outputs=[gallery, detail, progress])
        clear_btn.click(on_clear, outputs=[input_img, detail, progress])

    return app

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--fabric-dir", default="dir/fabric")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    print("=" * 50)
    print(" Fabric Pattern Matcher")
    print("=" * 50)
    index = FabricIndex()
    index.build(args.fabric_dir)
    app = build_ui(index)
    app.launch(server_port=args.port, share=args.share, inbrowser=True)
