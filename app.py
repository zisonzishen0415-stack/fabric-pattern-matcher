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
        if not os.path.isdir(self._fabric_dir):
            raise FileNotFoundError(f"Directory not found: {self._fabric_dir}")
        cache_emb = os.path.join(self.cache_dir, "embeddings.npy")
        cache_names = os.path.join(self.cache_dir, "names.txt")
        cache_hash = os.path.join(self.cache_dir, "hash.txt")
        os.makedirs(self.cache_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(fabric_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
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
        if len(self.embeddings) == 0:
            raise RuntimeError(f"No valid images found in {self._fabric_dir}")
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
    n_fabrics = len(index.names)
    css = """
    footer { display: none !important; }
    """
    with gr.Blocks(css=css, theme=gr.themes.Monochrome(), title="Fabric Pattern Matcher") as app:
        # Top row: upload + controls
        with gr.Row():
            with gr.Column(scale=3):
                input_img = gr.ImageEditor(
                    type="pil",
                    label="Upload photo, click ✂ to crop fabric area",
                    canvas_size=(800, 800),
                    transforms=["crop"],
                    brush=False,
                    layers=False,
                )
            with gr.Column(scale=1):
                gr.Markdown(f"**{n_fabrics}** fabrics  \n**CLIP ViT-B/32**")
                top_k = gr.Slider(3, 50, value=15, step=1, label="Results")
                with gr.Row():
                    btn = gr.Button("Search", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear", size="sm")

        # Results: full-width gallery
        gallery = gr.Gallery(
            label="Results",
            columns=5,
            rows=1,
            height=300,
            object_fit="contain",
        )

        def conf_label(s):
            if s > 0.85: return "Very High"
            if s > 0.75: return "High"
            if s > 0.65: return "Medium"
            if s > 0.50: return "Low"
            return "Very Low"

        def on_search(img, k):
            if img is None:
                return None
            if isinstance(img, dict):
                img = img.get("composite") or img.get("background") or img
            if img is None:
                return None

            t0 = time.time()
            try:
                results = index.search(img, int(k))
            except Exception:
                return None

            elapsed = time.time() - t0
            items = [(pil, f"#{i+1} ({sim:.3f}) {conf_label(sim)}")
                     for i, (_, sim, pil) in enumerate(results)]

            # Auto-compute rows based on result count
            g_rows = max(1, (len(items) + 4) // 5)
            g_height = g_rows * 200

            return gr.update(value=items, rows=g_rows, height=g_height)

        def on_clear():
            return None

        btn.click(on_search, inputs=[input_img, top_k], outputs=[gallery])
        clear_btn.click(on_clear, outputs=[input_img])

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
    # Pre-load CLIP model before building UI
    print("Pre-loading CLIP...")
    get_clip()
    index = FabricIndex()
    index.build(args.fabric_dir)
    app = build_ui(index)
    app.launch(server_port=args.port, share=args.share, inbrowser=True)
