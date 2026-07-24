"""
Fabric Pattern Matcher - CLIP + FAISS + Gradio GUI.
"""
import os, time, traceback, hashlib, html, gc, base64, io
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
    vec = emb.cpu().numpy().flatten().astype(np.float32)
    # Free GPU tensors to prevent CUDA cache accumulation
    del t, emb
    return vec

# ============================================================
# FAISS Index
# ============================================================
class FabricIndex:
    def __init__(self):
        self.names = []; self.embeddings = None; self.index = None

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
                self.names.append(f)
            except: pass
            finally:
                try: pil.close()
                except: pass
            if (i+1) % 100 == 0:
                print(f"  {i+1}/{len(files)}")
                gc.collect()

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
        results = []
        for s, i in zip(sims[0], idxs[0]):
            if 0 <= i < len(self.names):
                fname = self.names[i]
                try:
                    pil = Image.open(os.path.join(self._fabric_dir, fname)).convert("RGB")
                except Exception:
                    pil = None
                results.append((fname, float(s), pil))
        return results

# ============================================================
# GUI
# ============================================================
def build_ui(index):
    n_fabrics = len(index.names)

    head = r"""
    <script>
    (function boot() {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
        return;
      }
      if (window.__fabricBooted) return;
      window.__fabricBooted = true;

      // ── Modal overlay ─────────────────────────────────────────
      var modal = document.createElement('div');
      modal.id = 'img-modal';
      modal.innerHTML = '<span class="modal-close">&times;</span><img id="modal-img" src="">';
      document.body.appendChild(modal);
      var mimg = document.getElementById('modal-img');
      window._openModal = function(src) {
        mimg.src = src;
        modal.classList.add('active');
      };
      modal.addEventListener('click', function(e) {
        if (e.target === modal || e.target.classList.contains('modal-close'))
          modal.classList.remove('active');
      });
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') modal.classList.remove('active');
      });

      // ── Canvas crop helper ────────────────────────────────────
      function initCropCanvas(c) {
        if (!c || c.dataset.init) return;
        c.dataset.init = '1';
        var ow = parseInt(c.dataset.ow), oh = parseInt(c.dataset.oh);
        var dw = parseInt(c.dataset.dw), dh = parseInt(c.dataset.dh);
        var img = new Image();
        img.onload = function() {
          var ctx = c.getContext('2d');
          ctx.drawImage(img, 0, 0, dw, dh);
          var dragging = false, sx = 0, sy = 0, ex = 0, ey = 0;
          var hint = document.getElementById('crop-hint');
          function getPos(e) {
            var r = c.getBoundingClientRect();
            return {
              x: Math.min(Math.max((e.clientX - r.left) * dw / r.width, 0), dw),
              y: Math.min(Math.max((e.clientY - r.top) * dh / r.height, 0), dh)
            };
          }
          function overlay(rx, ry, rw, rh) {
            ctx.clearRect(0, 0, dw, dh);
            ctx.fillStyle = 'rgba(0,0,0,0.25)';
            ctx.fillRect(0, 0, dw, dh);
            ctx.drawImage(img, rx, ry, rw, rh, rx, ry, rw, rh);
            ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2;
            ctx.setLineDash([5, 3]); ctx.strokeRect(rx, ry, rw, rh);
          }
          function finishCrop() {
            if (!dragging) return;
            dragging = false;
            var rx = Math.min(sx, ex), ry = Math.min(sy, ey);
            var rw = Math.abs(ex - sx), rh = Math.abs(ey - sy);
            if (rw < 10 || rh < 10) { rx = 0; ry = 0; rw = dw; rh = dh; }
            var sc = ow / dw;
            var ox1 = Math.round(rx * sc), oy1 = Math.round(ry * sc);
            var ox2 = Math.round((rx + rw) * sc), oy2 = Math.round((ry + rh) * sc);
            var oc = document.createElement('canvas');
            oc.width = ox2 - ox1; oc.height = oy2 - oy1;
            oc.getContext('2d').drawImage(img, ox1, oy1, oc.width, oc.height, 0, 0, oc.width, oc.height);
            var b64 = oc.toDataURL('image/jpeg', 0.92);

            // Write to hidden textbox AND click Search button
            var tb = document.querySelector('#crop-result-textbox textarea');
            if (tb) {
              var s = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
              s.call(tb, b64);
              tb.dispatchEvent(new Event('input', { bubbles: true }));
            }
            var sb = document.querySelector('#search-btn');
            if (sb) setTimeout(function() { sb.click(); }, 80);

            ctx.clearRect(0, 0, dw, dh);
            ctx.drawImage(img, 0, 0, dw, dh);
            ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2;
            ctx.setLineDash([]); ctx.strokeRect(rx, ry, rw, rh);
            if (hint) hint.style.opacity = '1';
          }
          c.addEventListener('mousedown', function(e) {
            dragging = true; var p = getPos(e);
            sx = p.x; sy = p.y; ex = p.x; ey = p.y;
            if (hint) hint.style.opacity = '0';
          });
          c.addEventListener('mousemove', function(e) {
            if (!dragging) return;
            var p = getPos(e); ex = p.x; ey = p.y;
            overlay(Math.min(sx, ex), Math.min(sy, ey), Math.abs(ex - sx), Math.abs(ey - sy));
          });
          c.addEventListener('mouseup', finishCrop);
          c.addEventListener('mouseleave', function() {
            if (dragging) { dragging = false; ctx.clearRect(0, 0, dw, dh); ctx.drawImage(img, 0, 0, dw, dh); }
          });
        };
        img.src = 'data:image/jpeg;base64,' + c.dataset.b64;
      }

      // Watch whole document for crop‑canvas
      new MutationObserver(function() {
        var c = document.getElementById('crop-canvas');
        if (c && !c.dataset.init) initCropCanvas(c);
      }).observe(document.body, { childList: true, subtree: true });

      // Event delegation — click result image → modal
      document.addEventListener('click', function(e) {
        var img = e.target.closest('#results-grid img');
        if (img) { e.preventDefault(); window._openModal(img.src); }
      });
    })();
    </script>
    """

    css = """
    footer { display: none !important; }
    #results-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; padding: 12px 0; }
    #results-grid .card { border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden; background: #fff; }
    #results-grid .card img { width: 100%; height: 200px; object-fit: contain; display: block; background: #f5f5f5; cursor: pointer; }
    #results-grid .card img:hover { opacity: 0.85; }
    #results-grid .card .info { padding: 6px 8px; font-size: 12px; color: #555; line-height: 1.4; }
    #results-grid .card .rank { font-weight: 600; font-size: 14px; color: #222; }
    #img-modal { display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                  background: rgba(0,0,0,0.88); z-index: 9999; cursor: pointer; }
    #img-modal.active { display: flex; align-items: center; justify-content: center; }
    #img-modal img { max-width: 90vw; max-height: 90vh; object-fit: contain; cursor: default;
                     box-shadow: 0 4px 24px rgba(0,0,0,0.5); border-radius: 4px; background: #222; }
    #img-modal .modal-close { position: absolute; top: 16px; right: 28px; color: #fff;
                               font-size: 36px; cursor: pointer; line-height: 1; user-select: none; }
    #img-modal .modal-close:hover { opacity: 0.6; }
    """
    with gr.Blocks(css=css, head=head, theme=gr.themes.Monochrome(), title="Fabric Pattern Matcher") as app:
        with gr.Row():
            with gr.Column(scale=3):
                input_img = gr.Image(
                    type="pil",
                    label="Upload, paste, or drag photo here",
                    sources=["upload", "clipboard"],
                )
                crop_canvas = gr.HTML(
                    value='<div style="color:#999;text-align:center;padding:40px">Upload a photo to begin</div>',
                    elem_id="crop-canvas-container")
            with gr.Column(scale=1):
                gr.Markdown(f"**{n_fabrics}** fabrics  \n**CLIP ViT-B/32**")
                top_k = gr.Slider(3, 50, value=15, step=1, label="Results")
                with gr.Row():
                    btn = gr.Button("Search", variant="primary", size="lg", elem_id="search-btn")
                    clear_btn = gr.Button("Clear", size="sm")

        crop_b64 = gr.Textbox(visible=False, elem_id="crop-result-textbox")
        results_html = gr.HTML(value='<div style="color:#999;text-align:center;padding:40px">Results will appear here</div>')

        # ── helpers ──────────────────────────────────────────────
        def pil_to_b64(pil_img):
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()

        def conf_color(s):
            if s > 0.85: return "#2e7d32"
            if s > 0.75: return "#4caf50"
            if s > 0.65: return "#ff9800"
            if s > 0.50: return "#f57c00"
            return "#d32f2f"

        def conf_label(s):
            if s > 0.85: return "Very High"
            if s > 0.75: return "High"
            if s > 0.65: return "Medium"
            if s > 0.50: return "Low"
            return "Very Low"

        # ── canvas HTML generator ────────────────────────────────
        def render_crop_canvas(pil_img):
            if pil_img is None:
                return '<div style="color:#999;text-align:center;padding:40px">Upload a photo to begin</div>'
            if pil_img.mode == 'RGBA':
                pil_img = pil_img.convert('RGB')
            ow, oh = pil_img.size
            max_w, max_h = 800, 600
            scale = min(max_w / ow, max_h / oh, 1.0)
            dw, dh = int(ow * scale), int(oh * scale)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=92)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"""<div style="position:relative;display:inline-block;border:1px solid #ddd;
                border-radius:8px;overflow:hidden;background:#fafafa;max-width:100%;">
              <canvas id="crop-canvas" width="{dw}" height="{dh}"
                      style="display:block;cursor:crosshair;max-width:100%;height:auto;"
                      data-b64="{b64}" data-ow="{ow}" data-oh="{oh}"
                      data-dw="{dw}" data-dh="{dh}"></canvas>
              <div id="crop-hint" style="position:absolute;bottom:8px;left:50%;transform:translateX(-50%);
                   background:rgba(0,0,0,0.7);color:#fff;padding:4px 12px;border-radius:4px;font-size:12px;
                   pointer-events:none;transition:opacity 0.25s;">
                Drag to refine selection (optional)
              </div>
            </div>"""

        # ── search ───────────────────────────────────────────────
        def on_search(cropped_b64, k):
            if not cropped_b64:
                return '<div style="color:#999;text-align:center;padding:40px">Upload and crop a photo first</div>'
            try:
                if ',' in cropped_b64:
                    cropped_b64 = cropped_b64.split(',', 1)[1]
                img_data = base64.b64decode(cropped_b64)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            except Exception:
                return '<div style="color:#d32f2f;text-align:center;padding:40px">Failed to decode image</div>'
            w, h_img = img.size
            if w < 224 or h_img < 224:
                scale = max(224.0 / w, 224.0 / h_img)
                img = img.resize((int(w * scale), int(h_img * scale)), Image.LANCZOS)
            t0 = time.time()
            try:
                results = index.search(img, int(k))
            except Exception:
                return '<div style="color:#d32f2f;text-align:center;padding:40px">Search error</div>'
            elapsed = time.time() - t0
            if not results:
                return '<div style="color:#999;text-align:center;padding:40px">No results</div>'
            cards = []
            for i, (name, sim, pil) in enumerate(results):
                b64_img = pil_to_b64(pil)
                cc = conf_color(sim)
                cards.append(
                    f'<div class="card">'
                    f'<img src="data:image/jpeg;base64,{b64_img}" alt="{html.escape(name)}">'
                    f'<div class="info">'
                    f'<span class="rank" style="color:{cc}">#{i+1}</span> '
                    f'({sim:.3f}) {conf_label(sim)}<br>'
                    f'<span style="font-size:11px">{html.escape(name)}</span>'
                    f'</div></div>'
                )
            return (f'<div style="font-size:12px;color:#666;padding:4px 0">'
                    f'{len(results)} results · {elapsed*1000:.0f}ms</div>'
                    f'<div id="results-grid">{"".join(cards)}</div>')

        # ── callbacks ────────────────────────────────────────────
        def on_upload(pil_img):
            if pil_img is None:
                return render_crop_canvas(None), '', (
                    '<div style="color:#999;text-align:center;padding:40px">'
                    'Upload a photo to begin</div>')
            # Auto-search with full image immediately; crop canvas
            # remains available for optional refinement
            b64 = pil_to_b64(pil_img)
            return render_crop_canvas(pil_img), b64, ''

        def on_clear():
            return None, (
                '<div style="color:#999;text-align:center;padding:40px">'
                'Upload a photo to begin</div>'), '', (
                '<div style="color:#999;text-align:center;padding:40px">'
                'Results will appear here</div>')

        # ── event wiring ─────────────────────────────────────────
        input_img.change(on_upload,
                         inputs=[input_img],
                         outputs=[crop_canvas, crop_b64, results_html])
        crop_b64.change(on_search,
                        inputs=[crop_b64, top_k],
                        outputs=[results_html])
        btn.click(on_search,
                  inputs=[crop_b64, top_k],
                  outputs=[results_html])
        clear_btn.click(on_clear,
                        outputs=[input_img, crop_canvas, crop_b64, results_html])

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
