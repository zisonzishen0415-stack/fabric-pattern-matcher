"""
Fabric Pattern Matcher - CLIP + FAISS + Gradio GUI.
"""
import os, time, traceback, hashlib, html, gc, base64, io, json
import numpy as np
import faiss
import gradio as gr
from PIL import Image

# ============================================================
# Settings Persistence
# ============================================================
def _settings_path():
    return os.path.join(os.path.expanduser("~"), ".fabric_matcher", "settings.json")

def load_settings():
    defaults = {"top_k": 15}
    try:
        path = _settings_path()
        if os.path.exists(path):
            with open(path) as f:
                saved = json.load(f)
            defaults.update(saved)
    except Exception:
        pass
    return defaults

def save_settings(settings):
    try:
        path = _settings_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(settings, f)
    except Exception:
        pass

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
# GUI Helpers
# ============================================================
TOP_K_OPTIONS = [5, 10, 15, 20, 30, 50, 67, 85, 100]
EMPTY_STATE = """<div class="empty-state"><div class="empty-icon">&#128247;</div><div>Upload a photo to find matching fabrics</div></div>"""

def _logo_b64():
    logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_ico.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

def pil_to_b64(pil_img, quality=85):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
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

# ============================================================
# Top Bar HTML
# ============================================================
def make_top_bar_html(n_fabrics, logo_b64_str):
    logo_img = f'<img src="data:image/png;base64,{logo_b64_str}" alt="logo">' if logo_b64_str else ""
    return f"""
    <div id="top-bar-inner">
      <div class="tb-logo-area">
        {logo_img}
        <span class="tb-count">{n_fabrics} fabrics</span>
      </div>
      <div class="tb-upload-area">
        <button class="tb-upload-btn" id="upload-btn" title="Click, paste, or drag a photo anywhere">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
          <span id="upload-btn-label">Upload photo</span>
        </button>
        <div class="tb-thumb-wrap" id="thumb-wrap">
          <img id="thumb-img" src="" alt="uploaded" title="Click to crop region">
          <button class="tb-thumb-x" id="thumb-x" title="Remove">&times;</button>
        </div>
      </div>
    </div>
    """

# ============================================================
# CSS
# ============================================================
APP_CSS = """
.gradio-container { max-width: 100% !important; margin: 0 !important; }
footer { display: none !important; }

/* ── Top bar row ── */
#top-bar-row {
  background: #fafafa !important;
  border-bottom: 1px solid #e5e5e5 !important;
  border-radius: 0 0 12px 12px !important;
  padding: 8px 20px !important;
  gap: 0 !important;
  align-items: center !important;
  position: sticky !important; top: 0 !important; z-index: 999 !important;
  flex-wrap: nowrap !important;
}

/* ── Inner top bar ── */
#top-bar-inner {
  display: flex; align-items: center; gap: 16px; flex: 1; min-width: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  font-size: 13px; color: #333;
}
.tb-logo-area { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.tb-logo-area img { height: 28px; width: auto; border-radius: 4px; }
.tb-count { color: #888; white-space: nowrap; font-size: 12px; }

.tb-upload-area { flex: 1; display: flex; align-items: center; gap: 10px; min-width: 0; }
.tb-upload-btn {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 6px 14px; border: 1px dashed #bbb; border-radius: 6px;
  background: transparent; color: #555; cursor: pointer;
  font-size: 13px; white-space: nowrap; transition: all 0.2s;
  font-family: inherit;
}
.tb-upload-btn:hover { border-color: #4caf50; color: #333; }

.tb-thumb-wrap { display: none; position: relative; flex-shrink: 0; }
.tb-thumb-wrap.has-img { display: block; }
.tb-thumb-wrap img {
  width: 36px; height: 36px; border-radius: 5px; object-fit: cover;
  cursor: pointer; border: 2px solid transparent; transition: border-color 0.2s;
}
.tb-thumb-wrap img:hover { border-color: #4caf50; }
.tb-thumb-x {
  position: absolute; top: -6px; right: -6px;
  width: 16px; height: 16px; border-radius: 50%;
  background: #e53935; color: #fff; border: none; cursor: pointer;
  font-size: 10px; line-height: 16px; text-align: center; padding: 0;
}

/* Controls at row far right */
#tb-dropdown, #tb-eval-toggle { display: flex !important; align-items: center !important; gap: 6px !important; }
#tb-dropdown label, #tb-eval-toggle label { margin: 0 !important; font-size: 12px !important; white-space: nowrap !important; color: #888 !important; font-weight: 400 !important; }
#tb-eval-toggle input { accent-color: #ff9800; }
#tb-dropdown .wrap, #tb-eval-toggle .wrap { flex: none !important; }
#tb-dropdown select, #tb-dropdown input { font-size: 13px !important; }

/* ── Eval mode ── */
#eval-panel { display: none; padding: 10px 18px; gap: 10px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #e8e8e8; }
#eval-panel.show { display: flex; }
#eval-preview { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
#eval-preview-img { width: 100px; height: 100px; object-fit: cover; border-radius: 6px; border: 2px solid #e8e8e8; }
#eval-preview-label { font-size: 12px; color: #555; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#eval-buttons { display: flex; gap: 6px; }
#eval-buttons button {
  padding: 5px 12px; border-radius: 5px; cursor: pointer;
  font-size: 12px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#eval-export-btn { border: 1px solid #4caf50; background: #4caf50; color: #fff; }
#eval-export-btn:hover { background: #43a047; }
#eval-no-match-btn { border: 1px solid #e53935; background: #fff; color: #e53935; }
#eval-no-match-btn:hover { background: #fff5f5; }
#eval-prev-btn, #eval-next-btn { border: 1px solid #1976d2; background: #fff; color: #1976d2; }
#eval-prev-btn:hover, #eval-next-btn:hover { background: #f0f7ff; }
#eval-status { font-size: 11px; color: #888; }
#results-grid .card .eval-radio-label {
  display: none; align-items: center; justify-content: center; gap: 4px;
  padding: 5px 8px; border-top: 1px solid #e8e8e8; background: #fafafa;
  font-size: 11px; color: #666; cursor: pointer; user-select: none;
}
#results-grid .card .eval-radio-label input { accent-color: #4caf50; width: 14px; height: 14px; cursor: pointer; }
#results-grid .card.eval-active { border-color: #4caf50; box-shadow: 0 0 0 2px rgba(76,175,80,0.3); }
#results-grid .card.eval-active .eval-radio-label { background: #e8f5e9; color: #2e7d32; }

/* ── Results ── */
#results-wrap { padding: 14px 18px; min-height: calc(100vh - 80px); }
#results-meta { font-size: 12px; color: #999; padding-bottom: 10px; }
#results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 14px;
}
#results-grid .card {
  border: 1px solid #e8e8e8; border-radius: 10px; overflow: hidden;
  background: #fff; transition: transform 0.15s, box-shadow 0.15s;
}
#results-grid .card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.09); }
#results-grid .card .img-wrap {
  aspect-ratio: 1; background: #f9f9f9;
  display: flex; align-items: center; justify-content: center; overflow: hidden;
}
#results-grid .card .img-wrap img {
  width: 100%; height: 100%; object-fit: contain; cursor: pointer;
  transition: opacity 0.15s;
}
#results-grid .card .img-wrap img:hover { opacity: 0.8; }
#results-grid .card .info {
  padding: 8px 10px; font-size: 12px; color: #666; line-height: 1.5;
  border-top: 1px solid #f2f2f2; background: #fff;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
#results-grid .card .info .rank-num { font-weight: 700; font-size: 14px; }
#results-grid .card .info .fname { font-size: 11px; color: #999; display: block; }

.empty-state {
  display: flex; align-items: center; justify-content: center;
  height: calc(100vh - 200px); color: #ccc; font-size: 16px;
  flex-direction: column; gap: 10px; user-select: none;
}
.empty-state .empty-icon { font-size: 52px; opacity: 0.4; }

/* ── Comparison lightbox ── */
#img-modal {
  display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(0,0,0,0.92); z-index: 10000; cursor: default;
}
#img-modal.active { display: flex; align-items: center; justify-content: center; }
#img-modal .compare-wrap {
  display: flex; gap: 16px; align-items: flex-start; justify-content: center;
  max-width: 95vw;
}
#img-modal .compare-side {
  display: flex; flex-direction: column; align-items: center;
  max-width: 44vw; max-height: 85vh;
}
#img-modal .compare-side img {
  display: block;
  box-shadow: 0 4px 24px rgba(0,0,0,0.6); border-radius: 6px; background: #222;
}
#img-modal .compare-side .img-container {
  display: flex; align-items: center; justify-content: center;
  width: 100%; max-height: 65vh; overflow: hidden;
  border-radius: 6px; background: #222; position: relative;
  box-shadow: 0 4px 24px rgba(0,0,0,0.6);
}
#img-modal .compare-side .img-container img {
  max-width: 100%; max-height: 65vh; object-fit: contain; cursor: grab;
  box-shadow: none; border-radius: 0; background: transparent;
  transform-origin: 0 0; transition: none;
}
#img-modal .compare-side .img-container img.grabbing { cursor: grabbing; }
#img-modal .compare-side .zoom-bar {
  display: flex; align-items: center; gap: 6px; margin-top: 8px;
  color: #aaa; font-size: 12px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  user-select: none;
}
#img-modal .compare-side .zoom-bar button {
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.2);
  color: #ccc; border-radius: 4px; cursor: pointer; font-size: 14px;
  width: 26px; height: 26px; line-height: 1; padding: 0; transition: background 0.15s;
}
#img-modal .compare-side .zoom-bar button:hover { background: rgba(255,255,255,0.25); color: #fff; }
#img-modal .compare-side .zoom-bar .zoom-pct { min-width: 36px; text-align: center; }
#img-modal .compare-label {
  color: #aaa; font-size: 12px; margin-bottom: 6px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
#img-modal .modal-close {
  position: absolute; top: 16px; right: 28px; color: #fff;
  font-size: 36px; cursor: pointer; line-height: 1; user-select: none; z-index: 10001;
}
#img-modal .modal-close:hover { opacity: 0.6; }

/* ── Crop modal ── */
#crop-modal {
  display: none; position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
  background: rgba(0,0,0,0.78); z-index: 10001;
  align-items: center; justify-content: center;
}
#crop-modal.active { display: flex; }
#crop-modal .crop-dialog {
  background: #fff; border-radius: 12px; padding: 20px 24px 16px;
  max-width: 90vw; max-height: 90vh; position: relative;
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}
#crop-modal .crop-dialog h3 { margin: 0 0 10px; font-size: 15px; color: #333; }
#crop-modal canvas {
  display: block; cursor: crosshair; border-radius: 6px;
  border: 1px solid #e0e0e0; max-width: 100%; background: #fafafa;
}
#crop-modal .crop-hint { text-align: center; color: #aaa; font-size: 12px; margin-top: 6px; }
#crop-modal .crop-actions { display: flex; gap: 10px; justify-content: flex-end; margin-top: 12px; }
#crop-modal .crop-actions button {
  padding: 7px 18px; border-radius: 6px; border: 1px solid #ddd;
  cursor: pointer; font-size: 13px; background: #fff; color: #555;
}
#crop-modal .crop-actions button:hover { background: #f5f5f5; }
#crop-modal .crop-actions .btn-confirm { background: #4caf50; color: #fff; border-color: #4caf50; }
#crop-modal .crop-actions .btn-confirm:hover { background: #43a047; }
#crop-modal .crop-close {
  position: absolute; top: 6px; right: 14px;
  background: none; border: none; font-size: 24px; cursor: pointer; color: #aaa; line-height: 1;
}
#crop-modal .crop-close:hover { color: #333; }
"""

# ============================================================
# JavaScript
# ============================================================
APP_JS = r"""
(function boot() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
    return;
  }
  if (window.__fbBooted) return;
  window.__fbBooted = true;

  /* ── Comparison lightbox ── */
  var lb = document.createElement('div');
  lb.id = 'img-modal';
  lb.innerHTML = '<span class="modal-close">&times;</span>' +
    '<div class="compare-wrap">' +
    '<div class="compare-side">' +
      '<div class="compare-label">Your Photo</div>' +
      '<div class="img-container" id="lb-container-left"><img id="modal-img-left" src=""></div>' +
      '<div class="zoom-bar">' +
        '<button id="zoom-out-left" title="Zoom out">−</button>' +
        '<span class="zoom-pct" id="zoom-pct-left">100%</span>' +
        '<button id="zoom-in-left" title="Zoom in">+</button>' +
        '<button id="zoom-reset-left" title="Reset">1:1</button>' +
      '</div>' +
    '</div>' +
    '<div class="compare-side">' +
      '<div class="compare-label">Fabric Match</div>' +
      '<div class="img-container" id="lb-container-right"><img id="modal-img-right" src=""></div>' +
      '<div class="zoom-bar">' +
        '<button id="zoom-out-right" title="Zoom out">−</button>' +
        '<span class="zoom-pct" id="zoom-pct-right">100%</span>' +
        '<button id="zoom-in-right" title="Zoom in">+</button>' +
        '<button id="zoom-reset-right" title="Reset">1:1</button>' +
      '</div>' +
    '</div>' +
    '</div>';
  document.body.appendChild(lb);
  var lbLeft  = document.getElementById('modal-img-left');
  var lbRight = document.getElementById('modal-img-right');
  var lbCntL  = document.getElementById('lb-container-left');
  var lbCntR  = document.getElementById('lb-container-right');

  /* ── Zoom state per side ── */
  var zoom = {
    left:  { s: 1, px: 0, py: 0, dragging: false, sx: 0, sy: 0 },
    right: { s: 1, px: 0, py: 0, dragging: false, sx: 0, sy: 0 }
  };

  function clampZoom(v) { return Math.min(Math.max(v, 0.5), 5.0); }

  function applyZoom(side, imgEl, container) {
    var z = zoom[side];
    imgEl.style.transform = 'translate(' + z.px + 'px, ' + z.py + 'px) scale(' + z.s + ')';
    var pct = Math.round(z.s * 100);
    var pctEl = document.getElementById('zoom-pct-' + side);
    if (pctEl) pctEl.textContent = pct + '%';
  }

  function resetZoom(side, imgEl) {
    zoom[side].s = 1; zoom[side].px = 0; zoom[side].py = 0;
    applyZoom(side, imgEl);
  }

  /* ── Wheel → zoom ── */
  function onWheel(e, side, imgEl, container) {
    e.preventDefault();
    var z = zoom[side];
    // Zoom toward cursor position
    var rect = container.getBoundingClientRect();
    var mx = e.clientX - rect.left, my = e.clientY - rect.top;
    var fs = clampZoom(z.s * (e.deltaY < 0 ? 1.12 : 0.89));
    var ratio = fs / z.s;
    z.px = mx - ratio * (mx - z.px);
    z.py = my - ratio * (my - z.py);
    z.s = fs;
    if (fs <= 1.01) { z.s = 1; z.px = 0; z.py = 0; }
    applyZoom(side, imgEl);
  }

  /* ── Drag → pan (only when zoomed) ── */
  function onMouseDown(e, side, imgEl) {
    if (zoom[side].s <= 1.01) return;
    e.preventDefault();
    zoom[side].dragging = true;
    zoom[side].sx = e.clientX - zoom[side].px;
    zoom[side].sy = e.clientY - zoom[side].py;
    imgEl.classList.add('grabbing');
  }

  function onMouseMove(e, side, imgEl) {
    var z = zoom[side];
    if (!z.dragging) return;
    z.px = e.clientX - z.sx;
    z.py = e.clientY - z.sy;
    applyZoom(side, imgEl);
  }

  function onMouseUp(side, imgEl) {
    zoom[side].dragging = false;
    imgEl.classList.remove('grabbing');
  }

  /* Wire left image: wheel + drag */
  lbLeft.addEventListener('wheel', function(e) { onWheel(e, 'left', lbLeft, lbCntL); });
  lbLeft.addEventListener('mousedown', function(e) { onMouseDown(e, 'left', lbLeft); });
  document.addEventListener('mousemove', function(e) { onMouseMove(e, 'left', lbLeft); });
  document.addEventListener('mouseup',   function()  { onMouseUp('left', lbLeft); });

  /* Wire right image: wheel + drag (independent zoom) */
  lbRight.addEventListener('wheel', function(e) { onWheel(e, 'right', lbRight, lbCntR); });
  lbRight.addEventListener('mousedown', function(e) { onMouseDown(e, 'right', lbRight); });
  document.addEventListener('mousemove', function(e) { onMouseMove(e, 'right', lbRight); });
  document.addEventListener('mouseup',   function()  { onMouseUp('right', lbRight); });

  /* Zoom bar buttons */
  function zoomBtn(side, imgEl, delta) {
    var z = zoom[side];
    var fs = clampZoom(z.s * delta);
    if (fs <= 1.01) { z.s = 1; z.px = 0; z.py = 0; }
    else { z.s = fs; }
    applyZoom(side, imgEl);
  }
  document.getElementById('zoom-in-left').addEventListener('click', function() { zoomBtn('left', lbLeft, 1.25); });
  document.getElementById('zoom-out-left').addEventListener('click', function() { zoomBtn('left', lbLeft, 0.8); });
  document.getElementById('zoom-reset-left').addEventListener('click', function() { resetZoom('left', lbLeft); });
  document.getElementById('zoom-in-right').addEventListener('click', function() { zoomBtn('right', lbRight, 1.25); });
  document.getElementById('zoom-out-right').addEventListener('click', function() { zoomBtn('right', lbRight, 0.8); });
  document.getElementById('zoom-reset-right').addEventListener('click', function() { resetZoom('right', lbRight); });

  /* Close lightbox → reset zooms */
  var origActivate = function() { zoom.left = { s:1, px:0, py:0, dragging:false, sx:0, sy:0 }; zoom.right = { s:1, px:0, py:0, dragging:false, sx:0, sy:0 }; applyZoom('left', lbLeft); applyZoom('right', lbRight); };
  lb.addEventListener('click', function(e) {
    if (e.target === lb || e.target.classList.contains('modal-close')) {
      lb.classList.remove('active');
      origActivate();
    }
  });
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      lb.classList.remove('active');
      origActivate();
      closeCrop();
    }
  });

  /* ── Crop modal ── */
  var cm = document.createElement('div');
  cm.id = 'crop-modal';
  cm.innerHTML = '<div class="crop-dialog">' +
    '<button class="crop-close" id="crop-close-btn">&times;</button>' +
    '<h3>Crop &amp; Search</h3>' +
    '<canvas id="crop-canvas"></canvas>' +
    '<div class="crop-hint">Drag to select fabric area, then confirm</div>' +
    '<div class="crop-actions">' +
    '<button id="crop-reset-btn">Reset</button>' +
    '<button class="btn-confirm" id="crop-confirm-btn">Confirm Search</button>' +
    '</div></div>';
  document.body.appendChild(cm);
  document.getElementById('crop-close-btn').addEventListener('click', closeCrop);
  cm.addEventListener('click', function(e) { if (e.target === cm) closeCrop(); });

  /* ── Crop state ── */
  var crop = { img: null, ow: 0, oh: 0, dw: 0, dh: 0, cropped: null };
  var cSX = 0, cSY = 0, cEX = 0, cEY = 0, cDragging = false;
  var cropWired = false;

  function wireCropCanvas() {
    var c = document.getElementById('crop-canvas');
    if (!c || cropWired) return;
    cropWired = true;

    function getPos(e) {
      var r = c.getBoundingClientRect();
      return { x: clamp((e.clientX - r.left) * crop.dw / r.width, 0, crop.dw),
               y: clamp((e.clientY - r.top) * crop.dh / r.height, 0, crop.dh) };
    }
    function clamp(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

    function overlay(rx, ry, rw, rh) {
      var ctx = c.getContext('2d');
      ctx.clearRect(0, 0, crop.dw, crop.dh);
      ctx.drawImage(crop.img, 0, 0, crop.dw, crop.dh);
      ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]); ctx.strokeRect(rx, ry, rw, rh);
      ctx.setLineDash([]);
    }

    c.addEventListener('mousedown', function(e) {
      cDragging = true; var p = getPos(e);
      cSX = p.x; cSY = p.y; cEX = p.x; cEY = p.y;
    });
    c.addEventListener('mousemove', function(e) {
      if (!cDragging) return;
      var p = getPos(e); cEX = p.x; cEY = p.y;
      overlay(Math.min(cSX, cEX), Math.min(cSY, cEY), Math.abs(cEX - cSX), Math.abs(cEY - cSY));
    });
    c.addEventListener('mouseup', finishCrop);
    c.addEventListener('mouseleave', function() {
      if (cDragging) { cDragging = false; var ctx = c.getContext('2d'); ctx.clearRect(0, 0, crop.dw, crop.dh); ctx.drawImage(crop.img, 0, 0, crop.dw, crop.dh); }
    });

    function finishCrop() {
      if (!cDragging) return;
      cDragging = false;
      var rx = Math.min(cSX, cEX), ry = Math.min(cSY, cEY);
      var rw = Math.abs(cEX - cSX), rh = Math.abs(cEY - cSY);
      if (rw < 10 || rh < 10) { rx = 0; ry = 0; rw = crop.dw; rh = crop.dh; }
      var sc = crop.ow / crop.dw;
      var ox1 = Math.max(0, Math.round(rx * sc)), oy1 = Math.max(0, Math.round(ry * sc));
      var ox2 = Math.min(crop.ow, Math.round((rx + rw) * sc)), oy2 = Math.min(crop.oh, Math.round((ry + rh) * sc));
      var pw = Math.max(1, ox2 - ox1), ph = Math.max(1, oy2 - oy1);
      var oc = document.createElement('canvas');
      oc.width = pw; oc.height = ph;
      var octx = oc.getContext('2d');
      octx.fillStyle = '#ffffff'; octx.fillRect(0, 0, pw, ph);
      octx.drawImage(crop.img, ox1, oy1, pw, ph, 0, 0, pw, ph);
      crop.cropped = oc.toDataURL('image/jpeg', 0.92).split(',')[1];
      var ctx = c.getContext('2d');
      ctx.clearRect(0, 0, crop.dw, crop.dh);
      ctx.drawImage(crop.img, 0, 0, crop.dw, crop.dh);
      ctx.strokeStyle = '#4caf50'; ctx.lineWidth = 2;
      ctx.strokeRect(rx, ry, rw, rh);
      // Write to hidden textarea; Gradio's Svelte bind:value picks up the input event
      setTextareaValue(crop.cropped);
    }

    document.getElementById('crop-confirm-btn').addEventListener('click', function() {
      if (crop.cropped) setTextareaValue(crop.cropped);
      closeCrop();
    });
    document.getElementById('crop-reset-btn').addEventListener('click', function() {
      crop.cropped = null;
      var ctx = c.getContext('2d');
      ctx.clearRect(0, 0, crop.dw, crop.dh);
      ctx.drawImage(crop.img, 0, 0, crop.dw, crop.dh);
    });
  }

  /* Helper: set hidden Gradio Textbox value, dispatching input for Svelte reactivity */
  function setTextareaValue(val) {
    var ta = document.querySelector('#crop-result-textbox textarea');
    if (!ta) return;
    var desc = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
    if (desc && desc.set) {
      desc.set.call(ta, val);
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }
  }

  function closeCrop() { cm.classList.remove('active'); }

  window.openCropModal = function() {
    var thumb = document.getElementById('thumb-img');
    if (!thumb || !thumb.src) return;
    wireCropCanvas();
    var c = document.getElementById('crop-canvas');
    if (!c) return;
    crop.img = new Image();
    crop.img.onload = function() {
      crop.ow = crop.img.naturalWidth; crop.oh = crop.img.naturalHeight;
      var maxW = Math.min(window.innerWidth * 0.88, 1200), maxH = Math.min(window.innerHeight * 0.82, 800);
      var sc = Math.min(maxW / crop.ow, maxH / crop.oh, 1.0);
      crop.dw = Math.round(crop.ow * sc); crop.dh = Math.round(crop.oh * sc);
      c.width = crop.dw; c.height = crop.dh;
      c.getContext('2d').drawImage(crop.img, 0, 0, crop.dw, crop.dh);
      crop.cropped = null;
      cm.classList.add('active');
    };
    crop.img.src = thumb.src;
  };

  /* ── Top-k select → Gradio dropdown change ── */
  function setGradioSelect(val) {
    var sel = document.querySelector('#tb-dropdown select');
    if (!sel) return;
    var desc = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value');
    if (desc && desc.set) {
      desc.set.call(sel, String(val));
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    }
  }

  /* ── Wire components after Gradio renders ── */
  var obs = new MutationObserver(function(muts, self) {
    var upBtn = document.getElementById('upload-btn');
    var hiddenFile = document.querySelector('#hidden-upload input[type="file"]');
    var thumbWrap = document.getElementById('thumb-wrap');
    var thumbImg = document.getElementById('thumb-img');
    var thumbX = document.getElementById('thumb-x');
    var upLabel = document.getElementById('upload-btn-label');
    var resultsEl = document.getElementById('results-wrap');

    if (!upBtn || !hiddenFile) return;

    /* Click upload button → file dialog (multi-file) */
    upBtn.addEventListener('click', function() { hiddenFile.multiple = true; hiddenFile.click(); });

    /* File selected → process all */
    hiddenFile.addEventListener('change', function() {
      var files = this.files;
      if (!files || files.length === 0) return;
      for (var fi = 0; fi < files.length; fi++) {
        (function(file, idx, total) {
          var reader = new FileReader();
          reader.onload = function(e) {
            var dataUrl = e.target.result;
            if (idx === 0) {
              if (thumbImg) thumbImg.src = dataUrl;
              if (thumbWrap) thumbWrap.classList.add('has-img');
              if (upLabel) upLabel.textContent = files.length > 1 ? (idx + 1) + '/' + total : 'Change photo';
            }
            window._evalNextFname = file.name || '';
            setTextareaValue(dataUrl);
          };
          reader.readAsDataURL(file);
        })(files[fi], fi, files.length);
      }
      var self = this;
      setTimeout(function() { self.value = ''; }, 500);
    });

    /* Drag anywhere → upload */
    document.addEventListener('dragover', function(e) { e.preventDefault(); });
    document.addEventListener('drop', function(e) {
      e.preventDefault();
      var f = e.dataTransfer.files[0];
      if (f && hiddenFile) {
        var dt = new DataTransfer(); dt.items.add(f);
        hiddenFile.files = dt.files;
        hiddenFile.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    /* Ctrl+V → paste */
    document.addEventListener('paste', function(e) {
      var item = e.clipboardData.items[0];
      if (item && item.type.startsWith('image/') && hiddenFile) {
        var f = item.getAsFile();
        var dt = new DataTransfer(); dt.items.add(f);
        hiddenFile.files = dt.files;
        hiddenFile.dispatchEvent(new Event('change', { bubbles: true }));
      }
    });

    /* Result click → comparison modal (left=uploaded, right=fabric) */
    document.addEventListener('click', function(e) {
      var img = e.target.closest('#results-grid img');
      if (img) {
        e.preventDefault();
        var ts = document.getElementById('thumb-img');
        if (lbLeft && ts && ts.src) lbLeft.src = ts.src;
        if (lbRight) lbRight.src = img.src;
        resetZoom('left', lbLeft);
        resetZoom('right', lbRight);
        lb.classList.add('active');
      }
    });

    /* Thumb click → crop modal */
    if (thumbImg) thumbImg.addEventListener('click', function() { window.openCropModal(); });

    /* Thumb X → clear textarea (triggers empty search via Gradio change event) */
    /* Thumb X → clear visual state + empty textarea (triggers empty search).
       Don't touch hiddenFile.value — that breaks Gradio's internal file tracking. */
    if (thumbX) {
      thumbX.addEventListener('click', function(e) {
        e.stopPropagation();
        if (thumbWrap) thumbWrap.classList.remove('has-img');
        if (thumbImg) thumbImg.src = '';
        if (upLabel) upLabel.textContent = 'Upload photo';
        setTextareaValue('');
      });
    }

    /* Hide spinner when results update */
    if (resultsEl) {
      new MutationObserver(function() {
        var sp = document.getElementById('search-spinner');
        if (sp) sp.classList.remove('show');
      }).observe(resultsEl, { childList: true, subtree: true });
    }

    /* ── Eval mode: simple mark-and-record ── */
    var evalOn = false;
    var sessionTrials = [];
    var seenPhotoIds = {};
    var evalPhotoList = [];     // [{b64, id, fname}]
    var evalPhotoIdx = -1;

    var evalPanel = document.createElement('div');
    evalPanel.id = 'eval-panel';
    evalPanel.innerHTML =
      '<div id="eval-preview"><img id="eval-preview-img" src=""><span id="eval-preview-label"></span></div>' +
      '<div id="eval-buttons">' +
        '<button id="eval-no-match-btn">No Match</button>' +
        '<button id="eval-prev-btn">◀ Prev</button>' +
        '<button id="eval-next-btn">Next ▶</button>' +
        '<button id="eval-export-btn">Export All</button>' +
      '</div>' +
      '<span id="eval-status"></span>';
    if (resultsEl && resultsEl.parentNode) resultsEl.parentNode.insertBefore(evalPanel, resultsEl);

    /* Helpers */
    function photoHash(b64) {
      var s = b64.length > 200 ? b64.substring(b64.length - 200) : b64;
      var h = 0;
      for (var i = 0; i < s.length; i++) { h = ((h << 5) - h + s.charCodeAt(i)) | 0; }
      return (h >>> 0).toString(16).substring(0, 8).padStart(8, '0');
    }
    function showEvalRadios() {
      var labels = document.querySelectorAll('#results-grid .eval-radio-label');
      labels.forEach(function(l) { l.style.display = 'flex'; });
    }

    /* ── Load photo from list ── */
    function loadPhotoAt(idx) {
      if (idx < 0 || idx >= evalPhotoList.length) {
        evalPhotoIdx = -1;
        updatePreviewAndStatus();
        return;
      }
      evalPhotoIdx = idx;
      var p = evalPhotoList[idx];
      // Eval panel preview
      document.getElementById('eval-preview-img').src = p.b64;
      document.getElementById('eval-preview-label').textContent = (idx + 1) + '/' + evalPhotoList.length + ' ' + (p.fname || p.id);
      // Also update top-bar thumbnail
      var ti = document.getElementById('thumb-img');
      var tw = document.getElementById('thumb-wrap');
      var ul = document.getElementById('upload-btn-label');
      if (ti) ti.src = p.b64;
      if (tw) tw.classList.add('has-img');
      if (ul) ul.textContent = (idx + 1) + '/' + evalPhotoList.length;
      updatePreviewAndStatus();
      // Bypass intercept — trigger real search
      _origSetTA(p.b64);
    }

    function prevPhoto() {
      if (evalPhotoIdx > 0) loadPhotoAt(evalPhotoIdx - 1);
    }
    function nextPhoto() {
      if (evalPhotoIdx + 1 < evalPhotoList.length) loadPhotoAt(evalPhotoIdx + 1);
    }

    /* ── Gradio Eval checkbox ── */
    var evalCBEl = null;
    function findEvalCheckbox() {
      var w = document.getElementById('tb-eval-toggle');
      return w ? w.querySelector('input[type="checkbox"]') : null;
    }
    function syncEvalState() {
      var cb = findEvalCheckbox();
      if (!cb) return;
      evalCBEl = cb;
      var wasOn = evalOn;
      evalOn = cb.checked;
      if (evalOn && !wasOn) { evalPanel.classList.add('show'); }
      else if (!evalOn && wasOn) { evalPanel.classList.remove('show'); }
      updatePreviewAndStatus();
    }
    var evalCheckWrap = document.getElementById('tb-eval-toggle');
    if (evalCheckWrap) {
      new MutationObserver(function() {
        var cb = findEvalCheckbox();
        if (cb && cb !== evalCBEl) { evalCBEl = cb; cb.addEventListener('change', syncEvalState); syncEvalState(); }
      }).observe(evalCheckWrap, { childList: true, subtree: true });
    }
    new MutationObserver(function() {
      var cb = findEvalCheckbox();
      if (cb && cb !== evalCBEl) { evalCBEl = cb; cb.addEventListener('change', syncEvalState); syncEvalState(); }
    }).observe(document.body, { childList: true, subtree: true });

    /* ── Intercept uploads → add to list ── */
    var _origSetTA = setTextareaValue;
    window._evalIntercept = function(b64) {
      if (!evalOn || !b64 || b64.length < 100) return false;
      // Don't intercept crop results (plain base64 without data: prefix)
      if (b64.indexOf('data:') !== 0) return false;
      var fname = window._evalNextFname || '';
      window._evalNextFname = '';
      var pid = photoHash(b64);
      if (!seenPhotoIds[pid]) seenPhotoIds[pid] = new Date().toISOString();
      // Dedup
      for (var i = 0; i < evalPhotoList.length; i++) { if (evalPhotoList[i].id === pid) return true; /* already there */ }
      evalPhotoList.push({b64: b64, id: pid, fname: fname});
      if (evalPhotoList.length === 1 && evalPhotoIdx < 0) {
        loadPhotoAt(0);
      } else {
        updatePreviewAndStatus();
      }
      return true;
    };
    setTextareaValue = function (val) {
      if (window._evalIntercept(val)) return;
      _origSetTA(val);
    };

    /* ── Log ── */
    function logTrial(radio) {
      var card = radio.closest('.card');
      if (!card) return;
      card.classList.add('eval-active');
      var p = evalPhotoIdx >= 0 && evalPhotoIdx < evalPhotoList.length ? evalPhotoList[evalPhotoIdx] : null;
      sessionTrials.push({
        timestamp: new Date().toISOString(),
        photo_id: p ? p.id : 'none',
        query_fname: p ? p.fname : '',
        photo_index: evalPhotoIdx,
        result_count: document.querySelectorAll('#results-grid .card').length,
        correct_rank: parseInt(card.getAttribute('data-rank') || '0'),
        correct_name: card.getAttribute('data-fname') || '',
        clip_sim: parseFloat(card.getAttribute('data-sim') || '0')
      });
      updatePreviewAndStatus();
    }
    function logNoMatch() {
      var p = evalPhotoIdx >= 0 && evalPhotoIdx < evalPhotoList.length ? evalPhotoList[evalPhotoIdx] : null;
      if (!p) return;
      sessionTrials.push({
        timestamp: new Date().toISOString(),
        photo_id: p.id,
        query_fname: p.fname,
        photo_index: evalPhotoIdx,
        result_count: document.querySelectorAll('#results-grid .card').length,
        correct_rank: -1,
        correct_name: '',
        clip_sim: 0
      });
      updatePreviewAndStatus();
    }

    function updateEvalCardHighlight() {
      var cards = document.querySelectorAll('#results-grid .card');
      cards.forEach(function(c) { c.classList.remove('eval-active'); });
      var checked = document.querySelector('#results-grid .eval-cb:checked');
      if (checked) { var card = checked.closest('.card'); if (card) card.classList.add('eval-active'); }
    }

    function updatePreviewAndStatus() {
      var st = document.getElementById('eval-status');
      var prev = document.getElementById('eval-preview');
      if (!evalOn) {
        if (prev) prev.style.display = 'none';
        if (st) st.textContent = '';
        return;
      }
      if (prev) prev.style.display = 'flex';
      var n = evalPhotoList.length;
      var idx = evalPhotoIdx;
      var p = (idx >= 0 && idx < n) ? evalPhotoList[idx] : null;
      if (st) {
        if (n === 0) {
          st.textContent = 'Upload photos — ' + sessionTrials.length + ' trials logged';
        } else if (p) {
          st.textContent = sessionTrials.length + ' trials / ' + Object.keys(seenPhotoIds).length + ' photos';
        } else {
          st.textContent = Object.keys(seenPhotoIds).length + ' photos loaded — ' + sessionTrials.length + ' trials';
        }
      }
    }

    function exportAllTrials() {
      if (sessionTrials.length === 0) { alert('No trials yet.'); return; }
      var blob = new Blob([JSON.stringify({unique_photos: Object.keys(seenPhotoIds).length, total_trials: sessionTrials.length, trials: sessionTrials}, null, 2)], {type: 'application/json'});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url; a.download = 'eval-session-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
      a.click(); URL.revokeObjectURL(url);
    }

    /* Events */
    document.addEventListener('change', function(e) {
      if (!evalOn) return;
      if (!e.target || !e.target.classList.contains('eval-cb')) return;
      updateEvalCardHighlight();
      logTrial(e.target);
    });
    document.getElementById('eval-export-btn').addEventListener('click', exportAllTrials);
    document.getElementById('eval-no-match-btn').addEventListener('click', logNoMatch);
    document.getElementById('eval-prev-btn').addEventListener('click', prevPhoto);
    document.getElementById('eval-next-btn').addEventListener('click', nextPhoto);

    /* Watch body → re-show radios after results reload */
    var bodyWatcher = new MutationObserver(function() {
      if (!evalOn) return;
      clearTimeout(bodyWatcher._tid);
      bodyWatcher._tid = setTimeout(function() {
        var grid = document.getElementById('results-grid');
        if (!grid) return;
        var labels = grid.querySelectorAll('.eval-radio-label');
        if (labels.length > 0 && getComputedStyle(labels[0]).display === 'none') {
          showEvalRadios();
        }
        updatePreviewAndStatus();
      }, 150);
    });
    bodyWatcher.observe(document.body, { childList: true, subtree: true });

    self.disconnect();
  });
  obs.observe(document.body, { childList: true, subtree: true });
})();
"""

# ============================================================
# build_ui
# ============================================================
def build_ui(index):
    n_fabrics = len(index.names)
    logo_b64_str = _logo_b64()
    settings = load_settings()
    default_k = settings.get("top_k", 15)

    top_bar_html = make_top_bar_html(n_fabrics, logo_b64_str)

    with gr.Blocks(
        css=APP_CSS,
        head=f"<script>{APP_JS}</script>",
        theme=gr.themes.Monochrome(),
        title="Fabric Pattern Matcher"
    ) as app:
        # ── Top bar: logo + upload (HTML) | Color slider + Results (right group) ──
        with gr.Row(elem_id="top-bar-row", equal_height=True):
            gr.HTML(top_bar_html, elem_id="tb-html-area")
            with gr.Column(elem_id="tb-controls-wrap", scale=0):
                top_k = gr.Dropdown(
                    choices=TOP_K_OPTIONS, value=default_k,
                    label="Results", interactive=True,
                    elem_id="tb-dropdown",
                )
                eval_mode = gr.Checkbox(
                    value=False, label="Eval", interactive=True,
                    elem_id="tb-eval-toggle",
                )

        # ── Hidden data pipe components ──
        input_img = gr.Image(
            type="pil", visible=False, elem_id="hidden-upload",
            sources=["upload", "clipboard"],
        )
        crop_b64 = gr.Textbox(visible=False, elem_id="crop-result-textbox")

        # ── Results ──
        results_html = gr.HTML(value=EMPTY_STATE, elem_id="results-wrap")

        # ── Callbacks ──────────────────────────────────────────

        def on_upload(pil_img):
            """PIL → b64 data URL. Gradio writes this to crop_b64, which triggers change → search."""
            if pil_img is None:
                return "", EMPTY_STATE
            return "data:image/jpeg;base64," + pil_to_b64(pil_img, quality=90)

        def on_search(b64_val, k_val):
            if not b64_val or not b64_val.strip():
                return EMPTY_STATE
            try:
                clean = b64_val.split(',', 1)[1] if ',' in b64_val else b64_val
                img_data = base64.b64decode(clean)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            except Exception:
                return """<div class="empty-state"><div style="color:#d32f2f">Failed to decode image</div></div>"""
            w, h = img.size
            if w < 224 or h < 224:
                s = max(224.0 / w, 224.0 / h)
                img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
            t0 = time.time()
            try:
                results = index.search(img, int(k_val))
            except Exception:
                return """<div class="empty-state"><div style="color:#d32f2f">Search error</div></div>"""
            elapsed_ms = (time.time() - t0) * 1000
            if not results:
                return """<div class="empty-state"><div>No matching fabrics found</div></div>"""
            cards = []
            for i, (name, sim, pil) in enumerate(results):
                try:
                    card_b64 = pil_to_b64(pil)
                except Exception:
                    continue
                cc = conf_color(sim)
                cards.append(
                    f'<div class="card" data-fname="{html.escape(name)}" data-sim="{sim:.4f}" data-rank="{i+1}">'
                    f'<div class="img-wrap"><img src="data:image/jpeg;base64,{card_b64}" alt="{html.escape(name)}" loading="lazy"></div>'
                    f'<div class="info">'
                    f'<span class="rank-num" style="color:{cc}">#{i+1}</span> '
                    f'<span style="color:{cc}">{sim:.3f}</span> {conf_label(sim)}'
                    f'<span class="fname">{html.escape(name)}</span>'
                    f'</div><label class="eval-radio-label"><input type="radio" name="eval-correct" class="eval-cb" value="{html.escape(name)}">Correct</label>'
                    f'</div>'
                )
            return (
                f'<div id="results-meta">{len(results)} results &middot; {elapsed_ms:.0f}ms</div>'
                f'<div id="results-grid">{"".join(cards)}</div>'
            )

        def on_top_k_change(k_val):
            save_settings({"top_k": int(k_val)})

        # ── Event wiring ──

        crop_b64.change(
            on_search,
            inputs=[crop_b64, top_k],
            outputs=[results_html]
        )

        top_k.change(
            on_top_k_change,
            inputs=[top_k], outputs=None
        ).then(
            on_search,
            inputs=[crop_b64, top_k],
            outputs=[results_html]
        )

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

    import socket
    print("=" * 50)
    print(" Fabric Pattern Matcher")
    print("=" * 50)
    settings = load_settings()
    print(f"Settings: top_k={settings.get('top_k', 15)}")
    print("Pre-loading CLIP...")
    get_clip()
    index = FabricIndex()
    index.build(args.fabric_dir)
    app = build_ui(index)

    # Print LAN access URLs
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"
    print("=" * 50)
    print(f"  Local:   http://127.0.0.1:{args.port}")
    print(f"  Network: http://{local_ip}:{args.port}")
    print("=" * 50)

    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, inbrowser=True)
