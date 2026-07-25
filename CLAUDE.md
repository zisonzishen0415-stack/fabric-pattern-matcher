# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fabric Pattern Matcher (面料花型智能匹配工具) — upload a photo, find the most similar fabric patterns in a library using CLIP embeddings + FAISS vector search.

## Commands

```bash
# Start the GUI (defaults to dir/fabric, port 7860)
python app.py --fabric-dir dir/fabric

# Start with sharing enabled
python app.py --fabric-dir dir/fabric --share

# Run the smoke test (builds index, searches, validates output)
python smoke_test.py

# Run CLIP evaluation
python eval_clip.py
```

## Architecture

```
app.py                          # Everything: CLIP model, FAISS index, Gradio GUI, custom JS/CSS
├── get_clip()                  # Lazy-loads CLIP ViT-B/32 (open_clip) once, GPU if available
├── extract_embedding()         # PIL image → 512-dim normalized embedding
├── FabricIndex                 # Scans fabric dir, caches embeddings (.fabric_cache/), builds FAISS IndexFlatIP
│   ├── build(fabric_dir)       # MD5-based cache invalidation; rebuilds embeddings only when files change
│   └── search(pil_img, k)      # Embed query → FAISS inner-product search → (name, score, PIL) tuples
├── build_ui(index)             # Constructs the Gradio Blocks UI with all event wiring
└── Settings persistence        # Saves top_k to ~/.fabric_matcher/settings.json via load/save_settings()
```

**Pipeline:** Photo upload → CLIP ViT-B/32 extract 512-dim embedding → FAISS IndexFlatIP (inner product) → ranked results with confidence labels.

**UI data flow:** All upload paths (click button, drag-drop, paste) go through custom JavaScript that writes a base64 data URL to a hidden `crop_b64` textarea. Gradio's `.change()` on that textarea is the single search trigger. There's also a crop modal (canvas-based) that replaces the textarea value with a cropped image before search.

**Caching:** Embeddings are cached under `.fabric_cache/` as `embeddings.npy` + `names.txt` + `hash.txt` (MD5 of sorted filenames). The cache auto-invalidates when the file list changes.

## matcher_v3/ — Legacy CV approach

Pre-CLIP implementation using handcrafted features (LBP, Gabor, GLCM, FFT, ORB keypoints, autocorrelation fingerprints, phase-only correlation). `FabricIndex` (indexer.py) builds these features; `FabricMatcher` (matcher.py) runs POC matching with ORB tiebreaking. `eval_all()` computes Top-K recall metrics. `clip_matcher.py` is a CLIP variant that preceded `app.py`. This directory is not used by the current `app.py`.

## Dependencies

`open-clip-torch`, `faiss-cpu`, `gradio`, `pillow`, `torch`, `numpy`. `matcher_v3/` additionally needs `opencv-python` (cv2).
