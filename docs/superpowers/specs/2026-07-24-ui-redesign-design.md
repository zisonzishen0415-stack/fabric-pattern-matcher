# Fabric Pattern Matcher — UI Redesign Spec

**Date:** 2026-07-24
**Goal:** Redesign UI for speed-first fabric matching: one upload → instant results.

## Current State

- Gradio app with `gr.Image` + custom HTML canvas for crop + hidden textbox for base64 passthrough
- Layout: left column (upload + crop canvas) | right column (stats + slider + buttons) | below (results HTML)
- Upload → must drag to crop → auto-search (awkward mandatory crop step)
- Settings not persisted between sessions

## Target State

### Architecture

Replace the Gradio component-heavy layout with a **single-page app** feel:

- **Top bar**: fixed, compact, all controls in one row
- **Result area**: fills remaining viewport, responsive CSS grid
- **Crop modal**: overlay, triggered on demand (not in the main flow)
- Most UI rendered via `gr.HTML` with inline JS/CSS; only the slider/dropdown and hidden inputs use Gradio components

### Top Bar (48-56px, fixed, full width)

Left to right:

1. **Logo + brand** — `logo_ico.png` (small, ~32px) + fabric count ("5,649 fabrics")
2. **Upload zone** — compact upload button/drop target. States:
   - *Empty*: Dashed-border placeholder text "Drop or paste photo here"
   - *Has image*: Small thumbnail (48×48, rounded) with a tiny ✕ to remove
   - Click thumbnail → opens crop modal
3. **Results selector** — dropdown with preset values: 5 / 10 / 15 / 20 / 30 / 50. Default 15
4. **Clear button** — resets everything to empty state
5. **Search status** — subtle spinner/pulse during search, hidden otherwise

Design: dark/neutral background (`#1a1a2e` or similar), white text, subtle border-bottom separator.

### Result Area (fills remaining space)

- Responsive CSS grid, no Gradio wrapper
- Column count auto-adapts: 5-6 cols (wide), 3-4 (medium), 2 (narrow)
- Each card:
  - Image (contained in fixed-aspect-ratio container, `object-fit: contain`, light gray background)
  - Info row below: rank number, similarity score (color-coded), filename
  - Hover: subtle scale-up or shadow
  - Click: open full-size lightbox (existing modal, keep)
- Color coding for similarity:
  - `>0.85` green (`#2e7d32`)
  - `>0.75` light green (`#4caf50`)
  - `>0.65` orange (`#ff9800`)
  - `>0.50` dark orange (`#f57c00`)
  - `≤0.50` red (`#d32f2f`)

### Crop Modal (overlay)

- Triggered by clicking the upload thumbnail in top bar
- Dark semi-transparent backdrop, centered
- Canvas with the uploaded image, same crop logic as current
- Green dashed rectangle on drag
- Buttons: "Reset" (restore full image), "Confirm Search" (re-search with cropped region)
- Close on backdrop click, ✕ button, or Escape key
- Current results stay visible underneath until "Confirm Search" is clicked

### Data Flow

```
Upload (click/drag/paste)
  → gr.Image receives PIL image
  → Python: PIL → base64 → write to hidden crop_b64 textbox
  → crop_b64.change triggers search
  → Python: decode base64 → CLIP → FAISS → build HTML
  → gr.HTML renders results

Crop (in modal)
  → JS: canvas crop → toDataURL → write to hidden crop_b64 textbox
  → triggers same search pipeline
```

### Persistence

- Settings stored at `~/.fabric_matcher/settings.json`
- On startup: load `top_k` from file, default 15 if missing
- On dropdown change: save to file, re-search with current image (if any)
- Use `platformdirs` or manual `os.path.expanduser` for cross-platform path

### Empty State

- No image uploaded: result area shows centered hint text "Upload a photo to find matching fabrics"
- No results from search: "No matching fabrics found"

### Upload Triggers

- Click upload button in top bar → file dialog
- Drag file anywhere on page → upload
- Ctrl+V paste → upload (clipboard image)

### Error Handling

- Invalid file: show toast/inline error message
- Empty directory: show meaningful message
- Search failure: inline error in result area

## Non-Goals

- Multi-image upload
- History/undo
- User accounts
- Server-side settings sync

## Files to Change

- `app.py` — primary file, ~400 lines, complete rewrite of `build_ui()`
- No new files unless extracting JS to a separate asset

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Top-k selector | Dropdown (preset values) | Simpler than slider for known-good values |
| Crop | Modal overlay | Hidden from main flow, on-demand only |
| Rendering approach | `gr.HTML` with inline CSS/JS | Maximum control over layout and responsiveness |
| Theme | Keep Monochrome base | Lightweight, doesn't fight custom CSS |
| Persistence | JSON file in home dir | Simple, no dependencies, survives restarts |
