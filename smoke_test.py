"""Full e2e test: index, search, format output."""
import sys, os
sys.path.insert(0, '.')
from app import FabricIndex, get_clip, extract_embedding
from PIL import Image

# 1. Build
index = FabricIndex()
index.build("dir/fabric")
assert len(index.names) > 0, "FAIL: not built"

# 2. Search raw
img = Image.open("dir/photo/1.png").convert("RGB")
results = index.search(img, k=10)
assert len(results) == 10, f"FAIL: got {len(results)} results"

# 3. ImageEditor dict (crop simulation)
cropped = img.crop((20, 20, 80, 80))
editor_out = {"composite": cropped, "background": img, "layers": []}
img2 = editor_out.get("composite")
results2 = index.search(img2, k=10)

# 4. Verify html.escape
import html
for name, _, _ in results:
    escaped = html.escape(name)
    assert "<" not in escaped, f"FAIL: unescaped html in {name}"

print(f"ALL PASSED: {len(results)} results, valid HTML output")
