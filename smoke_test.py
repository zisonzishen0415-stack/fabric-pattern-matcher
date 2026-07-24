"""Full smoke test: build index + search with real photo."""
import sys, os
sys.path.insert(0, '.')

from PIL import Image
from app import FabricIndex, get_clip, extract_embedding

# 1. Build index
print("=== Test 1: Build Index ===")
index = FabricIndex()
index.build("dir/fabric")
print(f"OK: {len(index.names)} fabrics indexed\n")

# 2. Search with a real photo
print("=== Test 2: Search ===")
photo = Image.open("dir/photo/1.png").convert("RGB")
results = index.search(photo, k=5)
print(f"OK: {len(results)} results")
for name, sim, _ in results:
    print(f"  {name}: {sim:.4f}")

# 3. Search with cropped photo (simulate ImageEditor output)
print("\n=== Test 3: Search with crop ===")
half = photo.crop((0, 0, photo.width // 2, photo.height // 2))
results2 = index.search(half, k=5)
print(f"OK: {len(results2)} results")
for name, sim, _ in results2:
    print(f"  {name}: {sim:.4f}")

# 4. ImageEditor dict format
print("\n=== Test 4: ImageEditor dict handling ===")
# ImageEditor returns {"composite": <PIL>, "background": <PIL>, "layers": []}
editor_output = {"composite": photo, "background": half, "layers": []}
img = editor_output.get("composite") or editor_output.get("background") or editor_output

# Verify it's a valid PIL and can be embedded
model, preprocess = get_clip()
emb = extract_embedding(img, model, preprocess)
print(f"OK: embedding shape = {emb.shape}")

# 5. Search with the editor dict
results3 = index.search(img, k=5)
print(f"OK: {len(results3)} results")
for name, sim, _ in results3:
    print(f"  {name}: {sim:.4f}")

# 6. Verify html.escape works
import html
print(f"\n=== Test 5: html.escape ===")
test_name = "4521_HX2504521.jpg"
print(f"  {test_name} -> {html.escape(test_name)}")

print("\n=== ALL TESTS PASSED ===")
