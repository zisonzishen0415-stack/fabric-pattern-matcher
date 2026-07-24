"""
ResNet50 + FAISS fabric pattern matcher.
- ResNet50 extracts 2048-dim embeddings from penultimate layer
- Pre-trained on ImageNet (1M images). No network needed after first load.
- FAISS for fast vector search.
"""
import os, time
import cv2
import numpy as np
import faiss
from .preprocessing import imread_unicode


# Lazy model
_model = None
_preprocess = None


def _get_model():
    global _model, _preprocess
    if _model is None:
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        print("  Loading ResNet50 from local cache...")
        _model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        _model.eval()
        _model.fc = torch.nn.Identity()
        _preprocess = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return _model, _preprocess


def _extract_embedding(img_rgb, model, preprocess):
    """Extract 2048-dim ResNet50 embedding."""
    import torch
    tensor = preprocess(img_rgb).unsqueeze(0)
    with torch.no_grad():
        embedding = model(tensor)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding.numpy().flatten().astype(np.float32)


class FabricIndex:
    def __init__(self):
        self.fabric_names = []
        self.embeddings = None
        self.faiss_index = None
        self.is_built = False

    def build(self, fabric_dir):
        model, preprocess = _get_model()
        files = sorted(os.listdir(fabric_dir))

        print(f"  Extracting ResNet50 embeddings from {len(files)} fabrics...")
        vectors = []
        for i, f in enumerate(files):
            try:
                img = imread_unicode(os.path.join(fabric_dir, f))
                if img is None or img.size == 0:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                emb = _extract_embedding(img_rgb, model, preprocess)
                vectors.append(emb)
                self.fabric_names.append(f)
            except Exception:
                continue
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(files)}")

        self.embeddings = np.array(vectors, dtype=np.float32)
        n, d = self.embeddings.shape
        print(f"  {n} valid embeddings ({d}-dim)")

        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.embeddings)
        print(f"  Index built. {n} fabrics.")

        self.is_built = True
        return self


class FabricMatcher:
    def __init__(self, index):
        self.index = index

    def match(self, photo_path, top_n=None, verbose=True):
        model, preprocess = _get_model()
        t0 = time.time()

        photo = imread_unicode(photo_path)
        photo_rgb = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        q_emb = _extract_embedding(photo_rgb, model, preprocess)

        k = min(len(self.index.fabric_names), 100)
        similarities, indices = self.index.faiss_index.search(q_emb.reshape(1, -1), k=k)

        results = []
        for sim, i in zip(similarities[0], indices[0]):
            if 0 <= i < len(self.index.fabric_names):
                results.append({'name': self.index.fabric_names[i], 'score': float(sim)})

        t_match = time.time() - t0
        if verbose:
            n = top_n or min(5, len(results))
            print(f"  {os.path.basename(photo_path)} [{t_match*1000:.0f}ms] "
                  f"top: {[r['name'][:30] for r in results[:3]]}")

        return results

    def eval_all(self, photo_dir, fabric_dir):
        model, preprocess = _get_model()

        ffiles = sorted([f for f in os.listdir(fabric_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))],
                        key=lambda x: int(''.join(c for c in os.path.splitext(x)[0] if c.isdigit()) or 0))
        pfiles = sorted([f for f in os.listdir(photo_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))],
                        key=lambda x: int(''.join(c for c in os.path.splitext(x)[0] if c.isdigit()) or 0))

        levels = [1, 3, 5, 10, 20, 50, 100]
        recalls = {k: 0 for k in levels}
        total, ranks = 0, []

        for pf in pfiles:
            pid = ''.join(c for c in os.path.splitext(pf)[0] if c.isdigit())
            truth = [f for f in ffiles if ''.join(c for c in os.path.splitext(f)[0] if c.isdigit()) == pid]
            if not truth: continue
            total += 1
            try:
                results = self.match(os.path.join(photo_dir, pf), verbose=False)
            except Exception as e:
                print(f"  ERR {pf}: {e}")
                continue
            names = [r['name'] for r in results]
            rank = names.index(truth[0]) + 1 if truth[0] in names else len(names) + 1
            ranks.append(rank)
            for k in levels:
                if truth[0] in names[:k]:
                    recalls[k] += 1
            print(f"  {pf:<10} #{rank:<4} {names[:3]}")

        print(f"\n  Library: {len(ffiles)}  Mean rank: {np.mean(ranks):.1f}")
        for k in levels:
            if k > len(ffiles): break
            acc = recalls[k] / total * 100 if total > 0 else 0
            bar = '#' * int(acc / 5) + '-' * (20 - int(acc / 5))
            print(f"  Top-{k:<6} {recalls[k]}/{total} = {acc:3.0f}% {bar}")
        return {'recalls': recalls, 'total': total, 'ranks': ranks}
