"""Phase-Only Correlation (POC) matcher. Lighting-invariant structural matching."""
import os, time
import cv2
import numpy as np


def _poc_score(img1, img2, size=256):
    """
    Phase-only correlation: peak value indicates structural similarity.
    Completely invariant to brightness/contrast/illumination.
    """
    g1 = cv2.resize(img1, (size, size)).astype(np.float64)
    g2 = cv2.resize(img2, (size, size)).astype(np.float64)

    # Hann window to reduce boundary effects
    wy = np.hanning(size)
    wx = np.hanning(size)
    window = np.outer(wy, wx)
    g1 = g1 * window
    g2 = g2 * window

    # FFT
    f1 = np.fft.fft2(g1)
    f2 = np.fft.fft2(g2)

    # Phase-only: normalize magnitude → keep only phase (structure)
    eps = 1e-8
    r = (f1 * np.conj(f2)) / (np.abs(f1) * np.abs(f2) + eps)

    # Inverse FFT → correlation plane
    poc = np.fft.fftshift(np.fft.ifft2(r).real)

    # Peak value = similarity score
    peak = np.max(poc)  # higher = more similar structure

    # Also compute sharpness: ratio of peak to surrounding
    cy, cx = size // 2, size // 2
    surrounding = poc[max(0, cy - 5):cy + 6, max(0, cx - 5):cx + 6]
    peak_sharpness = peak / (np.mean(np.abs(surrounding)) + eps)

    return float(peak), float(peak_sharpness)


class FabricMatcher:
    def __init__(self, index):
        self.index = index

    def match(self, photo_path, top_n=None, verbose=True):
        from .preprocessing import imread_unicode
        t0 = time.time()

        photo = imread_unicode(photo_path)
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        scores = {}
        for fname in self.index.fabric_names:
            fgray = self.index.fabric_grays.get(fname)
            if fgray is None:
                scores[fname] = 0
                continue
            peak, sharpness = _poc_score(photo_gray, fgray)
            scores[fname] = peak + sharpness * 0.3  # sharpness as bonus

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        t_match = time.time() - t0

        results = [{'name': f, 'score': s, 'orb_good': 0} for f, s in ranked]

        # ORB on top-30 as tiebreaker
        from .features import extract_orb, match_orb as m_orb
        kp_p, des_p = extract_orb(photo)
        for i in range(min(30, len(results))):
            fn = results[i]['name']
            kp_f = self.index.orb_keypoints.get(fn)
            des_f = self.index.orb_descriptors.get(fn)
            if des_p is not None and des_f is not None and kp_p and kp_f:
                gc, tc, ad, inl = m_orb(des_p, des_f, kp_p, kp_f)
                results[i]['orb_good'] = gc
                results[i]['score'] += inl * 0.01

        results.sort(key=lambda x: x['score'], reverse=True)

        # ORB override
        orb_c = [(i, r['orb_good']) for i, r in enumerate(results[:30])]
        mx = max(g for _, g in orb_c) if orb_c else 0
        if mx >= 5:
            mi = [i for i, g in orb_c if g == mx][0]
            rest = [g for i, g in orb_c if i != mi]
            if mx > 3 * max(rest + [1]):
                boosted = results.pop(mi)
                results.insert(0, boosted)
                results[0]['orb_override'] = True

        if verbose:
            n = top_n or min(5, len(results))
            ovr = " [ORB!]" if results[0].get('orb_override') else ""
            print(f"  {os.path.basename(photo_path)} [{t_match*1000:.0f}ms] "
                  f"top: {[r['name'][:30] for r in results[:3]]}{ovr}")

        return results

    def eval_all(self, photo_dir, fabric_dir):
        print("  Loading fabric grays...")
        for fname in self.index.fabric_names:
            fpath = os.path.join(fabric_dir, fname)
            try:
                self.index.fabric_grays[fname] = cv2.imdecode(
                    np.frombuffer(open(fpath, 'rb').read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            except Exception:
                self.index.fabric_grays[fname] = np.zeros((256, 256), dtype=np.uint8)

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
            ovr = " [ORB!]" if results[0].get('orb_override') else ""
            print(f"  {pf:<10} #{rank:<4} {names[:3]}{ovr}")

        print(f"\n  Library: {len(ffiles)}  Mean rank: {np.mean(ranks):.1f}")
        for k in levels:
            if k > len(ffiles): break
            acc = recalls[k] / total * 100 if total > 0 else 0
            bar = '#' * int(acc / 5) + '-' * (20 - int(acc / 5))
            print(f"  Top-{k:<6} {recalls[k]}/{total} = {acc:3.0f}% {bar}")
        return {'recalls': recalls, 'total': total, 'ranks': ranks}

