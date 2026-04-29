
import fitz  # PyMuPDF
import pdfplumber
import re, json, glob, os
from typing import Dict, List, Tuple
import time
import time
from contextlib import contextmanager
import concurrent.futures
from pyinstrument import Profiler

METRICS = {"total_s": 0.0, "core_s": 0.0}  # accumulates across files in that run

@contextmanager
def timer(bucket_key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        METRICS[bucket_key] = METRICS.get(bucket_key, 0.0) + (time.perf_counter() - t0)
# ----------------------------
# Config / Regex
# ----------------------------
NUM_RE = re.compile(r'^\d{1,4}(\.\d{1,2})?$')
MZ_MIN, MZ_MAX = 10, 1500

def cx(b): return 0.5*(b[0]+b[2])
def cy(b): return 0.5*(b[1]+b[3])

def get_axis_max(words, baseline_y, X0p, X1p, tolerance=15.0):
    """
    Extract the maximum x-axis tick value from words near the baseline.
    Returns None if no valid tick labels found.
    """
    tick_values = []
    
    for w in words:
        t = w["text"].strip()
        b = (w["x0"], w["y0"], w["x1"], w["y1"])
        cx_val = cx(b)
        cy_val = cy(b)
        
        # must be within plot x-range
        if not (X0p <= cx_val <= X1p):
            continue
        
        # must be near the baseline (x-axis tick labels sit just below baseline)
        if not (baseline_y - tolerance <= cy_val <= baseline_y + tolerance):
            continue
        
        # must be a plain integer (x-axis ticks are always integers in GC-MS)
        try:
            v = float(t)
            if v > 0 and v == int(v):  # positive integer only
                tick_values.append(v)
        except:
            continue
    
    return max(tick_values) if tick_values else None
# ----------------------------
# Robust grouped-number splitter
# ----------------------------
def split_grouped_numbers(text, bbox, mz_min=10, mz_max=999):
    import re
    x0, y0, x1, y1 = bbox
    total_w = max(1e-6, x1 - x0)

    def in_range(s):
        try:
            v = int(s)
            return mz_min <= v <= mz_max
        except:
            return False

    def enum_partitions(n, min_len=2, max_len=4, min_parts=2, max_parts=4):
        res = []
        def dfs(rem, cur):
            if rem == 0 and min_parts <= len(cur) <= max_parts:
                res.append(cur[:]); return
            if rem <= 0 or len(cur) >= max_parts:
                return
            for L in range(min_len, max_len+1):
                if L <= rem:
                    cur.append(L)
                    dfs(rem-L, cur)
                    cur.pop()
        dfs(n, [])
        return res

    def score_partition(lengths, pieces):
        s = 10 * sum((L - 3) * (L - 3) for L in lengths)
        s += 6 * (len(lengths) - 2)
        s += 1 * sum(1 for p in pieces if len(p) == 2)
        s += 100 * sum(1 for p in pieces if p and p[0] == '0')
        return s

    def best_split_digits(s):
        print(f"DEBUG best_split_digits input: '{s}'")
        n = len(s)
        if 1 <= n <= 3 and in_range(s):
            print(f"  → early return: ['{s}']")
            return [s]
        parts = enum_partitions(n, 2, 4, 2, 4)
        print(f"  → partitions found: {parts}")
        best = None; best_score = 1e9
        for lens in parts:
            pos = 0
            pcs = []
            ok = True
            for L in lens:
                piece = s[pos:pos+L]; pos += L
                if in_range(piece) and not (len(piece) > 1 and piece[0] == '0'):
                    pcs.append(piece)
                else: ok = False; break
            if not ok: continue
            sc = score_partition(lens, pcs)
            print(f"  → lens={lens}, pcs={pcs}, score={sc}")
            if sc < best_score:
                best_score, best = sc, pcs
        print(f"  → returning: {best}")
        return best or []

    def bbox_for_span(i, j):
        char_w = total_w / max(1, len(text))
        sub_x0 = x0 + i * char_w
        sub_x1 = x0 + j * char_w
        cx_ = 0.5 * (sub_x0 + sub_x1)
        return (sub_x0, y0, sub_x1, y1), cx_

    text_norm = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    tokens = re.split(r'\s+', text_norm.strip()) if text_norm.strip() else []
    candidates = []
    cursor = 0

    def find_from(substr, start):
        k = text.find(substr, start)
        return k if k >= 0 else text.find(re.sub(r'[^0-9]', '', substr), start)

    any_found = False
    for tok in tokens if tokens else [text_norm]:
        if not tok:
            cursor = min(len(text), cursor + 1); continue
        digits = re.sub(r'[^0-9]', '', tok)
        if not digits:
            k = find_from(tok, cursor)
            if k >= 0: cursor = k + len(tok)
            continue

        pieces = best_split_digits(digits)
        k = find_from(tok, cursor)
        tok_len = len(tok) if k >= 0 else len(digits)
        if k < 0: k = cursor

        if pieces:
            any_found = True
            pos = 0
            for p in pieces:
                i = k + pos; j = i + len(p)
                sub_bbox, sub_cx = bbox_for_span(i, j)
                candidates.append({
                    'text': p, 'value': int(p),
                    'bbox': sub_bbox, 'cx': sub_cx, 'cy': (y0 + y1)/2.0,
                    'grouped': True
                })
                pos += len(p)
        cursor = max(cursor, k + tok_len)

    if not any_found:
        digits_all = re.sub(r'[^0-9]', '', text_norm)
        pieces = best_split_digits(digits_all)
        pos = 0
        for p in pieces:
            i = pos; j = pos + len(p)
            sub_bbox, sub_cx = bbox_for_span(i, j)
            candidates.append({
                'text': p, 'value': int(p),
                'bbox': sub_bbox, 'cx': sub_cx, 'cy': (y0 + y1)/2.0,
                'grouped': True
            })
            pos = j
    return candidates

# ----------------------------
# PDF helpers
# ----------------------------
def get_words(page):
    return [{"x0":w[0], "y0":w[1], "x1":w[2], "y1":w[3], "text":w[4]}
            for w in page.get_text("words")]

def get_segments(page):
    segs = []
    for dr in page.get_drawings():
        for it in dr.get("items", []):
            if it[0] == "l":
                (x0,y0), (x1,y1) = it[1], it[2]
                segs.append((x0,y0,x1,y1))
    return segs

def split_axes(segs, W, H, words, loosen: float = 2.0):
    H_TOL = 0.8
    V_TOL = 0.8
    horizontals = [ln for ln in segs if abs(ln[3] - ln[1]) < H_TOL]
    verticals   = [ln for ln in segs if abs(ln[2] - ln[0]) < V_TOL]

    baseline_y = None
    x_axis_seg = None
    if horizontals:
        lower = [ln for ln in horizontals if ln[1] > 0.55 * H]
        cand = max(lower if lower else horizontals, key=lambda ln: abs(ln[2] - ln[0]))
        x_axis_seg = cand
        baseline_y = cand[1]
    if baseline_y is None:
        baseline_y = 0.90 * H

    top_y = None
    tick100 = [w for w in words if w["text"].strip() == "100"]
    if tick100:
        # use bottom (y1) of the "100" text so the line coincides with the number’s baseline
        top_y = min(tick100, key=lambda w: w["y1"])["y1"]
    if top_y is None:
        if verticals:
            y_tops = [min(v[1], v[3]) for v in verticals]
            top_est = min(y_tops) if y_tops else 0.10 * H
            top_y = min(top_est, 0.10 * H)
        else:
            top_y = 0.10 * H

    LEFT_X_MAX     = 0.18 * W
    BASE_Y_TOL     = 4.5
    plot_h_est     = max(1.0, baseline_y - top_y)
    MIN_YAXIS_H    = max(0.25 * H, 0.50 * plot_h_est)

    y_axis_seg = None
    for (x0,y0,x1,y1) in verticals:
        x = 0.5*(x0+x1)
        y_top, y_bot = min(y0,y1), max(y0,y1)
        if x <= LEFT_X_MAX and abs(y_bot - baseline_y) <= BASE_Y_TOL and (y_bot - y_top) >= MIN_YAXIS_H:
            if (y_axis_seg is None) or ((y_axis_seg[3]-y_axis_seg[1]) < (y_bot - y_top)):
                y_axis_seg = (x0,y0,x1,y1)

    if plot_h_est > 0:
        base_clear = max(10.0, 0.10 * plot_h_est, 0.012 * H)
    else:
        base_clear = max(12.0, 0.012 * H)

    CLEAR         = base_clear / max(1.0, loosen)
    LEFT_OFFSET   = 6.0 / max(1.0, loosen)
    LEFT_MARGIN   = (0.09/ (max(1.0, loosen) ** 0.5)) * W
    RIGHT_MARGINX = 0.99 * W

    if y_axis_seg is not None:
        yx = 0.5 * (y_axis_seg[0] + y_axis_seg[2])
        X0_plot = max(LEFT_MARGIN, yx + LEFT_OFFSET)
    else:
        X0_plot = LEFT_MARGIN

    X1_plot = RIGHT_MARGINX

    plot_h_est = max(1.0, baseline_y - top_y)
    HEADROOM = max(0.10 * plot_h_est, 0.03 * H, 18.0)

    Y0_plot = max(0.01 * H, top_y - HEADROOM)
    Y1_plot = baseline_y - 0.25 * max(10.0, 0.10 * plot_h_est, 0.012 * H)

    if Y0_plot >= Y1_plot:
        mid = (Y0_plot + Y1_plot) / 2.0
        Y0_plot = max(0.05 * H, mid - 1.0)
        Y1_plot = min(baseline_y - 2.0, mid + 1.0)

    plot_box = (X0_plot, Y0_plot, X1_plot, Y1_plot)
    return {"baseline_y": baseline_y, "top_y": top_y,
            "x_axis_seg": x_axis_seg, "y_axis_seg": y_axis_seg,
            "plot_box":   plot_box}

def extract_compound(page) -> str:
    raw = page.get_text("words")
    norm = []
    for w in raw:
        if isinstance(w, dict):
            x0 = w.get("x0"); y0 = w.get("y0"); txt = w.get("text", "")
        else:
            if len(w) >= 5:
                x0, y0, txt = w[0], w[1], w[4]
            else:
                continue
        if not txt: continue
        txt = str(txt).strip()
        if not txt: continue
        norm.append((x0, y0, txt))

    if not norm:
        return ""

    lines = {}
    for x0, y0, txt in norm:
        key = round((y0 or 0.0) / 2.0)
        lines.setdefault(key, []).append((x0, txt))

    best_txt, best_len = "", -1
    for _, items in lines.items():
        items.sort(key=lambda t: t[0])
        line_text = " ".join(t[1] for t in items)
        line_text = line_text.replace("(mainlib)", "").strip()
        if re.search(r"[A-Za-z]", line_text):
            if len(line_text) > best_len:
                best_txt, best_len = line_text, len(line_text)
    return best_txt


def compute_from_vector_pdf_algo2(pdf_path: str) -> Tuple[Dict, float]:
    
    doc = fitz.open(pdf_path)
    page = doc[0]
    W, H = page.rect.width, page.rect.height

    words = get_words(page)
    segs  = get_segments(page)
    axes  = split_axes(segs, W, H, words)

    baseline_y = axes["baseline_y"]
    top_y      = axes["top_y"]
    X0p, Y0p, X1p, Y1p = axes["plot_box"]

    axis_max = get_axis_max(words, baseline_y, X0p, X1p)
    effective_mz_max = int(axis_max) if axis_max is not None else MZ_MAX

    compound = extract_compound(page).strip()

    # 1) Build peak lists from vectors
    V_TOL = 0.8
    verticals_all = [ln for ln in segs if abs(ln[2] - ln[0]) < V_TOL]

    def y_tip(seg): return min(seg[1], seg[3])

    # adaptive tick-height floor
    tick_heights = []
    for (x0, y0, x1, y1) in verticals_all:
        yt, yb = min(y0, y1), max(y0, y1)
        if abs(yb - baseline_y) <= 3.0:
            h = baseline_y - yt
            if 0 < h <= 22.0:
                tick_heights.append(h)
    tick_heights.sort()
    p90_tick = tick_heights[int(round(0.9 * (len(tick_heights) - 1)))] if tick_heights else 2.5
    MIN_PEAK_H_SMALL = max(p90_tick + 0.3, 1.0)

    # peaks = “real” peaks (tall enough)
    # peaks_any = ANY baseline-anchored vertical (even tiny) for fallback
    peaks, peaks_any = [], []
    for (x0, y0, x1, y1) in verticals_all:
        xt = 0.5*(x0+x1)
        yt = y_tip((x0, y0, x1, y1))
        yb = max(y0, y1)
        if abs(yb - baseline_y) <= 4.0:
            h = baseline_y - yt
            if h > 0.5:  # keep almost everything for fallback
                peaks_any.append({"x": xt, "y_tip": yt, "h": h})
            if h >= MIN_PEAK_H_SMALL:
                peaks.append({"x": xt, "y_tip": yt, "h": h})

    peaks.sort(key=lambda p: p["x"])
    peaks_any.sort(key=lambda p: p["x"])

    # normalization height
    if (top_y is not None) and (top_y < baseline_y):
        plot_h = baseline_y - top_y
    else:
        plot_h = max((p["h"] for p in peaks), default=None)
    if not plot_h or plot_h <= 0:
        return {"chemical_name": compound, "spectrum": [], "relative_abundance": []}

    for p in peaks:
        p["RA"] = max(0.0, min(100.0, (p["h"] / plot_h) * 100.0))
    # also precompute RA for tiny peaks (so fallback is cheap)
    for p in peaks_any:
        p["RA"] = max(0.0, min(100.0, (p["h"] / plot_h) * 100.0))

    # --- helper: RA from closest peak *below* a label (strict→relaxed→nearest) ---
    def ra_from_peak_below(label):
        lx, ly = label["x"], label["y"]
        lbl_w = (label["bbox"][2] - label["bbox"][0]) or 8.0
        lbl_h = (label["bbox"][3] - label["bbox"][1]) or 8.0

        # horizontal tolerance (slightly generous; same spirit as algo-1)
        XTOL_strict  = max(2.5, 0.0045 * W, 0.55 * lbl_w)
        XTOL_relaxed = XTOL_strict * 1.6

        def ok_gap(p, allow_looser=False):
            gap = p["y_tip"] - ly  # >0 → label above tip
            # strict
            lo = max(-1.0, -0.12 * lbl_h)
            hi = 5.0 * lbl_h
            if allow_looser:
                lo = max(-1.4, -0.16 * lbl_h)  # allow a tad more overshoot
                hi = 6.0 * lbl_h               # a bit farther above
            return (gap > lo) and (gap <= hi)

        # 1) try among real peaks (strict window)
        best = None; best_key = None
        for p in peaks:
            dx = abs(p["x"] - lx)
            if dx > XTOL_strict:
                continue
            if not ok_gap(p, allow_looser=False):
                continue
            key = (dx, -p["h"])
            if best_key is None or key < best_key:
                best_key, best = key, p
        if best is not None:
            return best["RA"]

        # 2) try among ANY peaks (relaxed window)
        for p in peaks_any:
            dx = abs(p["x"] - lx)
            if dx > XTOL_relaxed:
                continue
            if not ok_gap(p, allow_looser=True):
                continue
            key = (dx, -p["h"])
            if best_key is None or key < best_key:
                best_key, best = key, p
        if best is not None:
            return best["RA"]

        # 3) final fallback: nearest-in-x from peaks_any (ignore gap), within a wide band
        nearest = None; best_dx = None
        for p in peaks_any:
            dx = abs(p["x"] - lx)
            if best_dx is None or dx < best_dx:
                best_dx, nearest = dx, p
        return None if nearest is None else nearest["RA"]

    # 2) Extract labels (with splitting) inside plot window
    X0, X1 = X0p, X1p
    Y0, Y1 = Y0p, 369
    labels = []
    for w in words:
        t = w["text"].strip()
        b = (w["x0"], w["y0"], w["x1"], w["y1"])
        cx_val, cy_val = cx(b), cy(b)
        if not (X0 <= cx_val <= X1 and Y0 <= cy_val <= Y1):
            continue
        # direct numeric
        if NUM_RE.match(t):
            try:
                v = float(t)
                if MZ_MIN <= v <= MZ_MAX and len(t.replace('.','')) <= 3:
                    labels.append({"x": cx_val, "y": cy_val, "text": t, "bbox": b})
                    continue
            except:
                pass
        # split grouped
        for sc in split_grouped_numbers(t, b, MZ_MIN, effective_mz_max):
            if (X0 <= sc["cx"] <= X1) and (Y0 <= sc["cy"] <= Y1):
                labels.append({"x": sc["cx"], "y": sc["cy"], "text": sc["text"], "bbox": sc["bbox"]})

    # de-dup labels
    labels.sort(key=lambda d: (d["x"], d["y"], d["text"]))
    dedup = []
    def close(a, b, tol=1.4): return abs(a - b) <= tol
    for lab in labels:
        if not dedup:
            dedup.append(lab); continue
        last = dedup[-1]
        if (lab["text"] == last["text"]) and close(lab["x"], last["x"]) and close(lab["y"], last["y"], tol=3.0):
            continue
        dedup.append(lab)
    labels = dedup
    labels.sort(key=lambda l: l["x"])

    # 3) Monotone assignment (DP) peaks ↔ labels (left→right)
    with timer("core_s"):
        profiler = Profiler()
        profiler.start()
        t0 = time.perf_counter()
        nP, nL = len(peaks), len(labels)
        if nL == 0:
            return {"chemical_name": compound, "spectrum": [], "relative_abundance": []}

        match_penalty = 8.0
        gap_penalty   = 8.0
        INF = 1e12
        dp = [[INF]*(nL+1) for _ in range(nP+1)]
        bt = [[None]*(nL+1) for _ in range(nP+1)]
        dp[0][0] = 0.0


        for i in range(nP+1):
            for j in range(nL+1):
                if dp[i][j] >= INF:
                    continue
                if i < nP:
                    v = dp[i][j] + match_penalty
                    if v < dp[i+1][j]:
                        dp[i+1][j] = v; bt[i+1][j] = ("skipP", i, j)
                if j < nL:
                    v = dp[i][j] + gap_penalty
                    if v < dp[i][j+1]:
                        dp[i][j+1] = v; bt[i][j+1] = ("skipL", i, j)
                if i < nP and j < nL:
                    cost = abs(peaks[i]["x"] - labels[j]["x"])
                    v = dp[i][j] + cost
                    if v < dp[i+1][j+1]:
                        dp[i+1][j+1] = v; bt[i+1][j+1] = ("match", i, j)

        i, j = nP, nL
        pairs = []
        while i > 0 or j > 0:
            op = bt[i][j]
            if op is None:
                if j > 0:
                    pairs.append((j-1, None)); j -= 1
                else:
                    i -= 1
                continue
            typ, pi, pj = op
            if typ == "match":
                pairs.append((pj, pi)); i, j = pi, pj
            elif typ == "skipP":
                i, j = pi, pj
            elif typ == "skipL":
                pairs.append((pj, None)); i, j = pi, pj
        pairs.reverse()

    # 4) Emit in label x-order (use matched RA or fallback to peak-below RA)
        spectrum, relative_abundance = [], []
        for (j_idx, i_idx) in pairs:
            if j_idx is None or j_idx < 0 or j_idx >= nL:
                continue
            lbl = labels[j_idx]
            spectrum.append(lbl["text"])
            if i_idx is not None:
                relative_abundance.append(peaks[i_idx]["RA"])
            else:
                ra = ra_from_peak_below(lbl)
                relative_abundance.append(ra if ra is not None else 0.0)
        elapsed = time.perf_counter() - t0
        profiler.stop()
        result = {
        "chemical_name": compound,
        "spectrum": spectrum,
        "relative_abundance": relative_abundance
        }
        return result,elapsed


# ----------------------------
# Batch runner
# ----------------------------
def process_folder(folder_path: str, output_json: str = "Algo2_Parallel.json"):
    """
    Run compute_from_vector_pdf on all PDFs in a folder in parallel and save results into a JSON file.
    """
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    print(f"[info] found {len(pdf_files)} PDF files in {folder_path}")

    results = []
    core_times = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_pdf = {
            executor.submit(compute_from_vector_pdf_algo2, pdf_path): pdf_path
            for pdf_path in pdf_files
        }
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf_path = future_to_pdf[future]
            try:
                res,elapsed = future.result()
                res["file"] = os.path.basename(pdf_path)
                results.append(res)
                core_times.append(elapsed)
                print(f"[done] {pdf_path} in {elapsed: .3f}s")
            except Exception as e:
                print(f"[error] {pdf_path}: {e}")
                
    total_core = sum(core_times)
    avg_core = total_core / len(core_times) if core_times else 0.0
    print(f"[core timing] total across all files = {total_core}s, average per file = {avg_core}s")

    # Optionally sort results by filename if you want deterministic order/output
    results.sort(key=lambda r: r["file"])
    with open(output_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] results saved to {output_json}")

    return results

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    
    folder = "/Users/pramathkp/Desktop/FinalDataset"
    t0 = time.perf_counter()
    all_results = process_folder(folder, output_json="Algo2_Parallel.json")
    total_wall = time.perf_counter() - t0
    print(f"[WALL CLOCK] total run = {total_wall:.3f}s")
    print(len(all_results))
    
    for r in all_results:
        print("\n=== ", r["file"], " ===")
        print("Chemical:", r["chemical_name"])
        print("Spectrum:", r["spectrum"])
        print("RA     :", r["relative_abundance"])

#print(f"[TIMING] total = {METRICS['total_s']:.6f}s | core = {METRICS['core_s']:.6f}s")

