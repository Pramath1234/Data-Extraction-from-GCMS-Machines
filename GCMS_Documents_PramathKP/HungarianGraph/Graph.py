import fitz  # PyMuPDF
import re, glob, os, json, time
from typing import Dict, List, Tuple
import concurrent.futures
import time
from contextlib import contextmanager
from pyinstrument import Profiler

METRICS = {"total_s": 0.0, "core_s": 0.0}  # accumulates across files in that run

@contextmanager
def timer(bucket_key: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        METRICS[bucket_key] = METRICS.get(bucket_key, 0.0) + (time.perf_counter() - t0)

# ---------- optional Hungarian -------------
try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ---------- config ----------
NUM_RE = re.compile(r'^\d{1,4}(\.\d{1,2})?$')
MZ_MIN, MZ_MAX = 10, 1500

def cx(b): return 0.5*(b[0]+b[2])
def cy(b): return 0.5*(b[1]+b[3])

# ---------- text & vector helpers ----------

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
def get_words(page):
    # unify PyMuPDF words → dict
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

def extract_compound(page) -> str:
    # pick the longest alpha line on the page
    raw = page.get_text("words")
    norm = []
    for w in raw:
        if isinstance(w, dict):
            x0 = w.get("x0"); y0 = w.get("y0"); txt = w.get("text","")
        else:
            if len(w) >= 5:
                x0,y0,txt = w[0],w[1],w[4]
            else:
                continue
        if not txt: continue
        t = str(txt).strip()
        if not t: continue
        norm.append((x0,y0,t))
    if not norm: return ""
    lines = {}
    for x0, y0, t in norm:
        key = round((y0 or 0.0)/2.0)
        lines.setdefault(key, []).append((x0,t))
    best, blen = "", -1
    for _, items in lines.items():
        items.sort(key=lambda x: x[0])
        line = " ".join(s for _,s in items)
        line = line.replace("(mainlib)", "").strip()
        if re.search(r"[A-Za-z]", line) and len(line) > blen:
            best, blen = line, len(line)
    return best

# ---------- grouped-number splitter ----------
def split_grouped_numbers(text, bbox, mz_min=10, mz_max=1500):
    import re
    x0,y0,x1,y1 = bbox
    total_w = max(1e-6, x1-x0)

    def in_range(s):
        try:
            v = int(s); return mz_min <= v <= mz_max
        except: return False

    def enum_partitions(n, min_len=2, max_len=4, min_parts=2, max_parts=4):
        res=[]
        def dfs(rem,cur):
            if rem==0 and min_parts<=len(cur)<=max_parts:
                res.append(cur[:]); return
            if rem<=0 or len(cur)>=max_parts: return
            for L in range(min_len, max_len+1):
                if L<=rem:
                    cur.append(L); dfs(rem-L,cur); cur.pop()
        dfs(n,[]); return res

    def score_partition(lengths, pieces):
        s = 10 * sum((L-3)*(L-3) for L in lengths)   # prefer 3-digit
        s += 6 * (len(lengths)-2)                    # fewer parts
        s += 1 * sum(1 for p in pieces if len(p)==2) # slight 2-digit penalty
        s += 100* sum(1 for p in pieces if p and p[0]=='0')
        return s

    def best_split_digits(s):
        n = len(s)
        if 1 <= n <= 3 and in_range(s): return [s]
        best=None; best_sc=1e9
        for lens in enum_partitions(n,2,4,2,4):
            pos=0; pcs=[]; ok=True
            for L in lens:
                piece=s[pos:pos+L]; pos+=L
                if in_range(piece): pcs.append(piece)
                else: ok=False; break
            if not ok: continue
            sc = score_partition(lens, pcs)
            if sc < best_sc:
                best_sc, best = sc, pcs
        return best or []

    def bbox_for_span(i,j):
        char_w = total_w / max(1,len(text))
        sub_x0 = x0 + i*char_w
        sub_x1 = x0 + j*char_w
        return (sub_x0,y0,sub_x1,y1), 0.5*(sub_x0+sub_x1)

    # normalize exotic spaces
    text_norm = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    tokens = re.split(r'\s+', text_norm.strip()) if text_norm.strip() else []
    out=[]; cursor=0

    def find_from(substr,start):
        k = text.find(substr,start)
        return k if k>=0 else text.find(re.sub(r'[^0-9]','',substr), start)

    any_found=False
    for tok in tokens if tokens else [text_norm]:
        if not tok:
            cursor=min(len(text),cursor+1); continue
        digits = re.sub(r'[^0-9]','', tok)
        if not digits:
            k = find_from(tok, cursor)
            if k>=0: cursor = k+len(tok)
            continue
        pieces = best_split_digits(digits)
        k = find_from(tok, cursor)
        tok_len = len(tok) if k>=0 else len(digits)
        if k<0: k=cursor
        if pieces:
            any_found=True
            pos=0
            for p in pieces:
                i = k+pos; j = i+len(p)
                sub_bbox, sub_cx = bbox_for_span(i,j)
                out.append({
                    "text": p, "value": int(p),
                    "bbox": sub_bbox, "cx": sub_cx, "cy": (y0+y1)/2.0,
                    "grouped": True
                })
                pos += len(p)
        cursor = max(cursor, k+tok_len)

    if not any_found:
        digits_all = re.sub(r'[^0-9]','', text_norm)
        pieces = best_split_digits(digits_all)
        pos=0
        for p in pieces:
            i=pos; j=pos+len(p)
            sub_bbox, sub_cx = bbox_for_span(i,j)
            out.append({
                "text": p, "value": int(p),
                "bbox": sub_bbox, "cx": sub_cx, "cy": (y0+y1)/2.0,
                "grouped": True
            })
            pos=j
    return out

# ---------- axis detection & plot box ----------
def split_axes(segs, W, H, words):
    H_TOL = 0.8; V_TOL = 0.8
    horizontals = [ln for ln in segs if abs(ln[3]-ln[1]) < H_TOL]
    verticals   = [ln for ln in segs if abs(ln[2]-ln[0]) < V_TOL]

    # --- baseline (x-axis) ---
    baseline_y = None; x_axis_seg = None
    if horizontals:
        lower = [ln for ln in horizontals if ln[1] > 0.55 * H]
        cand  = max(lower if lower else horizontals, key=lambda ln: abs(ln[2]-ln[0]))
        baseline_y = cand[1]; x_axis_seg = cand
    if baseline_y is None:
        baseline_y = 0.90 * H

    # --- top reference ---
    top_y = None
    tick100 = [w for w in words if (w.get("text","").strip() == "100")]
    if tick100:
        top_y = min(tick100, key=lambda w: w["y1"])["y1"]
    if top_y is None:
        if verticals:
            y_tops = [min(v[1], v[3]) for v in verticals]
            top_y = min(y_tops) if y_tops else 0.10 * H
        else:
            top_y = 0.10 * H

    # --- detect y-axis (near left, tall, anchored at baseline) ---
    LEFT_X_MAX  = 0.18 * W
    BASE_Y_TOL  = 4.5
    plot_h_est  = max(1.0, baseline_y - top_y)
    MIN_YAXIS_H = max(0.25 * H, 0.50 * plot_h_est)

    y_axis_seg = None
    for (x0,y0,x1,y1) in verticals:
        x = 0.5*(x0+x1)
        y_top, y_bot = min(y0,y1), max(y0,y1)
        if x <= LEFT_X_MAX and abs(y_bot - baseline_y) <= BASE_Y_TOL and (y_bot - y_top) >= MIN_YAXIS_H:
            if (y_axis_seg is None) or ((y_axis_seg[3]-y_axis_seg[1]) < (y_bot - y_top)):
                y_axis_seg = (x0,y0,x1,y1)

    # --- plot box: start just right of y-axis if found ---
    LEFT_MARGIN = 0.09 * W
    LEFT_OFFSET = 6.0  # pts to clear tick text & axis stroke
    if y_axis_seg is not None:
        y_axis_x = 0.5 * (y_axis_seg[0] + y_axis_seg[2])
        X0_plot  = max(LEFT_MARGIN, y_axis_x + LEFT_OFFSET)
    else:
        y_axis_x = None
        X0_plot  = LEFT_MARGIN

    X1_plot = 0.99 * W
    HEADROOM = max(0.10*plot_h_est, 0.03*H, 18.0)
    Y0_plot = max(0.01 * H, top_y - HEADROOM)
    Y1_plot = 369.0 

    return {
        "baseline_y": baseline_y,
        "top_y": top_y,
        "x_axis_seg": x_axis_seg,
        "y_axis_seg": y_axis_seg,
        "y_axis_x": y_axis_x,
        "plot_box": (X0_plot, Y0_plot, X1_plot, Y1_plot)
    }


# ---------- consolidate verticals into unique peaks ----------
def consolidate_verticals_to_peaks(verticals, baseline_y, base_tol=4.0, x_tol=1.0, min_h=0.5):
    items=[]
    for (x0,y0,x1,y1) in verticals:
        yb = max(y0,y1); yt=min(y0,y1)
        if abs(yb - baseline_y) <= base_tol:
            h = baseline_y - yt
            if h > min_h:
                xm = 0.5*(x0+x1)
                items.append((xm, yt, h))
    items.sort(key=lambda t: t[0])
    groups, cur = [], []
    for rec in items:
        if not cur or abs(rec[0]-cur[-1][0]) <= x_tol:
            cur.append(rec)
        else:
            groups.append(cur); cur=[rec]
    if cur: groups.append(cur)

    peaks=[]
    for g in groups:
        xs=[r[0] for r in g]; yts=[r[1] for r in g]; hs=[r[2] for r in g]
        peaks.append({"x": sum(xs)/len(xs), "y_tip": min(yts), "h": max(hs)})
    peaks.sort(key=lambda p: p["x"])
    return peaks

# ---------- build label candidates (with splitting) ----------
def collect_labels(words, X0, Y0, X1, Y1,effective_mz_max):
    out=[]
    for w in words:
        t = (w["text"] or "").strip()
        b = (w["x0"], w["y0"], w["x1"], w["y1"])
        cx_, cy_ = cx(b), cy(b)
        if not (X0 <= cx_ <= X1 and Y0 <= cy_ <= Y1):  # Y1 is 369
            continue
        # direct numeric
        if NUM_RE.match(t):
            try:
                v=float(t)
                if MZ_MIN <= v <= MZ_MAX and len(t.replace('.','')) <= 3:
                    out.append({"text": t, "value": v, "bbox": b, "cx": cx_, "cy": cy_})
                    continue
            except: pass
        # grouped numbers
        for sc in split_grouped_numbers(t, b, MZ_MIN, effective_mz_max):
            if X0 <= sc["cx"] <= X1 and Y0 <= sc["cy"] <= Y1:
                out.append({"text": sc["text"], "value": float(sc["text"]),
                            "bbox": sc["bbox"], "cx": sc["cx"], "cy": sc["cy"]})
    # de-dup near-identical
    out.sort(key=lambda d:(d["text"], d["cx"], d["cy"]))
    dedup=[]
    def close(a,b,tol): return abs(a-b)<=tol
    for c in out:
        if not dedup: dedup.append(c); continue
        last=dedup[-1]
        if c["text"]==last["text"] and close(c["cx"],last["cx"],1.4) and close(c["cy"],last["cy"],3.0):
            continue
        dedup.append(c)
    dedup.sort(key=lambda d: d["cx"])
    return dedup

# ---------- cost matrix (labels ↔ peaks) ----------
def build_cost_matrix(labels, peaks, W,
                      invalid_cost=1e6, unmatched_cost=200.0, add_dummies=True):
    rows=[]
    for lab in labels:
        x0,y0,x1,y1 = lab["bbox"]
        lbl_w = (x1-x0) or 8.0
        lbl_h = (y1-y0) or 8.0
        XTOL   = max(1.8, 0.0035*W, 0.45*lbl_w) * 1.25
        GAP_LO = max(-0.4, -0.08*lbl_h)
        GAP_HI = 5.0*lbl_h
        GAP_T  = 0.35*lbl_h

        row=[]
        for pk in peaks:
            dx  = abs(lab["cx"] - pk["x"])
            if dx > XTOL:
                row.append(invalid_cost); continue
            gap = pk["y_tip"] - lab["cy"]     # >0 => label above tip
            if not (GAP_LO < gap <= GAP_HI):
                row.append(invalid_cost); continue
            cost = dx + 0.35*abs(gap - GAP_T) + 8.0*(1.0/(1.0+pk["h"]))
            row.append(cost)
        rows.append(row)

    nL = len(labels); nP = len(peaks)
    if add_dummies:
        nD = max(nL - nP, 0) or nL  # safe: one dummy per label
        for i in range(nL):
            rows[i].extend([unmatched_cost]*nD)
        peak_cols = nP
    else:
        peak_cols = nP

    if _HAVE_SCIPY:
        C = np.array(rows, dtype=float)
    else:
        # poor man's 2D array
        C = [list(r) for r in rows]
        C.shape = (len(C), len(C[0]) if C else 0)  # just to mimic
    return C, nL, peak_cols

def hungarian_assign(C):
    if _HAVE_SCIPY:
        return linear_sum_assignment(C)

    # Greedy fallback (if SciPy isn't available)
    nL = C.shape[0] if _HAVE_SCIPY else len(C)
    nC = C.shape[1] if _HAVE_SCIPY else len(C[0]) if nL else 0
    triples=[]
    for i in range(nL):
        for j in range(nC):
            cost = C[i,j] if _HAVE_SCIPY else C[i][j]
            triples.append((cost,i,j))
    triples.sort(key=lambda t:t[0])
    used_r=set(); used_c=set(); row_ind=[]; col_ind=[]
    for cost,i,j in triples:
        if i in used_r or j in used_c: continue
        row_ind.append(i); col_ind.append(j)
        used_r.add(i); used_c.add(j)
        if len(row_ind)==nL: break
    return row_ind, col_ind

# ---------- RA helpers ----------
def ra_from_height(h, baseline_y, top_y):
    if h is None or h <= 0: return 0.0
    if (top_y is None) or not (top_y < baseline_y):
        # fallback: scale by tallest peak height later (caller can rescale)
        return None
    plot_h = baseline_y - top_y
    if plot_h <= 0: return 0.0
    return max(0.0, min(100.0, (h/plot_h)*100.0))

# ---------- main: compute from a single PDF ----------
def compute_from_vector_pdf_algo3(pdf_path: str, page_index: int = 0) -> Tuple[Dict,float]:
    
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    W,H = page.rect.width, page.rect.height

    words = get_words(page)
    segs  = get_segments(page)
    axes  = split_axes(segs, W, H, words)
    baseline_y = axes["baseline_y"]
    top_y      = axes["top_y"]
    X0, Y0, X1, Y1 = axes["plot_box"]

    axis_max = get_axis_max(words, baseline_y, X0, X1)
    effective_mz_max = int(axis_max) if axis_max is not None else MZ_MAX

    # classify segments
    V_TOL = 0.8
    verticals_all = [ln for ln in segs if abs(ln[2]-ln[0]) < V_TOL]

    # peaks
    peaks = consolidate_verticals_to_peaks(verticals_all, baseline_y,
                                           base_tol=4.0, x_tol=1.0, min_h=0.5)
    # labels
    labels = collect_labels(words, X0, Y0, X1, Y1, effective_mz_max)

    # nothing found → empty
    if not labels or not peaks:
        return {"chemical_name": extract_compound(page),
                "spectrum": [], "relative_abundance": []}

    # cost matrix + Hungarian
    with timer("core_s"):
        profiler = Profiler () 
        profiler.start()
        t0 = time.perf_counter()
        C, nL, peak_cols = build_cost_matrix(labels, peaks, W,
                                         invalid_cost=1e6, unmatched_cost=200.0, add_dummies=True)
        row_ind, col_ind = hungarian_assign(C)

    # map assignments
    mapping = {i: None for i in range(nL)}
    for r,c in zip(row_ind, col_ind):
        if c >= peak_cols:
            mapping[r] = None
        else:
            edge_cost = (C[r,c] if _HAVE_SCIPY else C[r][c])
            if edge_cost >= 1e6 * 0.5:
                mapping[r] = None
            else:
                mapping[r] = c

    # compute RA
    # if no usable top_y, re-scale by tallest peak height as fallback
    tallest = max((p["h"] for p in peaks), default=0.0)
    spectrum=[]; ra=[]
    for i, lab in enumerate(labels):
        pk_idx = mapping.get(i)
        if pk_idx is None:
            # unmatched → set RA to 0.0
            spectrum.append(lab["text"]); ra.append(0.0); continue
        h = peaks[pk_idx]["h"]
        r = ra_from_height(h, baseline_y, top_y)
        if r is None:
            r = 0.0 if tallest<=0 else max(0.0, min(100.0, (h/tallest)*100.0))
        spectrum.append(lab["text"]); ra.append(r)
    elapsed = time.perf_counter() - t0
    profiler.stop()
    out_html = os.path.splitext(os.path.basename(pdf_path))[0] + f"_profile_{os.getpid()}.html"
    with open(out_html, "w") as f:
        f.write(profiler.output_html())
    result= {
        "chemical_name": extract_compound(page).strip(),
        "spectrum": spectrum,
        "relative_abundance": ra
    }
    return result,elapsed

# ---------- batch ----------
def process_folder(folder_path: str, output_json: str = "Algo3_Parallel.json"):
    """
    Run compute_from_vector_pdf on all PDFs in a folder in parallel and save results into a JSON file.
    """
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    print(f"[info] found {len(pdf_files)} PDF files in {folder_path}")

    results = []
    core_times = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_pdf = {
            executor.submit(compute_from_vector_pdf_algo3, pdf_path): pdf_path
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
    print(f"[core timing] total across all files = {total_core:}s, average per file = {avg_core:}s")

    # Optionally sort results by filename if you want deterministic order/output
    results.sort(key=lambda r: r["file"])
    with open(output_json, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] results saved to {output_json}")

    return results

# -------------- CLI --------------
if __name__ == "__main__":
    
    folder = "/Users/pramathkp/Desktop/FinalDataset"
    t0 = time.perf_counter()
    all_results = process_folder(folder, output_json="Graph.json")
    total_wall = time.perf_counter() - t0
    print(f"[WALL CLOCK] total run = {total_wall:.3f}s")

    for r in all_results:
        print("\n=== ", r["file"], " ===")
        print("Chemical:", r["chemical_name"])
        print("Spectrum:", r["spectrum"])
        print("RA     :", r["relative_abundance"])

#print(f"[TIMING] total = {METRICS['total_s']:.3f}s | core = {METRICS['core_s']:.3f}s")

