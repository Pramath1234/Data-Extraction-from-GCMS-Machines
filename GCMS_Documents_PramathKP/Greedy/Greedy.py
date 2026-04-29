# pip install pymupdf
# pip install pdfplumber
import fitz  # PyMuPDF
import re, json
from typing import Dict, List, Tuple, Optional, Any
import pdfplumber
import glob
import os
import json
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

def split_grouped_numbers(text, bbox, mz_min=10, mz_max=1500):
    """
    Robust splitter for concatenated / mixed-space numeric labels.
    Prefers 3-digit chunks and fewer parts. Falls back gracefully.
    """
    import re
    x0, y0, x1, y1 = bbox
    total_w = max(1e-6, x1 - x0)

    # --- small helpers -------------------------------------------------------
    def in_range(s):
        try:
            v = int(s)
            return mz_min <= v <= mz_max
        except:
            return False

    # enumerate all partitions of n into 2..max_parts parts, each length 2..4
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

    # score a partition: prefer 3-digit, fewer parts
    def score_partition(lengths, pieces):
        # strong pull toward 3-digit
        s = 10 * sum((L - 3) * (L - 3) for L in lengths)
        # slight penalty for more parts
        s += 6 * (len(lengths) - 2)
        # tiny penalty for 2-digit pieces (still allowed)
        s += 1 * sum(1 for p in pieces if len(p) == 2)
        # big penalty for leading zeros (unlikely)
        s += 100 * sum(1 for p in pieces if p and p[0] == '0')
        return s

    # try to split a digits-only string into best-scoring valid pieces
    def best_split_digits(s):
        n = len(s)
        # if it's already a single valid label, keep as-is
        if 1 <= n <= 3 and in_range(s):
            return [s]

        # enumerate and pick lowest-score valid
        parts = enum_partitions(n, 2, 4, 2, 4)
        best = None; best_score = 1e9
        for lens in parts:
            pos = 0
            pcs = []
            ok = True
            for L in lens:
                piece = s[pos:pos+L]; pos += L
                if in_range(piece):
                    pcs.append(piece)
                else:
                    ok = False; break
            if not ok: 
                continue
            sc = score_partition(lens, pcs)
            if sc < best_score:
                best_score, best = sc, pcs
        return best or []

    # proportional bbox for substring indices in *text* (not token)
    def bbox_for_span(i, j):
        char_w = total_w / max(1, len(text))
        sub_x0 = x0 + i * char_w
        sub_x1 = x0 + j * char_w
        cx_ = 0.5 * (sub_x0 + sub_x1)
        return (sub_x0, y0, sub_x1, y1), cx_

    # normalize exotic spaces to plain space
    text_norm = re.sub(r'[\u00A0\u2000-\u200B\u202F\u205F\u3000]', ' ', text)
    tokens = re.split(r'\s+', text_norm.strip()) if text_norm.strip() else []
    candidates = []
    cursor = 0  # index into original *text* for bbox mapping

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
        # map token span in original text
        k = find_from(tok, cursor)
        tok_len = len(tok) if k >= 0 else len(digits)
        if k < 0: k = cursor  # best effort

        if pieces:
            any_found = True
            # spread pieces sequentially within this token span
            pos = 0
            for p in pieces:
                i = k + pos
                j = i + len(p)
                sub_bbox, sub_cx = bbox_for_span(i, j)
                candidates.append({
                    'text': p,
                    'value': int(p),
                    'bbox': sub_bbox,
                    'cx': sub_cx,
                    'cy': (y0 + y1)/2.0,
                    'grouped': True
                })
                pos += len(p)

        cursor = max(cursor, k + tok_len)

    # last-resort: whole string without spaces
    if not any_found:
        digits_all = re.sub(r'[^0-9]', '', text_norm)
        pieces = best_split_digits(digits_all)
        pos = 0
        for p in pieces:
            i = pos; j = pos + len(p)
            sub_bbox, sub_cx = bbox_for_span(i, j)
            candidates.append({
                'text': p,
                'value': int(p),
                'bbox': sub_bbox,
                'cx': sub_cx,
                'cy': (y0 + y1)/2.0,
                'grouped': True
            })
            pos = j

    return candidates
    

def get_candidates_with_splitting(words, X0, Y0, X1, Y1, baseline_y, NUM_RE, MZ_MIN, MZ_MAX):
    candidates = []
    
    for w in words:
        t = w["text"].strip()
        b = (w["x0"], w["y0"], w["x1"], w["y1"])
        cx_val, cy_val = cx(b), cy(b)
        
        # Skip if outside bounds
        if not (X0 <= cx_val <= X1 and Y0 <= cy_val <= Y1):
            continue
        if (baseline_y - cy_val) < 0.01:
            continue
        
        # Try normal single number first
        if NUM_RE.match(t):
            try:
                v = float(t)
                if MZ_MIN <= v <= MZ_MAX and len(t.replace('.',',')) <= 3:
                    candidates.append({
                        "text": t, "bbox": b, "cx": cx_val, "cy": cy_val, "value": v
                    })
                    continue
            except:
                pass
        
        # If not a normal number, try splitting grouped numbers
        split_candidates = split_grouped_numbers(t, b, MZ_MIN, MZ_MAX)
        if split_candidates:
            print(f"Split '{t}' into: {[c['text'] for c in split_candidates]}")
            # Check if split candidates are in bounds
            for sc in split_candidates:
                if (X0 <= sc['cx'] <= X1 and Y0 <= sc['cy'] <= Y1 and 
                    (baseline_y - sc['cy']) >= 0.01):
                    candidates.append(sc)
    
    return candidates


def try_alternative_extractions(pdf_path, page_index=0):
    print("\n=== ALTERNATIVE TEXT EXTRACTION METHODS ===")
    
    doc = fitz.open(pdf_path)
    page = doc[page_index]
    
    # Method 1: Different PyMuPDF text extraction modes
    print("1. PyMuPDF 'dict' mode:")
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if "lines" in block:
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text and any(c.isdigit() for c in text):
                        bbox = span.get("bbox", [0,0,0,0])
                        print(f"  '{text}' at ({(bbox[0]+bbox[2])/2:.1f}, {(bbox[1]+bbox[3])/2:.1f})")
    
    # Method 2: PyMuPDF 'rawdict' mode (more detailed)
    print("\n2. PyMuPDF 'rawdict' mode:")
    try:
        raw_dict = page.get_text("rawdict")
        for block in raw_dict.get("blocks", []):
            if "lines" in block:
                for line in block["lines"]:
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text and any(c.isdigit() for c in text):
                            bbox = span.get("bbox", [0,0,0,0])
                            print(f"  '{text}' at ({(bbox[0]+bbox[2])/2:.1f}, {(bbox[1]+bbox[3])/2:.1f})")
    except:
        print("  rawdict mode not available")
    
    # Method 3: PyMuPDF 'html' mode (sometimes catches different text)
    print("\n3. PyMuPDF 'html' mode:")
    try:
        html_text = page.get_text("html")
        # Simple regex to find numbers in HTML
        import re
        numbers = re.findall(r'>\s*(\d{2,4})\s*<', html_text)
        print(f"  Found numbers in HTML: {numbers}")
    except:
        print("  HTML extraction failed")
    
    # Method 4: Check if there are any text annotations or form fields
    print("\n4. Annotations and widgets:")
    annots = page.annots()
    widgets = page.widgets()
    print(f"  Annotations: {len(annots)}")
    print(f"  Widgets: {len(widgets)}")
    
    for widget in widgets:
        if widget.field_value and any(c.isdigit() for c in str(widget.field_value)):
            print(f"  Widget text: '{widget.field_value}' at {widget.rect}")
    
    # Method 5: Different PDFPlumber extraction settings
    print("\n5. PDFPlumber with different settings:")
    
    with pdfplumber.open(pdf_path) as pdf:
        pg = pdf.pages[page_index]
        
        # Try extracting with different settings
        chars = pg.chars
        print(f"  Raw characters found: {len(chars)}")
        
        missing_numbers = ["139", "152", "168", "181", "194"]
        for char in chars:
            text = char.get('text', '')
            if text in missing_numbers or (len(text) >= 2 and text.isdigit()):
                print(f"  Character '{text}' at ({char.get('x0', 0):.1f}, {char.get('y0', 0):.1f})")

def get_words(page):
    # [x0,y0,x1,y1,text,block,line,word_no] -> unify to dicts
    return [{"x0":w[0], "y0":w[1], "x1":w[2], "y1":w[3], "text":w[4]}
            for w in page.get_text("words")]

def get_segments(page):
    # flatten vector path items into simple line segments
    segs = []
    for dr in page.get_drawings():
        for it in dr.get("items", []):
            if it[0] == "l":
                (x0,y0), (x1,y1) = it[1], it[2]
                segs.append((x0,y0,x1,y1))
            
    return segs

def draw_debug_on_pdf_copy(
    pdf_path: str, out_pdf: str,
    X0: float, Y0: float, X1: float, Y1: float,
    baseline_y: float, top_y: float,
    candidates: list = None,
    verticals_strict: list = None,
    verticals_loose: list = None
):
    doc = fitz.open(pdf_path)
    page = doc[0]
    shape = page.new_shape()

    # plot box (red)
    shape.draw_rect(fitz.Rect(X0, Y0, X1, Y1))
    shape.finish(color=(1,0,0), fill=None, width=1.5)

    # baseline (cyan)
    shape.draw_line(fitz.Point(0, baseline_y), fitz.Point(page.rect.width, baseline_y))
    shape.finish(color=(0,0.7,1), width=1.2)

    # top_y (green)
    if top_y is not None:
        shape.draw_line(fitz.Point(0, top_y), fitz.Point(page.rect.width, top_y))
        shape.finish(color=(0,1,0.5), width=1.2)

    # verticals
    if verticals_strict:
        for (x0,y0,x1,y1) in verticals_strict:
            xm = 0.5*(x0+x1)
            shape.draw_line(fitz.Point(xm, y0), fitz.Point(xm, y1))
        shape.finish(color=(1,0,1), width=1.0)

    if verticals_loose:
        for (x0,y0,x1,y1) in verticals_loose:
            xm = 0.5*(x0+x1)
            shape.draw_line(fitz.Point(xm, y0), fitz.Point(xm, y1))
        shape.finish(color=(1,0.55,0), width=1.0)

    # candidates (blue circles)
    if candidates:
        for c in candidates:
            cx, cy = c["cx"], c["cy"]
            r = 2.2
            shape.draw_circle(fitz.Point(cx, cy), r)
        shape.finish(color=(0,0.45,1), fill=(0,0.45,1), width=0.5)

    shape.commit()  # commit all drawings to the page
    doc.save(out_pdf)
    print(f"[debug] wrote {out_pdf}")


# --- NEW: collect near-baseline numbers with pdfplumber --------------------
def collect_near_baseline_numbers_pdfplumber(
    pdf_path: str,
    page_index: int,
    mz_min: float,
    mz_max: float,
    num_re: re.Pattern,
    band_pts: float,
    x_margin_frac: float = 0.06,
    right_margin_frac: float = 0.99,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        pg = pdf.pages[page_index]
        Wp, Hp = pg.width, pg.height

        # baseline via longest lower horizontal (single pass, same decision rule)
        lines = pg.lines or []
        max_len_all = -1.0
        max_len_lower = -1.0
        y_all = None
        y_lower = None
        half_h = 0.55 * Hp

        for ln in lines:
            y0 = ln.get("y0", 0.0)
            y1 = ln.get("y1", 0.0)
            # horizontal check
            if abs(y1 - y0) >= 0.8:
                continue
            x0 = ln.get("x0", 0.0)
            x1 = ln.get("x1", 0.0)
            length = abs(x1 - x0)

            if length > max_len_all:
                max_len_all = length
                y_all = y0

            if y0 > half_h and length > max_len_lower:
                max_len_lower = length
                y_lower = y0

        baseline_y = y_lower if y_lower is not None else (y_all if y_all is not None else 0.90 * Hp)

        # thin band just above baseline
        X0 = x_margin_frac * Wp
        X1 = right_margin_frac * Wp
        Y1 = baseline_y - 1.0
        Y0 = max(0.01 * Hp, Y1 - band_pts)

        # cache for speed
        append = out.append
        in_range = mz_min <= mz_max  # assume true, keeps branch predictor happy

        for w in pg.extract_words():
            t = (w.get("text") or "").strip()
            if not t or not num_re.match(t):
                continue
            # parse number
            try:
                v = float(t)
            except Exception:
                continue
            if not (mz_min <= v <= mz_max):
                continue

            x0 = w.get("x0", 0.0)
            x1 = w.get("x1", 0.0)
            y0 = w.get("top", 0.0)
            y1 = w.get("bottom", 0.0)

            cx_ = 0.5 * (x0 + x1)
            cy_ = 0.5 * (y0 + y1)

            # within band and to the left-right bounds
            if cx_ < X0 or cx_ > X1 or cy_ < Y0 or cy_ > Y1:
                continue
            # stay above x-axis
            if cy_ >= baseline_y:
                continue

            append({"text": t, "value": v, "bbox": (x0, y0, x1, y1), "cx": cx_, "cy": cy_})

    return out

# --- NEW: merge without dupes ----------------------------------------------
def merge_candidates(primary, extra, x_dupe_tol: float = 1.4, y_dupe_tol: float = 3.0):
    merged = primary[:]
    for e in extra:
        ex, ey, et = e["cx"], e["cy"], e["text"]
        dup = False
        for p in primary:
            if p["text"] != et:
                continue
            if abs(p["cx"] - ex) <= x_dupe_tol and abs(p["cy"] - ey) <= y_dupe_tol:
                dup = True
                break
        if not dup:
            merged.append(e)
    return merged


# --- axis detection helpers -----------------------------------------------

def split_axes(segs, W, H, words, loosen: float = 2.0):
    """
    Detect x-axis (baseline) and y-axis, then return a *looser* plot box
    to clamp candidate labels. Increase `loosen` to make it even looser (e.g., 2.0).
    """
    H_TOL = 0.8
    V_TOL = 0.8
    horizontals = [ln for ln in segs if abs(ln[3] - ln[1]) < H_TOL]
    verticals   = [ln for ln in segs if abs(ln[2] - ln[0]) < V_TOL]

    # --- X-AXIS (baseline) ---
    baseline_y = None
    x_axis_seg = None
    if horizontals:
        lower = [ln for ln in horizontals if ln[1] > 0.55 * H]
        cand = max(lower if lower else horizontals, key=lambda ln: abs(ln[2] - ln[0]))
        x_axis_seg = cand
        baseline_y = cand[1]
    if baseline_y is None:
        baseline_y = 0.90 * H

    # --- TOP reference (or '100' tick) ---
    top_y = None
    
    
    tick100 = [w for w in words if w["text"].strip() == "100"]
    if tick100:
        top_y = min(tick100, key=lambda w: w["y1"])["y1"]
    if top_y is None:
        # fallback estimate from verticals or 10% page top
        if verticals:
            y_tops = [min(v[1], v[3]) for v in verticals]
            top_est = min(y_tops) if y_tops else 0.10 * H
            top_y = min(top_est, 0.10 * H)
        else:
            top_y = 0.10 * H

    # --- Y-AXIS (near left, touches baseline, tall) ---
    LEFT_X_MAX     = 0.18 * W
    BASE_Y_TOL     = 4.5
    plot_h_est     = max(1.0, baseline_y - top_y)
    MIN_YAXIS_H    = max(0.25 * H, 0.50 * plot_h_est)  # slightly easier than before

    y_axis_seg = None
    for (x0,y0,x1,y1) in verticals:
        x = 0.5*(x0+x1)
        y_top, y_bot = min(y0,y1), max(y0,y1)
        if x <= LEFT_X_MAX and abs(y_bot - baseline_y) <= BASE_Y_TOL and (y_bot - y_top) >= MIN_YAXIS_H:
            if (y_axis_seg is None) or ((y_axis_seg[3]-y_axis_seg[1]) < (y_bot - y_top)):
                y_axis_seg = (x0,y0,x1,y1)

    # --- Looser plot box ----------------------------------------------------
    # base clearance above baseline (smaller = looser)
    if plot_h_est > 0:
        base_clear = max(10.0, 0.10 * plot_h_est, 0.012 * H)
    else:
        base_clear = max(12.0, 0.012 * H)

    # make it looser by dividing the clearance and offsets by `loosen`
    CLEAR         = base_clear / max(1.0, loosen)
    LEFT_OFFSET   = 6.0 / max(1.0, loosen)          # pts to the right of y-axis
    LEFT_MARGIN   = (0.09/ (max(1.0, loosen) ** 0.5)) * W  # shrink left margin a bit
    RIGHT_MARGINX = 0.99 * W                        # allow closer to right edge

    if y_axis_seg is not None:
        yx = 0.5 * (y_axis_seg[0] + y_axis_seg[2])
        X0_plot = max(LEFT_MARGIN, yx + LEFT_OFFSET)
    else:
        X0_plot = LEFT_MARGIN

    X1_plot = RIGHT_MARGINX

    # loosen vertical bounds: nearer to top and nearer to baseline
         # allow labels above the '100' tick (smaller y = higher)
    plot_h_est = max(1.0, baseline_y - top_y)  # already computed earlier
    HEADROOM = max(0.10 * plot_h_est, 0.03 * H, 18.0)  # go noticeably above top_y

    Y0_plot = max(0.01 * H, top_y - HEADROOM)   # NOTE the minus → higher than top_y
    Y1_plot = baseline_y - 0.25 * max(10.0, 0.10 * plot_h_est, 0.012 * H)
       # was 1.00*CLEAR → 0.60*CLEAR

    # sanity: keep Y order correct
    if Y0_plot >= Y1_plot:
        mid = (Y0_plot + Y1_plot) / 2.0
        Y0_plot = max(0.05 * H, mid - 1.0)
        Y1_plot = min(baseline_y - 2.0, mid + 1.0)

    plot_box = (X0_plot, Y0_plot, X1_plot, Y1_plot)
    return {
        "baseline_y": baseline_y,
        "top_y": top_y,
        "x_axis_seg": x_axis_seg,
        "y_axis_seg": y_axis_seg,
        "plot_box":   plot_box
    }





def extract_compound(page) -> str:
    
    raw = page.get_text("words")
    norm = []  # (x0, y0, text)

    for w in raw:
        if isinstance(w, dict):
            x0 = w.get("x0"); y0 = w.get("y0"); x1 = w.get("x1"); y1 = w.get("y1")
            txt = w.get("text", "")
        else:
            # tuple/list: accept both 5-tuple and 8-tuple forms
            if len(w) >= 5:
                x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], w[4]
            else:
                continue
        if txt is None:
            continue
        txt = str(txt).strip()
        if not txt:
            continue
        norm.append((x0, y0, txt))

    if not norm:
        return ""

    # group into lines by coarse y0 bin, then take the longest line containing letters
    lines = {}
    for x0, y0, txt in norm:
        key = round((y0 or 0.0) / 2.0)
        lines.setdefault(key, []).append((x0, txt))

    best_txt, best_len = "", -1
    for _, items in lines.items():
        items.sort(key=lambda t: t[0])           # left→right
        line_text = " ".join(t[1] for t in items)
        line_text = line_text.replace("(mainlib)", "").strip()
        if re.search(r"[A-Za-z]", line_text):    # must contain letters
            if len(line_text) > best_len:
                best_txt, best_len = line_text, len(line_text)

    return best_txt
    # choose the longest alpha line on the page (works well for names)


def compute_from_vector_pdf(pdf_path: str) -> Tuple[Dict, float]:
    
    
    # t0 = time.perf_counter()
    doc = fitz.open(pdf_path)
    page = doc[0]
    W,H = page.rect.width, page.rect.height

    words = get_words(page)
    segs = get_segments(page)
    axes = split_axes(segs, W, H, words)

    baseline_y = axes["baseline_y"]
    top_y      = axes["top_y"]       # may be None if not found
    X0p, Y0p, X1p, Y1p = axes["plot_box"]

    axis_max = get_axis_max(words, baseline_y, X0p, X1p)
    effective_mz_max = int(axis_max) if axis_max is not None else MZ_MAX

    #print(f"[debug] baseline_y = {baseline_y:.1f}")
    #print(f"[debug] top_y line drawn at y = {top_y:.1f} (100% reference tick)")

    verticals_all = [ln for ln in segs if abs(ln[2]-ln[0]) < 0.8]      # near-vertical

    # ---- compound name ----
    compound = extract_compound(page)

    # ---- classify verticals anchored on the baseline (real peaks) ----
    verticals_strict = []
    verticals_loose  = []
    for (x0,y0,x1,y1) in verticals_all:
        y_top, y_bot = min(y0,y1), max(y0,y1)
        anchored = abs(y_bot - baseline_y) <= 3.0
        height   = baseline_y - y_top
        if anchored and (y_top < baseline_y - 6.0):
            verticals_strict.append((x0,y0,x1,y1))
        if anchored and height >= 2.0:
            verticals_loose.append((x0,y0,x1,y1))

    # ---- numeric label candidate search window (unchanged) ----
    X0, X1 = X0p, X1p
    Y0, Y1 = Y0p, 369  

    # ---- candidates (with grouped-number splitting) ----
    candidates = get_candidates_with_splitting(words, X0, Y0, X1, Y1, baseline_y, NUM_RE, MZ_MIN, effective_mz_max)

    # ---- rescue near-baseline labels with pdfplumber, then merge ----
    extra_near = collect_near_baseline_numbers_pdfplumber(
        pdf_path=pdf_path,
        page_index=0,
        mz_min=MZ_MIN, mz_max=MZ_MAX,
        num_re=NUM_RE,
        band_pts=20.0,
        x_margin_frac=0.06, right_margin_frac=0.99,
    )
    candidates = merge_candidates(candidates, extra_near)

    # ---- accept only labels aligned to a peak and above its tip ----
    with timer("core_s"):
        profiler = Profiler()
        profiler.start()
        t0 = time.perf_counter()
        def nearest_in(lines, x_target: float, x_tol: float):
            best=None; best_dx=1e9
            for (x0,y0,x1,y1) in lines:
                x = 0.5*(x0+x1); dx = abs(x - x_target)
                if dx <= x_tol and dx < best_dx:
                    best, best_dx = (x0,y0,x1,y1), dx
            return best

        def peak_tip_y(seg):  # smaller y = higher
            return min(seg[1], seg[3])

        def peak_height(seg):  # relative to baseline
            return baseline_y - peak_tip_y(seg)

    # adaptive tick-height → small-peak floor
        tick_heights = []
        for (x0,y0,x1,y1) in verticals_all:
            yt, yb = min(y0,y1), max(y0,y1)
            if abs(yb - baseline_y) <= 3.0:
                h = baseline_y - yt
                if 0 < h <= 22.0:
                    tick_heights.append(h)
        tick_heights.sort()
        p90_tick = tick_heights[int(round(0.9*(len(tick_heights)-1)))] if tick_heights else 2.5
        MIN_PEAK_H_SMALL = max(p90_tick + 0.6, 2.0)

        vals, abund = [], []
        plot_h = (baseline_y - top_y) if (top_y is not None and top_y < baseline_y) else None

        for n in candidates:
        # adaptive x tolerance (page + label width)
            lbl_w = (n["bbox"][2] - n["bbox"][0]) or 8.0
            XTOL  = max(1.8, 0.0035 * W, 0.45 * lbl_w)

        # try strict peaks first
            pk = nearest_in(verticals_strict, n["cx"], XTOL)

        # fallback: loose (tiny but real) peaks
            if pk is None:
                pk = nearest_in(verticals_loose, n["cx"], XTOL * 1.25)

        # last resort: any near-vertical, but require height > tick floor
            if pk is None:
                best=None; best_dx=1e9
                for (x0,y0,x1,y1) in verticals_all:
                    xmid = 0.5*(x0+x1); dx = abs(xmid - n["cx"])
                    if dx > XTOL * 1.5: 
                        continue
                    yt, yb = min(y0,y1), max(y0,y1)
                    if abs(yb - baseline_y) > 4.0:  # must be near baseline
                        continue
                    h = baseline_y - yt
                    if h < MIN_PEAK_H_SMALL:        # taller than ticks
                        continue
                    if dx < best_dx:
                        best, best_dx = (x0,y0,x1,y1), dx
                pk = best
        
            if pk is None:
                best = None; best_h = 0.0
                for (x0,y0,x1,y1) in verticals_all:
                    xmid = 0.5*(x0+x1)
                    dx = abs(xmid - n["cx"])
                    if dx > XTOL * 2.0:
                        continue
                    yb = max(y0,y1); yt = min(y0,y1)
                    if abs(yb - baseline_y) > 4.0:
                        continue
                    h = baseline_y - yt
                    if h > best_h:
                        best_h = h
                        best = (x0,y0,x1,y1)
                pk = best

        # --- NEW FALLBACK: if verticals are negligible / no match, keep with 0 ---
            if pk is None:
                vals.append((n["cx"], n["text"]))
                abund.append((n["cx"], 0.0))
                continue

        # require label above peak tip (same logic as before)
            y_tip = min(pk[1], pk[3])
            lbl_h = (n["bbox"][3] - n["bbox"][1]) or 8.0
            gap   = y_tip - n["cy"]  # >0 means label above the tip

            if gap <= max(-0.4, -0.08 * lbl_h):
            # below tip → treat as negligible alignment, keep with 0
                vals.append((n["cx"], n["text"]))
                abund.append((n["cx"], 0.0))
                continue
            if gap > 5.0 * lbl_h:
            # unrealistically far → keep with 0
                vals.append((n["cx"], n["text"]))
                abund.append((n["cx"], 0.0))
                continue

        # accepted with a real peak → compute relative abundance
            rel = None
            if plot_h and plot_h > 0:
                rel = max(0.0, min(100.0, ( (baseline_y - y_tip) / plot_h ) * 100.0))

            vals.append((n["cx"], n["text"]))
            abund.append((n["cx"], rel))
            elapsed = time.perf_counter() - t0
        vals.sort(key=lambda t: t[0])
        abund.sort(key=lambda t: t[0])

        spectrum = [t for _, t in vals]                 # list of label strings
        relative_abundance = [r for _, r in abund]
    
    profiler.stop()
    out_html = os.path.splitext(os.path.basename(pdf_path))[0] + f"_profile_{os.getpid()}.html"
    with open(out_html, "w") as f:
        f.write(profiler.output_html())
    
    #draw_debug_on_pdf_copy(
    #pdf_path=pdf_path, out_pdf="debug_overlay.pdf",
    #X0=X0, Y0=Y0, X1=X1, Y1=Y1,
    #baseline_y=baseline_y, top_y=top_y,
    #candidates=candidates,
    #verticals_strict=verticals_strict,
    #verticals_loose=verticals_loose
#)      # list of floats (0–100) or None
    # elapsed = time.perf_counter() - t0
    result= {
        "chemical_name": compound.strip(),
        "spectrum": spectrum,
        "relative_abundance": relative_abundance
        
    }
    return result,elapsed

    
    

def process_folder(folder_path: str, output_json: str = "Algo1_Parallel.json"):
    """
    Run compute_from_vector_pdf on all PDFs in a folder in parallel and save results into a JSON file.
    """
    pdf_files = sorted(glob.glob(os.path.join(folder_path, "*.pdf")))
    print(f"[info] found {len(pdf_files)} PDF files in {folder_path}")

    results = []
    core_times = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
        future_to_pdf = {
            executor.submit(compute_from_vector_pdf, pdf_path): pdf_path
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
        json.dump(results, f, ensure_ascii=False,  indent=2)
    print(f"[done] results saved to {output_json}")

    return results
    


    


# -------------------------
# Example usage:

#result = compute_from_vector_pdf("/Users/pramathkp/Downloads/Samples-25/74002A.pdf")


#print(json.dumps(result, indent=2))

if __name__ == "__main__":
    folder = "/Users/pramathkp/Desktop/FinalDataset"
    with timer("total_s"):
        all_results = process_folder(folder, output_json="Greedy.json")
    print(len(all_results))

    # optional: pretty print
    for r in all_results:
        print("\n=== ", r["file"], " ===")
        print("Chemical:", r["chemical_name"])
        print("Spectrum:", r["spectrum"])
        print("RA     :", r["relative_abundance"])


print(f"[TIMING] total = {METRICS['total_s']:.6f}s | core = {METRICS['core_s']:.6f}s")

