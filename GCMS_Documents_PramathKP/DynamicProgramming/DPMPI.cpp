#include <mupdf/fitz.h>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <locale>
#include <limits>
#include <mpi.h>

using json = nlohmann::json;
namespace fs = std::filesystem;

const double INF = 1e12;
const std::regex NUM_RE(R"(^\d{1,4}(\.\d{1,2})?$)");
const int MZ_MIN = 10;
const int MZ_MAX = 1500;

// Metrics for timing
struct Metrics {
    double total_s = 0.0;
    double core_s = 0.0;
};
Metrics METRICS;

// Timer macro for measuring execution time
#define TIMER(bucket_key, block) do { \
    auto t0 = std::chrono::high_resolution_clock::now(); \
    block; \
    auto t1 = std::chrono::high_resolution_clock::now(); \
    METRICS.bucket_key += std::chrono::duration<double>(t1 - t0).count(); \
} while (0)

// Struct definitions
struct BBox {
    double x0, y0, x1, y1;
};

struct Word {
    double x0, y0, x1, y1;
    std::string text;
};

struct Segment {
    double x0, y0, x1, y1;
};

struct Label {
    std::string text;
    double value;
    BBox bbox;
    double cx, cy;
    bool grouped = false;
};

struct Peak {
    double x, y_tip, h, RA;
};

struct Axes {
    double baseline_y;
    double top_y;
    Segment x_axis_seg;
    Segment y_axis_seg;
    BBox plot_box;
};

// Helper functions
double cx(const BBox& b) { return 0.5 * (b.x0 + b.x1); }
double cy(const BBox& b) { return 0.5 * (b.y0 + b.y1); }
double get_axis_max(const std::vector<Word>& words, double baseline_y,
                    double X0p, double X1p, double tolerance = 15.0) {
    double max_val = -1.0;
    for (const auto& w : words) {
        BBox b = {w.x0, w.y0, w.x1, w.y1};
        double cx_val = cx(b);
        double cy_val = cy(b);
        if (!(X0p <= cx_val && cx_val <= X1p)) continue;
        if (!(baseline_y - tolerance <= cy_val && cy_val <= baseline_y + tolerance)) continue;
        try {
            double v = std::stod(w.text);
            if (v > 0 && v == std::floor(v))
                max_val = std::max(max_val, v);
        } catch (...) {}
    }
    return max_val;
}
// Forward declaration
std::vector<Word> get_words(fz_context* ctx, fz_page* page);

// extract_compound with Unicode support
std::string extract_compound(fz_context* ctx, fz_page* page) {
    auto words = get_words(ctx, page);
    std::vector<std::tuple<double, double, std::string>> norm;
    for (const auto& w : words) {
        if (!w.text.empty()) {
            norm.emplace_back(w.x0, w.y0, w.text);
        }
    }
    if (norm.empty()) return "";

    std::map<int, std::vector<std::pair<double, std::string>>> lines;
    for (const auto& [x0, y0, t] : norm) {
        int key = std::round(y0 / 2.0);
        lines[key].emplace_back(x0, t);
    }

    std::string best = "";
    int blen = -1;
    for (auto& [key, items] : lines) {
        //std::sort(items.begin(), items.end());
        std::string line;
        for (const auto& [x, s] : items) {
            if (!line.empty()) line += " ";
            line += s;
        }
        line = std::regex_replace(line, std::regex("\\(mainlib\\)"), "");
        size_t start = line.find_first_not_of(" \t");
        size_t end = line.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start, end - start + 1);
        if (std::regex_search(line, std::regex("[A-Za-z]")) && static_cast<int>(line.length()) > blen) {
            best = line;
            blen = line.length();
        }
    }
    std::cout << "[debug] Extracted compound: " << best << std::endl;
    return best;
}

// get_words with UTF-8 handling
std::vector<Word> get_words(fz_context* ctx, fz_page* page) {
    std::vector<Word> words;
    fz_stext_options options = {0};
    options.flags |= FZ_STEXT_PRESERVE_SPANS;
    fz_stext_page* text_page = fz_new_stext_page_from_page(ctx, page, &options);

    for (fz_stext_block* block = text_page->first_block; block; block = block->next) {
        if (block->type == FZ_STEXT_BLOCK_TEXT) {
            for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                std::string current_word;
                double min_x = INF, min_y = INF, max_x = -INF, max_y = -INF;
                bool in_word = false;
                for (fz_stext_char* ch = line->first_char; ch; ch = ch->next) {
                    fz_rect char_bbox = fz_rect_from_quad(ch->quad);
                    if (ch->c == ' ' || ch->c < 0) {
                        if (!current_word.empty()) {
                            words.push_back({min_x, min_y, max_x, max_y, current_word});
                            current_word.clear();
                            in_word = false;
                        }
                    } else {
                        in_word = true;
                        char buf[8];
                        int len = fz_runetochar(buf, ch->c);
                        current_word.append(buf, len);
                        min_x = std::min(min_x, (double)char_bbox.x0);
                        min_y = std::min(min_y, (double)char_bbox.y0);
                        max_x = std::max(max_x, (double)char_bbox.x1);
                        max_y = std::max(max_y, (double)char_bbox.y1);
                    }
                }
                if (!current_word.empty()) {
                    words.push_back({min_x, min_y, max_x, max_y, current_word});
                }
            }
        }
    }

    fz_drop_stext_page(ctx, text_page);
    return words;
}

// Custom walker for paths
struct line_walker {
    fz_point current;
    fz_matrix ctm;
    std::vector<Segment>* segs;
};

static void move_to_fn(fz_context* ctx, void* user, float x, float y) {
    line_walker* w = (line_walker*)user;
    fz_point p = {x, y};
    w->current = fz_transform_point(p, w->ctm);
}

static void line_to_fn(fz_context* ctx, void* user, float x, float y) {
    line_walker* w = (line_walker*)user;
    fz_point to = {x, y};
    to = fz_transform_point(to, w->ctm);
    w->segs->push_back({w->current.x, w->current.y, to.x, to.y});
    w->current = to;
}

static void curve_to_fn(fz_context* ctx, void* user, float x1, float y1, float x2, float y2, float x3, float y3) {}

static void close_fn(fz_context* ctx, void* user) {}

static fz_path_walker path_walker = {
    move_to_fn,
    line_to_fn,
    curve_to_fn,
    close_fn,
    nullptr, nullptr, nullptr, nullptr
};

// Custom device for drawings
struct my_device {
    fz_device device;
    std::vector<Segment>* segs;
};

static void my_stroke_path(fz_context* ctx, fz_device* dev, const fz_path* path, const fz_stroke_state* stroke, fz_matrix ctm, fz_colorspace* cs, const float* color, float alpha, fz_color_params cp) {
    my_device* mydev = (my_device*)dev;
    line_walker w;
    w.ctm = ctm;
    w.segs = mydev->segs;
    fz_walk_path(ctx, path, &path_walker, &w);
}

static my_device* create_my_device(fz_context* ctx, std::vector<Segment>* segs) {
    my_device* d = fz_new_derived_device(ctx, my_device);
    d->device.stroke_path = my_stroke_path;
    d->segs = segs;
    return d;
}

// get_segments
std::vector<Segment> get_segments(fz_context* ctx, fz_page* page) {
    std::vector<Segment> segs;
    my_device* dev = create_my_device(ctx, &segs);
    fz_run_page(ctx, page, (fz_device*)dev, fz_identity, nullptr);
    fz_drop_device(ctx, (fz_device*)dev);
    return segs;
}

// enum_partitions
std::vector<std::vector<int>> enum_partitions(int n, int min_len = 2, int max_len = 4, int min_parts = 2, int max_parts = 4) {
    std::vector<std::vector<int>> res;
    std::vector<int> cur;
    std::function<void(int)> dfs = [&](int rem) {
        if (rem == 0 && min_parts <= static_cast<int>(cur.size()) && static_cast<int>(cur.size()) <= max_parts) {
            res.push_back(cur);
            return;
        }
        if (rem <= 0 || static_cast<int>(cur.size()) >= max_parts) return;
        for (int L = min_len; L <= max_len; ++L) {
            if (L <= rem) {
                cur.push_back(L);
                dfs(rem - L);
                cur.pop_back();
            }
        }
    };
    dfs(n);
    return res;
}

// split_grouped_numbers
std::vector<Label> split_grouped_numbers(const std::string& text, const BBox& bbox, int mz_min = MZ_MIN, int mz_max=MZ_MAX) {
    double x0 = bbox.x0, y0 = bbox.y0, x1 = bbox.x1, y1 = bbox.y1;
    double total_w = std::max(1e-6, x1 - x0);

    auto in_range = [mz_min, mz_max](const std::string& s) -> bool {
        try {
            int v = std::stoi(s);
            return mz_min <= v && v <= mz_max;
        } catch (...) {
            return false;
        }
    };

    auto score_partition = [](const std::vector<int>& lengths, const std::vector<std::string>& pieces) -> int {
        int s = 10 * std::accumulate(lengths.begin(), lengths.end(), 0, [](int acc, int L) { return acc + (L - 3) * (L - 3); });
        s += 6 * (static_cast<int>(lengths.size()) - 2);
        s += std::count_if(pieces.begin(), pieces.end(), [](const std::string& p) { return p.length() == 2; });
        s += 100 * std::count_if(pieces.begin(), pieces.end(), [](const std::string& p) { return !p.empty() && p[0] == '0'; });
        return s;
    };

    auto best_split_digits = [&](const std::string& s) -> std::vector<std::string> {
        int n = s.length();
        if (1 <= n && n <= 3 && in_range(s)) return {s};
        std::vector<std::string> best;
        int best_sc = std::numeric_limits<int>::max();
        auto partitions = enum_partitions(n);
        for (const auto& lens : partitions) {
            size_t pos = 0;
            std::vector<std::string> pcs;
            bool ok = true;
            for (int L : lens) {
                std::string piece = s.substr(pos, L);
                pos += L;
                if (in_range(piece)) pcs.push_back(piece);
                else { ok = false; break; }
            }
            if (!ok) continue;
            int sc = score_partition(lens, pcs);
            if (sc < best_sc) {
                best_sc = sc;
                best = pcs;
            }
        }
        return best;
    };

    auto bbox_for_span = [&](int i, int j) -> std::pair<BBox, double> {
        double char_w = total_w / std::max(1, static_cast<int>(text.length()));
        double sub_x0 = x0 + i * char_w;
        double sub_x1 = x0 + j * char_w;
        return {{sub_x0, y0, sub_x1, y1}, 0.5 * (sub_x0 + sub_x1)};
    };

    std::string text_norm = std::regex_replace(text, std::regex(R"([\u00A0\u2000-\u200B\u202F\u205F\u3000])"), " ");
    std::vector<std::string> tokens;
    std::smatch match;
    std::string temp = text_norm;
    while (std::regex_search(temp, match, std::regex(R"(\s+)"))) {
        tokens.push_back(match.prefix());
        temp = match.suffix();
    }
    if (!temp.empty()) tokens.push_back(temp);
    if (tokens.empty()) tokens = {text_norm};

    std::vector<Label> out;
    size_t cursor = 0;
    bool any_found = false;

    auto find_from = [&](const std::string& substr, size_t start) -> size_t {
        size_t k = text.find(substr, start);
        if (k != std::string::npos) return k;
        std::string digits = std::regex_replace(substr, std::regex("[^0-9]"), "");
        return text.find(digits, start);
    };

    for (const auto& tok : tokens) {
        if (tok.empty()) {
            cursor = std::min(text.length(), cursor + 1);
            continue;
        }
        std::string digits = std::regex_replace(tok, std::regex("[^0-9]"), "");
        if (digits.empty()) {
            size_t k = find_from(tok, cursor);
            if (k != std::string::npos) cursor = k + tok.length();
            continue;
        }
        auto pieces = best_split_digits(digits);
        size_t k = find_from(tok, cursor);
        size_t tok_len = (k != std::string::npos) ? tok.length() : digits.length();
        if (k == std::string::npos) k = cursor;
        if (!pieces.empty()) {
            any_found = true;
            size_t pos = 0;
            for (const auto& p : pieces) {
                int i = static_cast<int>(k + pos);
                int j = i + p.length();
                auto [sub_bbox, sub_cx] = bbox_for_span(i, j);
                out.push_back({p, std::stod(p), sub_bbox, sub_cx, (y0 + y1) / 2.0, true});
                pos += p.length();
            }
        }
        cursor = std::max(cursor, k + tok_len);
    }

    if (!any_found) {
        std::string digits_all = std::regex_replace(text_norm, std::regex("[^0-9]"), "");
        auto pieces = best_split_digits(digits_all);
        size_t pos = 0;
        for (const auto& p : pieces) {
            int i = static_cast<int>(pos);
            int j = i + static_cast<int>(p.length());
            auto [sub_bbox, sub_cx] = bbox_for_span(i, j);
            out.push_back({p, std::stod(p), sub_bbox, sub_cx, (y0 + y1) / 2.0, true});
            pos = j;
        }
    }
    return out;
}

// split_axes
Axes split_axes(const std::vector<Segment>& segs, double W, double H, const std::vector<Word>& words, double loosen = 2.0) {
    const double H_TOL = 0.8;
    const double V_TOL = 0.8;
    std::vector<Segment> horizontals;
    for (const auto& ln : segs) {
        if (std::abs(ln.y1 - ln.y0) < H_TOL) horizontals.push_back(ln);
    }
    std::vector<Segment> verticals;
    for (const auto& ln : segs) {
        if (std::abs(ln.x1 - ln.x0) < V_TOL) verticals.push_back(ln);
    }

    double baseline_y = 0.0;
    Segment x_axis_seg = {0,0,0,0};
    if (!horizontals.empty()) {
        std::vector<Segment> lower;
        for (const auto& ln : horizontals) {
            if (ln.y0 > 0.55 * H) lower.push_back(ln);
        }
        auto cand_it = std::max_element((lower.empty() ? horizontals.begin() : lower.begin()), (lower.empty() ? horizontals.end() : lower.end()), 
            [](const Segment& a, const Segment& b) { return std::abs(a.x1 - a.x0) < std::abs(b.x1 - b.x0); });
        baseline_y = cand_it->y0;
        x_axis_seg = *cand_it;
    }
    if (baseline_y == 0.0) baseline_y = 0.90 * H;

    double top_y = 0.0;
    std::vector<Word> tick100;
    for (const auto& w : words) {
        if (w.text == "100") tick100.push_back(w);
    }
    if (!tick100.empty()) {
        auto min_it = std::min_element(tick100.begin(), tick100.end(), [](const Word& a, const Word& b) { return a.y1 < b.y1; });
        top_y = min_it->y1;
    }
    if (top_y == 0.0) {
        if (!verticals.empty()) {
            std::vector<double> y_tops;
            for (const auto& v : verticals) y_tops.push_back(std::min(v.y0, v.y1));
            if (!y_tops.empty()) top_y = *std::min_element(y_tops.begin(), y_tops.end());
            else top_y = 0.10 * H;
        } else {
            top_y = 0.10 * H;
        }
    }

    const double LEFT_X_MAX = 0.18 * W;
    const double BASE_Y_TOL = 4.5;
    double plot_h_est = std::max(1.0, baseline_y - top_y);
    const double MIN_YAXIS_H = std::max(0.25 * H, 0.50 * plot_h_est);

    Segment y_axis_seg = {0,0,0,0};
    for (const auto& ln : verticals) {
        double x = 0.5 * (ln.x0 + ln.x1);
        double y_top = std::min(ln.y0, ln.y1);
        double y_bot = std::max(ln.y0, ln.y1);
        double h = y_bot - y_top;
        if (x <= LEFT_X_MAX && std::abs(y_bot - baseline_y) <= BASE_Y_TOL && h >= MIN_YAXIS_H) {
            if (y_axis_seg.x0 == 0 && y_axis_seg.x1 == 0 || (y_axis_seg.y1 - y_axis_seg.y0) < h) {
                y_axis_seg = ln;
            }
        }
    }

    double base_clear = std::max({10.0, 0.10 * plot_h_est, 0.012 * H});
    double CLEAR = base_clear / std::max(1.0, loosen);
    double LEFT_OFFSET = 6.0 / std::max(1.0, loosen);
    double LEFT_MARGIN = (0.09 / std::sqrt(std::max(1.0, loosen))) * W;
    double RIGHT_MARGINX = 0.99 * W;

    double X0_plot = (y_axis_seg.x0 != 0 || y_axis_seg.x1 != 0) ? std::max(LEFT_MARGIN, 0.5 * (y_axis_seg.x0 + y_axis_seg.x1) + LEFT_OFFSET) : LEFT_MARGIN;
    double X1_plot = RIGHT_MARGINX;
    double HEADROOM = std::max({0.10 * plot_h_est, 0.03 * H, 18.0});
    double Y0_plot = std::max(0.01 * H, top_y - HEADROOM);
    double Y1_plot = baseline_y - 0.25 * std::max({10.0, 0.10 * plot_h_est, 0.012 * H});

    if (Y0_plot >= Y1_plot) {
        double mid = (Y0_plot + Y1_plot) / 2.0;
        Y0_plot = std::max(0.05 * H, mid - 1.0);
        Y1_plot = std::min(baseline_y - 2.0, mid + 1.0);
    }

    return {baseline_y, top_y, x_axis_seg, y_axis_seg, {X0_plot, Y0_plot, X1_plot, Y1_plot}};
}

// compute_from_vector_pdf_algo2
std::pair<json, double> compute_from_vector_pdf_algo2(const std::string& pdf_path) {
    // Reset METRICS.core_s for this thread
    METRICS.core_s = 0.0;

    fz_context* ctx = fz_new_context(nullptr, nullptr, FZ_STORE_UNLIMITED);
    if (!ctx) {
        std::cerr << "[error] Failed to create MuPDF context for " << pdf_path << std::endl;
        return {json{}, 0.0};
    }
    fz_register_document_handlers(ctx);

    fz_document* doc = nullptr;
    try {
        doc = fz_open_document(ctx, pdf_path.c_str());
    } catch (...) {
        std::cerr << "[error] Failed to open document: " << pdf_path << std::endl;
        fz_drop_context(ctx);
        return {json{}, 0.0};
    }

    fz_page* page = fz_load_page(ctx, doc, 0);
    fz_rect rect = fz_bound_page(ctx, page);
    double W = rect.x1 - rect.x0;
    double H = rect.y1 - rect.y0;

    auto words = get_words(ctx, page);
    auto segs = get_segments(ctx, page);
    auto axes = split_axes(segs, W, H, words);

    double baseline_y = axes.baseline_y;
    double top_y = axes.top_y;
    double X0 = axes.plot_box.x0, Y0 = axes.plot_box.y0, X1 = axes.plot_box.x1, Y1 = 369.0;

    double axis_max = get_axis_max(words, baseline_y, X0, X1);
    int effective_mz_max = (axis_max > 0) ? static_cast<int>(axis_max) : MZ_MAX;

    std::string compound = extract_compound(ctx, page);

    // Build peak lists from vectors
    const double V_TOL = 0.8;
    std::vector<Segment> verticals_all;
    for (const auto& ln : segs) {
        if (std::abs(ln.x1 - ln.x0) < V_TOL) verticals_all.push_back(ln);
    }

    auto y_tip = [](const Segment& seg) { return std::min(seg.y0, seg.y1); };

    // Adaptive tick-height floor
    std::vector<double> tick_heights;
    for (const auto& ln : verticals_all) {
        double yt = y_tip(ln);
        double yb = std::max(ln.y0, ln.y1);
        if (std::abs(yb - baseline_y) <= 3.0) {
            double h = baseline_y - yt;
            if (h > 0 && h <= 22.0) tick_heights.push_back(h);
        }
    }
    std::sort(tick_heights.begin(), tick_heights.end());
    double p90_tick = tick_heights.empty() ? 2.5 : tick_heights[std::round(0.9 * (tick_heights.size() - 1))];
    double MIN_PEAK_H_SMALL = std::max(p90_tick + 0.3, 1.0);

    // Peaks and peaks_any
    std::vector<Peak> peaks, peaks_any;
    for (const auto& ln : verticals_all) {
        double xt = 0.5 * (ln.x0 + ln.x1);
        double yt = y_tip(ln);
        double yb = std::max(ln.y0, ln.y1);
        if (std::abs(yb - baseline_y) <= 4.0) {
            double h = baseline_y - yt;
            if (h > 0.5) {
                peaks_any.push_back({xt, yt, h, 0.0});
            }
            if (h >= MIN_PEAK_H_SMALL) {
                peaks.push_back({xt, yt, h, 0.0});
            }
        }
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b) { return a.x < b.x; });
    std::sort(peaks_any.begin(), peaks_any.end(), [](const Peak& a, const Peak& b) { return a.x < b.x; });

    // Normalization height
    double plot_h = (top_y < baseline_y) ? baseline_y - top_y : 0.0;
    if (plot_h <= 0) {
        double max_h = 0.0;
        for (const auto& p : peaks) max_h = std::max(max_h, p.h);
        plot_h = max_h;
    }
    if (plot_h <= 0) {
        fz_drop_page(ctx, page);
        fz_drop_document(ctx, doc);
        fz_drop_context(ctx);
        return {json{{"chemical_name", compound}, {"spectrum", json::array()}, {"relative_abundance", json::array()}}, METRICS.core_s};
    }

    for (auto& p : peaks) {
        p.RA = std::max(0.0, std::min(100.0, (p.h / plot_h) * 100.0));
    }
    for (auto& p : peaks_any) {
        p.RA = std::max(0.0, std::min(100.0, (p.h / plot_h) * 100.0));
    }

    // Extract labels with splitting
    std::vector<Label> labels;
    for (const auto& w : words) {
        std::string t = w.text;
        t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
        BBox b = {w.x0, w.y0, w.x1, w.y1};
        double cx_val = cx(b);
        double cy_val = cy(b);
        if (!(X0 <= cx_val && cx_val <= X1 && Y0 <= cy_val && cy_val <= Y1)) continue;

        std::smatch m;
        if (std::regex_match(t, m, NUM_RE)) {
            try {
                double v = std::stod(t);
                if (MZ_MIN <= v && v <= effective_mz_max && t.length() <=3) {
                    labels.push_back({t, v, b, cx_val, cy_val});
                    continue;
                }
            } catch (...) {}
        }

        auto split_candidates = split_grouped_numbers(t, b,MZ_MIN, effective_mz_max);
        for (const auto& sc : split_candidates) {
            if (X0 <= sc.cx && sc.cx <= X1 && Y0 <= sc.cy && sc.cy <= Y1) {
                labels.push_back(sc);
            }
        }
    }

    // De-duplicate labels
    std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b) {
        return std::tie(a.cx, a.cy, a.text) < std::tie(b.cx, b.cy, b.text);
    });
    std::vector<Label> dedup;
    auto close = [](double a, double b, double tol = 1.4) { return std::abs(a - b) <= tol; };
    for (const auto& lab : labels) {
        if (dedup.empty()) {
            dedup.push_back(lab);
            continue;
        }
        const auto& last = dedup.back();
        if (lab.text == last.text && close(lab.cx, last.cx) && close(lab.cy, last.cy, 3.0)) {
            continue;
        }
        dedup.push_back(lab);
    }
    labels = std::move(dedup);
    std::sort(labels.begin(), labels.end(), [](const Label& a, const Label& b) { return a.cx < b.cx; });

    // Monotone assignment (Dynamic Programming)
    std::vector<std::string> spectrum;
    std::vector<double> relative_abundance;
    size_t nP = peaks.size();
    size_t nL = labels.size();
    if (nL == 0) {
        fz_drop_page(ctx, page);
        fz_drop_document(ctx, doc);
        fz_drop_context(ctx);
        return {json{{"chemical_name", compound}, {"spectrum", json::array()}, {"relative_abundance", json::array()}}, METRICS.core_s};
    }

    const double match_penalty = 8.0;
    const double gap_penalty = 8.0;
    std::vector<std::vector<double>> dp(nP + 1, std::vector<double>(nL + 1, INF));
    std::vector<std::vector<std::tuple<std::string, size_t, size_t>>> bt(nP + 1, std::vector<std::tuple<std::string, size_t, size_t>>(nL + 1));
    dp[0][0] = 0.0;

    TIMER(core_s, {
        for (size_t i = 0; i <= nP; ++i) {
            for (size_t j = 0; j <= nL; ++j) {
                if (dp[i][j] >= INF) continue;
                if (i < nP) {
                    double v = dp[i][j] + match_penalty;
                    if (v < dp[i + 1][j]) {
                        dp[i + 1][j] = v;
                        bt[i + 1][j] = std::make_tuple("skipP", i, j);
                    }
                }
                if (j < nL) {
                    double v = dp[i][j] + gap_penalty;
                    if (v < dp[i][j + 1]) {
                        dp[i][j + 1] = v;
                        bt[i][j + 1] = std::make_tuple("skipL", i, j);
                    }
                }
                if (i < nP && j < nL) {
                    double cost = std::abs(peaks[i].x - labels[j].cx);
                    double v = dp[i][j] + cost;
                    if (v < dp[i + 1][j + 1]) {
                        dp[i + 1][j + 1] = v;
                        bt[i + 1][j + 1] = std::make_tuple("match", i, j);
                    }
                }
            }
        }
    });

    // Backtrack
    std::vector<std::pair<size_t, size_t>> pairs;
    size_t i = nP, j = nL;
    while (i > 0 || j > 0) {
        const auto& op = bt[i][j];
        if (!std::get<0>(op).empty()) {
            auto [typ, pi, pj] = op;
            if (typ == "match") {
                pairs.emplace_back(pj, pi);
                i = pi;
                j = pj;
            } else if (typ == "skipP") {
                i = pi;
                j = pj;
            } else if (typ == "skipL") {
                pairs.emplace_back(pj, static_cast<size_t>(-1));
                i = pi;
                j = pj;
            }
        } else {
            if (j > 0) {
                pairs.emplace_back(j - 1, static_cast<size_t>(-1));
                --j;
            } else {
                --i;
            }
        }
    }
    std::reverse(pairs.begin(), pairs.end());

    // RA from peak below
    auto ra_from_peak_below = [&](const Label& label) -> double {
        double lx = label.cx, ly = label.cy;
        double lbl_w = std::max(label.bbox.x1 - label.bbox.x0, 8.0);
        double lbl_h = std::max(label.bbox.y1 - label.bbox.y0, 8.0);
        double XTOL_strict = std::max({2.5, 0.0045 * W, 0.55 * lbl_w});
        double XTOL_relaxed = XTOL_strict * 1.6;

        auto ok_gap = [&](const Peak& p, bool allow_looser) {
            double gap = p.y_tip - ly;
            double lo = allow_looser ? std::max(-1.4, -0.16 * lbl_h) : std::max(-1.0, -0.12 * lbl_h);
            double hi = allow_looser ? 6.0 * lbl_h : 5.0 * lbl_h;
            return gap > lo && gap <= hi;
        };

        const Peak* best = nullptr;
        std::tuple<double, double> best_key = {INF, INF};
        for (const auto& p : peaks) {
            double dx = std::abs(p.x - lx);
            if (dx > XTOL_strict) continue;
            if (!ok_gap(p, false)) continue;
            std::tuple<double, double> key = {dx, -p.h};
            if (key < best_key) {
                best_key = key;
                best = &p;
            }
        }
        if (best) return best->RA;

        for (const auto& p : peaks_any) {
            double dx = std::abs(p.x - lx);
            if (dx > XTOL_relaxed) continue;
            if (!ok_gap(p, true)) continue;
            std::tuple<double, double> key = {dx, -p.h};
            if (key < best_key) {
                best_key = key;
                best = &p;
            }
        }
        if (best) return best->RA;

        best = nullptr;
        double best_dx = INF;
        for (const auto& p : peaks_any) {
            double dx = std::abs(p.x - lx);
            if (dx < best_dx) {
                best_dx = dx;
                best = &p;
            }
        }
        return best ? best->RA : 0.0;
    };

    // Emit results
    for (const auto& [j_idx, i_idx] : pairs) {
        if (j_idx == static_cast<size_t>(-1) || j_idx >= nL) continue;
        const auto& lbl = labels[j_idx];
        spectrum.push_back(lbl.text);
        if (i_idx != static_cast<size_t>(-1)) {
            relative_abundance.push_back(peaks[i_idx].RA);
        } else {
            relative_abundance.push_back(ra_from_peak_below(lbl));
        }
    }

    json res;
    res["chemical_name"] = compound;
    res["spectrum"] = spectrum;
    res["relative_abundance"] = relative_abundance;

    fz_drop_page(ctx, page);
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return {res, METRICS.core_s};
}

// process_folder
std::vector<json> process_folder(const std::string& folder_path, const std::string& output_json, int rank, int size) {
    std::vector<std::string> pdf_files;
    int n_files = 0;

    if (rank == 0) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.path().extension() == ".pdf") pdf_files.push_back(entry.path().string());
        }
        std::sort(pdf_files.begin(), pdf_files.end());
        n_files = pdf_files.size();
        std::cout << "[info] found " << n_files << " PDF files in " << folder_path << std::endl;
    }

    MPI_Bcast(&n_files, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n_files == 0) {
        if (rank == 0) return {};
        else return {};
    }

    // Compute distribution
    std::vector<int> counts(size, n_files / size);
    std::vector<int> displs(size, 0);
    int remainder = n_files % size;
    for (int i = 0; i < remainder; ++i) {
        counts[i]++;
    }
    for (int i = 1; i < size; ++i) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }

    // Distribute PDF paths
    std::vector<std::string> local_files(counts[rank]);
    if (rank == 0) {
        // Send to others
        for (int p = 1; p < size; ++p) {
            for (int k = 0; k < counts[p]; ++k) {
                std::string s = pdf_files[displs[p] + k];
                int len = s.length();
                MPI_Send(&len, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
                MPI_Send(s.c_str(), len, MPI_CHAR, p, 0, MPI_COMM_WORLD);
            }
        }
        // Own share
        for (int k = 0; k < counts[0]; ++k) {
            local_files[k] = pdf_files[displs[0] + k];
        }
    } else {
        for (int k = 0; k < counts[rank]; ++k) {
            int len;
            MPI_Recv(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            std::vector<char> buf(len);
            MPI_Recv(buf.data(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            local_files[k] = std::string(buf.begin(), buf.end());
        }
    }

    // Process local files
    std::vector<json> local_results(local_files.size());
    std::vector<double> local_core_times(local_files.size(), 0.0);
    std::vector<std::string> local_errors(local_files.size());

    for (size_t i = 0; i < local_files.size(); ++i) {
        const auto& pdf_path = local_files[i];
        try {
            auto [res, elapsed] = compute_from_vector_pdf_algo2(pdf_path);
            res["file"] = fs::path(pdf_path).filename().string();
            local_results[i] = res;
            local_core_times[i] = elapsed;
            std::cout << "[done] " << pdf_path << " in " << elapsed << "s" << std::endl;
        } catch (const std::exception& e) {
            local_errors[i] = e.what();
            std::cerr << "[error] " << pdf_path << ": " << e.what() << std::endl;
        }
    }

    std::vector<json> all_results;
    std::vector<double> all_core_times;
    std::vector<std::string> all_errors;

    if (rank == 0) {
        all_results.resize(n_files);
        all_core_times.resize(n_files);
        all_errors.resize(n_files);

        // Own data
        for (int k = 0; k < counts[0]; ++k) {
            all_results[displs[0] + k] = local_results[k];
            all_core_times[displs[0] + k] = local_core_times[k];
            all_errors[displs[0] + k] = local_errors[k];
        }

        // Receive from others
        for (int p = 1; p < size; ++p) {
            for (int k = 0; k < counts[p]; ++k) {
                // Receive json
                int len;
                MPI_Recv(&len, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string json_str;
                if (len > 0) {
                    std::vector<char> buf(len);
                    MPI_Recv(buf.data(), len, MPI_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    json_str.assign(buf.begin(), buf.end());
                    all_results[displs[p] + k] = json::parse(json_str);
                } else {
                    all_results[displs[p] + k] = json();
                }

                // Receive core_time
                double elapsed;
                MPI_Recv(&elapsed, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                all_core_times[displs[p] + k] = elapsed;

                // Receive error
                int elen;
                MPI_Recv(&elen, 1, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                std::string err;
                if (elen > 0) {
                    std::vector<char> ebuf(elen);
                    MPI_Recv(ebuf.data(), elen, MPI_CHAR, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    err.assign(ebuf.begin(), ebuf.end());
                }
                all_errors[displs[p] + k] = err;
            }
        }
    } else {
        // Send to master
        for (size_t i = 0; i < local_files.size(); ++i) {
            std::string json_str = local_results[i].is_null() ? "" : local_results[i].dump();
            int len = json_str.length();
            MPI_Send(&len, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (len > 0) {
                MPI_Send(json_str.c_str(), len, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }

            double elapsed = local_core_times[i];
            MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

            std::string err = local_errors[i];
            int elen = err.length();
            MPI_Send(&elen, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            if (elen > 0) {
                MPI_Send(err.c_str(), elen, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
    }

    if (rank != 0) {
        return {};
    }

    // Filter valid
    std::vector<json> valid_results;
    std::vector<double> valid_core_times;
    for (size_t i = 0; i < all_results.size(); ++i) {
        if (!all_results[i].is_null()) {
            valid_results.push_back(all_results[i]);
            valid_core_times.push_back(all_core_times[i]);
        }
    }

    double total_core = std::accumulate(valid_core_times.begin(), valid_core_times.end(), 0.0);
    double avg_core = valid_core_times.empty() ? 0.0 : total_core / valid_core_times.size();
    std::cout << "[core timing] total across all files = " << total_core << "s, average per file = " << avg_core << "s" << std::endl;

    std::sort(valid_results.begin(), valid_results.end(), [](const json& a, const json& b) {
        return a["file"].get<std::string>() < b["file"].get<std::string>();
    });

    std::ofstream f(output_json, std::ios::out | std::ios::binary);
    f << json(valid_results).dump(2);
    f.close();

    std::cout << "[done] results saved to " << output_json << std::endl;

    return valid_results;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set console to UTF-8
    try {
        std::cout.imbue(std::locale("en_US.UTF-8"));
        std::cerr.imbue(std::locale("en_US.UTF-8"));
    } catch (const std::runtime_error& e) {
        std::cerr << "[warning] Failed to set UTF-8 locale: " << e.what() << ". Continuing with default locale." << std::endl;
    }

    std::string folder = "/Users/pramathkp/Desktop/FinalDataset";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto all_results = process_folder(folder, "Algo2_Parallel_MPI_Cpp.json", rank, size);
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_wall = std::chrono::duration<double>(t1 - t0).count();

    if (rank == 0) {
        std::cout << "[WALL CLOCK] total run = " << total_wall << "s" << std::endl;

        for (const auto& r : all_results) {
            std::cout << "\n=== " << r["file"] << " ===" << std::endl;
            std::cout << "Chemical: " << r["chemical_name"] << std::endl;
            std::cout << "Spectrum: ";
            for (const auto& s : r["spectrum"]) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "RA     : ";
            for (const auto& a : r["relative_abundance"]) std::cout << a << " ";
            std::cout << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}