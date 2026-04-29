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
// #include <omp.h>
#include <locale>

using json = nlohmann::json;
namespace fs = std::filesystem;

const double INF = 1e18;
const std::regex NUM_RE(R"(^\d{1,4}(\.\d{1,2})?$)");
const int MZ_MIN = 10;
const int MZ_MAX = 1500;

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

struct Axes {
    double baseline_y;
    double top_y;
    Segment x_axis_seg;
    Segment y_axis_seg;
    BBox plot_box;
};

// Helper functions
double get_cx(const BBox& b) { return 0.5 * (b.x0 + b.x1); }
double get_cy(const BBox& b) { return 0.5 * (b.y0 + b.y1); }

// get_words with UTF-8 handling
std::vector<Word> get_words(fz_context* ctx, fz_page* page) {
    std::vector<Word> words;
    fz_stext_options options = {0};
    options.flags |= FZ_STEXT_PRESERVE_SPANS; // Preserve detailed text information
    fz_stext_page* text_page = fz_new_stext_page_from_page(ctx, page, &options);

    for (fz_stext_block* block = text_page->first_block; block; block = block->next) {
        if (block->type == FZ_STEXT_BLOCK_TEXT) {
            for (fz_stext_line* line = block->u.t.first_line; line; line = line->next) {
                std::string current_word; // Directly use std::string for UTF-8
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
                        // Append character, assuming UTF-8 encoding
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

static void curve_to_fn(fz_context* ctx, void* user, float x1, float y1, float x2, float y2, float x3, float y3) {
    // Ignore curves
}

static void close_fn(fz_context* ctx, void* user) {
    // Ignore
}

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

// extract_compound with Unicode support
std::string extract_compound(fz_context* ctx, fz_page* page) {
    auto words = get_words(ctx, page);
    std::vector<std::tuple<double, double, std::string>> norm;
    for (const auto& w : words) {
        if (!w.text.empty()) {
            norm.emplace_back(w.x0, w.y0, w.text); // Keep original text with Unicode
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
        // std::sort(items.begin(), items.end());
        std::string line;
        for (const auto& [x, s] : items) {
            if (!line.empty()) line += " ";
            line += s;
        }
        // Minimal cleaning to preserve Unicode
        line = std::regex_replace(line, std::regex("\\(mainlib\\)"), "");
        // Only trim leading/trailing spaces, preserve internal spaces and Unicode
        size_t start = line.find_first_not_of(" \t");
        size_t end = line.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        line = line.substr(start, end - start + 1);
        if (std::regex_search(line, std::regex("[A-Za-z]")) && static_cast<int>(line.length()) > blen) {
            best = line;
            blen = line.length();
        }
    }
    // Debug: Print raw compound name
    std::cout << "[debug] Extracted compound: " << best << std::endl;
    return best;
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
std::vector<Label> split_grouped_numbers(const std::string& text, const BBox& bbox, int mz_min = MZ_MIN, int mz_max = MZ_MAX) {
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
        if (1 <= n && n <= 4 && in_range(s)) return {s};
        if (n == 6) {
            std::vector<std::string> cand = {s.substr(0, 3), s.substr(3)};
            if (std::all_of(cand.begin(), cand.end(), in_range)) return cand;
        }
        if (n == 9) {
            std::vector<std::string> cand = {s.substr(0, 3), s.substr(3, 3), s.substr(6)};
            if (std::all_of(cand.begin(), cand.end(), in_range)) return cand;
        }
        std::vector<std::string> best;
        int best_sc = INT_MAX;
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
        std::string digits = std::regex_replace(substr, std::regex(R"([^0-9])"), "");
        return text.find(digits, start);
    };

    for (const auto& tok : tokens) {
        if (tok.empty()) {
            cursor = std::min(text.length(), cursor + 1);
            continue;
        }
        std::string digits = std::regex_replace(tok, std::regex(R"([^0-9])"), "");
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
        std::string digits_all = std::regex_replace(text_norm, std::regex(R"([^0-9])"), "");
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

// get_candidates_with_splitting
std::vector<Label> get_candidates_with_splitting(const std::vector<Word>& words, double X0, double Y0, double X1, double Y1, double baseline_y) {
    std::vector<Label> candidates;
    for (const auto& w : words) {
        std::string t = w.text;
        t.erase(std::remove_if(t.begin(), t.end(), ::isspace), t.end());
        BBox b = {w.x0, w.y0, w.x1, w.y1};
        double cx_val = get_cx(b);
        double cy_val = get_cy(b);

        if (!(X0 <= cx_val && cx_val <= X1 && Y0 <= cy_val && cy_val <= Y1)) continue;
        if ((baseline_y - cy_val) < 0.01) continue;

        std::smatch m;
        if (std::regex_match(t, m, NUM_RE)) {
            try {
                double v = std::stod(t);
                if (MZ_MIN <= v && v <= MZ_MAX) {
                    candidates.push_back({t, v, b, cx_val, cy_val});
                    continue;
                }
            } catch (...) {}
        }

        auto split_candidates = split_grouped_numbers(t, b);
        for (const auto& sc : split_candidates) {
            if (X0 <= sc.cx && sc.cx <= X1 && Y0 <= sc.cy && sc.cy <= Y1 && (baseline_y - sc.cy) >= 0.01) {
                candidates.push_back(sc);
            }
        }
    }
    return candidates;
}

// compute_from_vector_pdf_algo1
std::pair<json, double> compute_from_vector_pdf_algo1(const std::string& pdf_path) {
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

    std::string compound = extract_compound(ctx, page);

    std::vector<Segment> verticals_all;
    const double V_TOL = 0.8;
    for (const auto& ln : segs) {
        if (std::abs(ln.x1 - ln.x0) < V_TOL) verticals_all.push_back(ln);
    }

    std::vector<Segment> verticals_strict, verticals_loose;
    for (const auto& ln : verticals_all) {
        double y_top = std::min(ln.y0, ln.y1);
        double y_bot = std::max(ln.y0, ln.y1);
        bool anchored = std::abs(y_bot - baseline_y) <= 3.0;
        double height = baseline_y - y_top;
        if (anchored && y_top < baseline_y - 6.0) {
            verticals_strict.push_back(ln);
        }
        if (anchored && height >= 2.0) {
            verticals_loose.push_back(ln);
        }
    }

    auto candidates = get_candidates_with_splitting(words, X0, Y0, X1, Y1, baseline_y);

    auto t0 = std::chrono::high_resolution_clock::now();

    auto nearest_in = [](const std::vector<Segment>& lines, double x_target, double x_tol) -> Segment* {
        Segment* best = nullptr;
        double best_dx = INF;
        for (const auto& ln : lines) {
            double x = 0.5 * (ln.x0 + ln.x1);
            double dx = std::abs(x - x_target);
            if (dx <= x_tol && dx < best_dx) {
                best = const_cast<Segment*>(&ln);
                best_dx = dx;
            }
        }
        return best;
    };

    auto peak_height = [baseline_y](const Segment& seg) { return baseline_y - std::min(seg.y0, seg.y1); };

    std::vector<double> tick_heights;
    for (const auto& ln : verticals_all) {
        double yt = std::min(ln.y0, ln.y1);
        double yb = std::max(ln.y0, ln.y1);
        if (std::abs(yb - baseline_y) <= 3.0) {
            double h = baseline_y - yt;
            if (h > 0 && h <= 22.0) tick_heights.push_back(h);
        }
    }
    std::sort(tick_heights.begin(), tick_heights.end());
    double MIN_PEAK_H_SMALL = tick_heights.empty() ? 2.5 : std::max(tick_heights[std::round(0.9 * (tick_heights.size() - 1))] + 0.6, 2.0);

    std::vector<std::pair<double, std::string>> vals;
    std::vector<std::pair<double, double>> abund;
    double plot_h = (top_y < baseline_y) ? baseline_y - top_y : 0.0;

    for (const auto& n : candidates) {
        double lbl_w = std::max(n.bbox.x1 - n.bbox.x0, 8.0);
        double XTOL = std::max({1.8, 0.0035 * W, 0.45 * lbl_w});

        Segment* pk = nearest_in(verticals_strict, n.cx, XTOL);
        if (!pk) pk = nearest_in(verticals_loose, n.cx, XTOL * 1.25);

        if (!pk) {
            Segment* best = nullptr;
            double best_dx = INF;
            for (const auto& ln : verticals_all) {
                double xmid = 0.5 * (ln.x0 + ln.x1);
                double dx = std::abs(xmid - n.cx);
                if (dx > XTOL * 1.5) continue;
                double yb = std::max(ln.y0, ln.y1);
                if (std::abs(yb - baseline_y) > 4.0) continue;
                double h = baseline_y - std::min(ln.y0, ln.y1);
                if (h < MIN_PEAK_H_SMALL) continue;
                if (dx < best_dx) {
                    best = const_cast<Segment*>(&ln);
                    best_dx = dx;
                }
            }
            pk = best;
        }

        if (!pk) {
            Segment* best = nullptr;
            double best_h = 0.0;
            for (const auto& ln : verticals_all) {
                double xmid = 0.5 * (ln.x0 + ln.x1);
                double dx = std::abs(xmid - n.cx);
                if (dx > XTOL * 2.0) continue;
                double yb = std::max(ln.y0, ln.y1);
                if (std::abs(yb - baseline_y) > 4.0) continue;
                double h = baseline_y - std::min(ln.y0, ln.y1);
                if (h > best_h) {
                    best_h = h;
                    best = const_cast<Segment*>(&ln);
                }
            }
            pk = best;
        }

        if (!pk) {
            vals.emplace_back(n.cx, n.text);
            abund.emplace_back(n.cx, 0.0);
            continue;
        }

        double y_tip = std::min(pk->y0, pk->y1);
        double lbl_h = std::max(n.bbox.y1 - n.bbox.y0, 8.0);
        double gap = y_tip - n.cy;

        if (gap <= std::max(-0.4, -0.08 * lbl_h) || gap > 5.0 * lbl_h) {
            vals.emplace_back(n.cx, n.text);
            abund.emplace_back(n.cx, 0.0);
            continue;
        }

        double rel = 0.0;
        if (plot_h > 0) {
            rel = std::max(0.0, std::min(100.0, (peak_height(*pk) / plot_h) * 100.0));
        }
        vals.emplace_back(n.cx, n.text);
        abund.emplace_back(n.cx, rel);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::sort(vals.begin(), vals.end());
    std::sort(abund.begin(), abund.end());

    std::vector<std::string> spectrum;
    std::vector<double> relative_abundance;
    for (const auto& v : vals) spectrum.push_back(v.second);
    for (const auto& a : abund) relative_abundance.push_back(a.second);

    json res;
    res["chemical_name"] = compound;
    res["spectrum"] = spectrum;
    res["relative_abundance"] = relative_abundance;

    fz_drop_page(ctx, page);
    fz_drop_document(ctx, doc);
    fz_drop_context(ctx);
    return {res, elapsed};
}

// process_folder
std::vector<json> process_folder(const std::string& folder_path, const std::string& output_json = "Algo1_Parallel_Cpp.json") {
    std::vector<std::string> pdf_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".pdf") pdf_files.push_back(entry.path().string());
    }
    std::sort(pdf_files.begin(), pdf_files.end());
    std::cout << "[info] found " << pdf_files.size() << " PDF files in " << folder_path << std::endl;

    std::vector<json> results(pdf_files.size());
    std::vector<double> core_times(pdf_files.size());
    std::vector<std::string> errors(pdf_files.size());

    // #pragma omp parallel for num_threads(6) schedule(dynamic)
    for (size_t i = 0; i < pdf_files.size(); ++i) {
        const auto& pdf_path = pdf_files[i];
        try {
            auto [res, elapsed] = compute_from_vector_pdf_algo1(pdf_path);
            res["file"] = fs::path(pdf_path).filename().string();
            results[i] = res;
            core_times[i] = elapsed;
            // #pragma omp critical
            std::cout << "[done] " << pdf_path << " in " << elapsed << "s" << std::endl;
        } catch (const std::exception& e) {
            errors[i] = e.what();
            // #pragma omp critical
            std::cerr << "[error] " << pdf_path << ": " << e.what() << std::endl;
        }
    }

    std::vector<json> valid_results;
    std::vector<double> valid_core_times;
    for (size_t i = 0; i < results.size(); ++i) {
        if (!results[i].is_null()) {
            valid_results.push_back(results[i]);
            valid_core_times.push_back(core_times[i]);
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

int main() {
    // Set console to UTF-8
    try {
        std::cout.imbue(std::locale("en_US.UTF-8"));
        std::cerr.imbue(std::locale("en_US.UTF-8"));
    } catch (const std::runtime_error& e) {
        std::cerr << "[warning] Failed to set UTF-8 locale: " << e.what() << ". Continuing with default locale." << std::endl;
    }

    std::string folder = "/Users/pramathkp/Desktop/FinalDataset";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto all_results = process_folder(folder, "Algo1_Serial_Cpp.json");
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_wall = std::chrono::duration<double>(t1 - t0).count();
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

    return 0;
}