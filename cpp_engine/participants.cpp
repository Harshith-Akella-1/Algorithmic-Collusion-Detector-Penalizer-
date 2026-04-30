// participants.cpp -- Market participant implementations.
//
// Direct port of Python simulator.py NoiseTrader, MarketMaker, ColluderPair.
// Same parameters, same logic flow, same RNG distribution calls.

#include "participants.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace engine {

// =======================================================================
// RNG helpers (match numpy distribution APIs)
// =======================================================================
static double uniform(std::mt19937_64& rng, double lo, double hi) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

static double normal(std::mt19937_64& rng, double mu, double sigma) {
    std::normal_distribution<double> dist(mu, sigma);
    return dist(rng);
}

static double exponential(std::mt19937_64& rng, double scale) {
    std::exponential_distribution<double> dist(1.0 / scale);
    return dist(rng);
}

static int64_t randint(std::mt19937_64& rng, int64_t lo, int64_t hi) {
    // [lo, hi) like numpy randint
    std::uniform_int_distribution<int64_t> dist(lo, hi - 1);
    return dist(rng);
}

static double rand01(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

static double round2(double x) {
    return std::round(x * 100.0) / 100.0;
}

static double sign(double x) {
    if (x > 0) return 1.0;
    if (x < 0) return -1.0;
    return 0.0;
}

// =======================================================================
// NoiseTrader
// =======================================================================
NoiseTrader::NoiseTrader(const std::string& name,
                         double rate, double p_aggressive,
                         double spread_dev, double p_cancel)
    : name(name), rate_(rate), p_aggressive_(p_aggressive),
      spread_dev_(spread_dev), p_cancel_(p_cancel) {}

void NoiseTrader::act(LOB& lob, double t, double anchor, double drift,
                      std::mt19937_64& rng) {
    // Python: if rng.random() > self.rate * DT: return
    // DT = 0.1
    if (rand01(rng) > rate_ * 0.1) return;

    // Lazy prune stale OIDs, then maybe cancel one
    if (!active_orders_.empty() && rand01(rng) < p_cancel_) {
        // Prune dead orders
        std::vector<int64_t> alive;
        for (int64_t oid : active_orders_) {
            if (lob.by_id.find(oid) != lob.by_id.end()) {
                alive.push_back(oid);
            }
        }
        active_orders_ = alive;
        if (!active_orders_.empty()) {
            int64_t idx = randint(rng, 0, (int64_t)active_orders_.size());
            int64_t cancel_oid = active_orders_[idx];
            lob.cancel(cancel_oid, t);
            active_orders_.erase(active_orders_.begin() + idx);
        }
    }

    // Bias: buy more when drift is positive
    double bias = 0.5 + 0.2 * sign(drift);
    std::string side = (rand01(rng) < bias) ? "buy" : "sell";
    int64_t qty = randint(rng, 10, 100);

    if (rand01(rng) < p_aggressive_) {
        lob.submit(name, side, "market", qty, t);
    } else {
        double off = exponential(rng, spread_dev_);
        double price = (side == "buy")
            ? round2(anchor - off)
            : round2(anchor + off);
        int64_t oid = lob.submit(name, side, "limit", qty, t, price);
        active_orders_.push_back(oid);
    }
}

// =======================================================================
// MarketMaker
// =======================================================================
MarketMaker::MarketMaker(const std::string& name, double half_spread,
                         int64_t qty, double refresh)
    : name_(name), half_spread_(half_spread), qty_(qty), refresh_(refresh),
      last_time_(-1e9), bid_id_(-1), ask_id_(-1) {}

void MarketMaker::act(LOB& lob, double t, double anchor) {
    if (t - last_time_ < refresh_) return;

    // Cancel old quotes
    if (bid_id_ >= 0) { lob.cancel(bid_id_, t); bid_id_ = -1; }
    if (ask_id_ >= 0) { lob.cancel(ask_id_, t); ask_id_ = -1; }

    bid_id_ = lob.submit(name_, "buy", "limit", qty_, t,
                          round2(anchor - half_spread_));
    ask_id_ = lob.submit(name_, "sell", "limit", qty_, t,
                          round2(anchor + half_spread_));
    last_time_ = t;
}

// =======================================================================
// ColluderPair
// =======================================================================
ColluderPair::ColluderPair(const std::string& name_a, const std::string& name_b,
                           const std::string& mode)
    : A_(name_a, 0.4, 0.25), B_(name_b, 0.4, 0.25), mode_(mode),
      win_start_(0), win_end_(0), has_window_(false),
      wash_next_time_(0),
      paint_phase_("idle"), paint_next_t_(0), paint_post_side_("buy"),
      paint_post_oid_(-1), paint_post_qty_(0),
      spoof_phase_("idle"), spoof_next_t_(0), spoof_side_("buy"),
      mirror_phase_("idle"), mirror_next_t_(0), mirror_side_("buy"),
      mirror_sync_lag_(0), mirror_B_build_delay_(0) {}

void ColluderPair::schedule(double t0, double t1, std::mt19937_64& rng) {
    win_start_ = t0;
    win_end_ = t1;
    has_window_ = true;

    if (mode_ == "wash") {
        wash_next_time_ = t0;
    } else if (mode_ == "paint") {
        paint_phase_ = "idle";
        paint_next_t_ = t0;
        paint_post_side_ = "buy";
        paint_post_oid_ = -1;
        paint_post_qty_ = 0;
    } else if (mode_ == "spoof") {
        spoof_phase_ = "idle";
        spoof_next_t_ = t0;
        spoof_oids_.clear();
        spoof_side_ = "buy";
    } else if (mode_ == "mirror") {
        mirror_phase_ = "idle";
        mirror_next_t_ = t0;
        mirror_side_ = "buy";
        mirror_A_oids_.clear();
        mirror_B_oids_.clear();
        mirror_sync_lag_ = 0.0;
        mirror_B_build_delay_ = 0.0;
    }
    (void)rng; // rng passed for API consistency
}

void ColluderPair::step(LOB& lob, double t, double anchor, double drift,
                        std::mt19937_64& rng) {
    // Always run cover trades
    A_.act(lob, t, anchor, drift, rng);
    B_.act(lob, t, anchor, drift, rng);

    if (mode_ == "none" || !has_window_) return;
    if (t < win_start_ || t > win_end_) return;

    if (mode_ == "wash")        do_wash(lob, t, anchor, rng);
    else if (mode_ == "paint")  do_paint(lob, t, anchor, rng);
    else if (mode_ == "spoof")  do_spoof(lob, t, anchor, rng);
    else if (mode_ == "mirror") do_mirror(lob, t, anchor, rng);
}

// -----------------------------------------------------------------------
// Wash trading: A posts a sell limit, B immediately buys via market order
// -----------------------------------------------------------------------
void ColluderPair::do_wash(LOB& lob, double t, double anchor,
                           std::mt19937_64& rng) {
    if (t >= wash_next_time_) {
        double p = round2(anchor + normal(rng, 0, 0.01));
        int64_t q = randint(rng, 50, 150);
        lob.submit(A_.name, "sell", "limit", q, t, p);
        lob.submit(B_.name, "buy", "market", q, t + 0.05);
        wash_next_time_ = t + uniform(rng, 8, 20);
    }
}

// -----------------------------------------------------------------------
// Paint the tape: A posts a limit, B crosses it shortly after
// -----------------------------------------------------------------------
void ColluderPair::do_paint(LOB& lob, double t, double anchor,
                            std::mt19937_64& rng) {
    if (paint_phase_ == "idle" && t >= paint_next_t_) {
        paint_phase_ = "post";
        paint_next_t_ = t;
    }
    else if (paint_phase_ == "post" && t >= paint_next_t_) {
        int64_t qty = randint(rng, 10, 99);
        double bb = lob.best_bid();
        double ba = lob.best_ask();
        double mid_val = lob.mid();
        if (std::isnan(mid_val)) mid_val = anchor;

        double price;
        if (!std::isnan(bb) && !std::isnan(ba) && (ba - bb) > 0.02) {
            price = (paint_post_side_ == "buy")
                ? round2(bb + 0.01)
                : round2(ba - 0.01);
        } else {
            double jit = std::abs(normal(rng, 0, 0.005));
            price = (paint_post_side_ == "buy")
                ? round2(mid_val - jit)
                : round2(mid_val + jit);
        }

        int64_t oid = lob.submit(A_.name, paint_post_side_, "limit", qty, t, price);
        paint_post_oid_ = oid;
        paint_post_qty_ = qty;
        paint_phase_ = "cross";
        paint_next_t_ = t + uniform(rng, 0.05, 0.20);
    }
    else if (paint_phase_ == "cross" && t >= paint_next_t_) {
        std::string cross_side = (paint_post_side_ == "buy") ? "sell" : "buy";
        int64_t cb_qty = paint_post_qty_;
        lob.submit(B_.name, cross_side, "market", cb_qty, t);

        // Cancel leftover if still resting
        if (paint_post_oid_ >= 0 && lob.by_id.find(paint_post_oid_) != lob.by_id.end()) {
            lob.cancel(paint_post_oid_, t);
        }

        paint_post_side_ = (paint_post_side_ == "buy") ? "sell" : "buy";
        paint_post_oid_ = -1;
        paint_post_qty_ = 0;
        paint_phase_ = "idle";
        paint_next_t_ = t + uniform(rng, 15, 35);
    }
}

// -----------------------------------------------------------------------
// Spoof and execute: A places fake orders, B executes on the opposite side
// -----------------------------------------------------------------------
void ColluderPair::do_spoof(LOB& lob, double t, double anchor,
                            std::mt19937_64& rng) {
    if (spoof_phase_ == "idle" && t >= spoof_next_t_) {
        int64_t n_layers = randint(rng, 2, 6);
        spoof_oids_.clear();
        for (int64_t i = 0; i < n_layers; ++i) {
            int64_t qty = randint(rng, 20, 120);
            double lvl = uniform(rng, 0.01, 0.05) * (i + 1) + normal(rng, 0, 0.004);
            double price = (spoof_side_ == "buy")
                ? round2(anchor - std::abs(lvl))
                : round2(anchor + std::abs(lvl));
            int64_t oid = lob.submit(A_.name, spoof_side_, "limit", qty, t, price);
            spoof_oids_.push_back(oid);
        }
        spoof_phase_ = "spoofed";
        spoof_next_t_ = t + uniform(rng, 0.05, 0.25);
    }
    else if (spoof_phase_ == "spoofed" && t >= spoof_next_t_) {
        std::string exec_side = (spoof_side_ == "buy") ? "sell" : "buy";
        int64_t exec_qty = randint(rng, 30, 150);
        lob.submit(B_.name, exec_side, "market", exec_qty, t);
        spoof_phase_ = "executing";
        spoof_next_t_ = t + uniform(rng, 0.05, 0.35);
    }
    else if (spoof_phase_ == "executing" && t >= spoof_next_t_) {
        // Cancel most/all spoof orders
        std::vector<int64_t> live;
        for (int64_t oid : spoof_oids_) {
            if (lob.by_id.find(oid) != lob.by_id.end()) {
                live.push_back(oid);
            }
        }
        double cancel_frac = uniform(rng, 0.60, 1.00);
        int64_t n_to_cancel = std::max((int64_t)1,
            (int64_t)std::round(cancel_frac * live.size()));

        // Shuffle live oids
        for (int64_t i = (int64_t)live.size() - 1; i > 0; --i) {
            int64_t j = randint(rng, 0, i + 1);
            std::swap(live[i], live[j]);
        }

        for (int64_t i = 0; i < n_to_cancel && i < (int64_t)live.size(); ++i) {
            lob.cancel(live[i], t);
        }

        spoof_side_ = (spoof_side_ == "buy") ? "sell" : "buy";
        spoof_oids_.clear();
        spoof_phase_ = "idle";
        spoof_next_t_ = t + uniform(rng, 10, 30);
    }
}

// -----------------------------------------------------------------------
// Quote mirroring: A builds a ladder, B mirrors it, then both cancel in sync
// -----------------------------------------------------------------------
void ColluderPair::do_mirror(LOB& lob, double t, double anchor,
                             std::mt19937_64& rng) {
    const double TICK = 0.02;

    if (mirror_phase_ == "idle" && t >= mirror_next_t_) {
        mirror_B_build_delay_ = uniform(rng, 0.0, 0.5);
        mirror_sync_lag_ = uniform(rng, 0.1, 1.0);
        mirror_phase_ = "build_A";
        mirror_next_t_ = t;
    }
    else if (mirror_phase_ == "build_A" && t >= mirror_next_t_) {
        int64_t n_layers = randint(rng, 3, 6);
        mirror_A_oids_.clear();
        for (int64_t i = 0; i < n_layers; ++i) {
            int64_t qty = randint(rng, 30, 80);
            double off = TICK * (i + 1) + uniform(rng, -TICK / 2, TICK / 2);
            double price = (mirror_side_ == "buy")
                ? round2(anchor - off)
                : round2(anchor + off);
            int64_t oid = lob.submit(A_.name, mirror_side_, "limit", qty, t, price);
            mirror_A_oids_.push_back(oid);
        }
        mirror_phase_ = "build_B";
        mirror_next_t_ = t + mirror_B_build_delay_;
    }
    else if (mirror_phase_ == "build_B" && t >= mirror_next_t_) {
        int64_t n_layers = randint(rng, 3, 6);
        mirror_B_oids_.clear();
        for (int64_t i = 0; i < n_layers; ++i) {
            int64_t qty = randint(rng, 30, 80);
            double off = TICK * (i + 1) + TICK / 2 + uniform(rng, -TICK / 2, TICK / 2);
            double price = (mirror_side_ == "buy")
                ? round2(anchor - off)
                : round2(anchor + off);
            int64_t oid = lob.submit(B_.name, mirror_side_, "limit", qty, t, price);
            mirror_B_oids_.push_back(oid);
        }
        mirror_phase_ = "hold";
        mirror_next_t_ = t + uniform(rng, 5, 15);
    }
    else if (mirror_phase_ == "hold" && t >= mirror_next_t_) {
        mirror_phase_ = "cancel_A";
        mirror_next_t_ = t;
    }
    else if (mirror_phase_ == "cancel_A" && t >= mirror_next_t_) {
        for (int64_t oid : mirror_A_oids_) {
            if (lob.by_id.find(oid) != lob.by_id.end()) {
                lob.cancel(oid, t);
            }
        }
        mirror_A_oids_.clear();
        mirror_phase_ = "cancel_B";
        mirror_next_t_ = t + mirror_sync_lag_;
    }
    else if (mirror_phase_ == "cancel_B" && t >= mirror_next_t_) {
        for (int64_t oid : mirror_B_oids_) {
            if (lob.by_id.find(oid) != lob.by_id.end()) {
                lob.cancel(oid, t);
            }
        }
        mirror_B_oids_.clear();
        mirror_side_ = (mirror_side_ == "buy") ? "sell" : "buy";
        mirror_phase_ = "idle";
        mirror_next_t_ = t + uniform(rng, 20, 40);
    }
}

} // namespace engine
