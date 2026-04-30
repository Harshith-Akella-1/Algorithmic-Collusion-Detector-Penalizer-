// participants.hpp -- Market participants for the LOB simulator.
//
// Direct port of Python simulator.py bots:
//   NoiseTrader  — drift-biased random orders, partial cancel
//   MarketMaker  — passive symmetric quoter
//   ColluderPair — pair of traders with 4 collusion modes + none
//
// All use C++ <random> for reproducible RNG.

#pragma once

#include "lob.hpp"

#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// -----------------------------------------------------------------------
// NoiseTrader: drift-biased random trader with partial cancel
// -----------------------------------------------------------------------
class NoiseTrader {
public:
    NoiseTrader(const std::string& name,
                double rate = 0.6,
                double p_aggressive = 0.3,
                double spread_dev = 0.10,
                double p_cancel = 0.45);

    void act(LOB& lob, double t, double anchor, double drift,
             std::mt19937_64& rng);

    std::string name;

private:
    double rate_;
    double p_aggressive_;
    double spread_dev_;
    double p_cancel_;
    std::vector<int64_t> active_orders_;
};

// -----------------------------------------------------------------------
// MarketMaker: passive symmetric quoter
// -----------------------------------------------------------------------
class MarketMaker {
public:
    MarketMaker(const std::string& name = "MM",
                double half_spread = 0.05,
                int64_t qty = 80,
                double refresh = 3.0);

    void act(LOB& lob, double t, double anchor);

private:
    std::string name_;
    double half_spread_;
    int64_t qty_;
    double refresh_;
    double last_time_;
    int64_t bid_id_;
    int64_t ask_id_;
};

// -----------------------------------------------------------------------
// ColluderPair: pair of traders with one of 5 modes
//   {none, wash, paint, spoof, mirror}
// -----------------------------------------------------------------------
class ColluderPair {
public:
    ColluderPair(const std::string& name_a, const std::string& name_b,
                 const std::string& mode = "none");

    // Schedule the collusion window
    void schedule(double t0, double t1, std::mt19937_64& rng);

    // Step: runs cover trades (NoiseTrader) + scheme actions
    void step(LOB& lob, double t, double anchor, double drift,
              std::mt19937_64& rng);

    std::string mode() const { return mode_; }

private:
    NoiseTrader A_;
    NoiseTrader B_;
    std::string mode_;

    // Collusion window
    double win_start_;
    double win_end_;
    bool   has_window_;

    // State for each scheme (union-like, but just flat fields for simplicity)
    // Wash state
    double wash_next_time_;

    // Paint state
    std::string paint_phase_;      // "idle", "post", "cross"
    double      paint_next_t_;
    std::string paint_post_side_;  // "buy" or "sell"
    int64_t     paint_post_oid_;
    int64_t     paint_post_qty_;

    // Spoof state
    std::string          spoof_phase_;     // "idle", "spoofed", "executing"
    double               spoof_next_t_;
    std::vector<int64_t> spoof_oids_;
    std::string          spoof_side_;      // "buy" or "sell"

    // Mirror state
    std::string          mirror_phase_;    // "idle", "build_A", "build_B", "hold", "cancel_A", "cancel_B"
    double               mirror_next_t_;
    std::string          mirror_side_;     // "buy" or "sell"
    std::vector<int64_t> mirror_A_oids_;
    std::vector<int64_t> mirror_B_oids_;
    double               mirror_sync_lag_;
    double               mirror_B_build_delay_;

    // Scheme dispatch
    void do_wash(LOB& lob, double t, double anchor, std::mt19937_64& rng);
    void do_paint(LOB& lob, double t, double anchor, std::mt19937_64& rng);
    void do_spoof(LOB& lob, double t, double anchor, std::mt19937_64& rng);
    void do_mirror(LOB& lob, double t, double anchor, std::mt19937_64& rng);
};

} // namespace engine
