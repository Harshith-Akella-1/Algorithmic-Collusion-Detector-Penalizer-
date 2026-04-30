// simulation.cpp -- Episode simulation implementation.
//
// Direct port of Python run_episode() from simulator.py.
// Same loop structure: advance anchor by brownian increment,
// all noise traders act, MM acts, colluder pair steps.

#include "simulation.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(d) _mkdir(d)
#else
#include <sys/stat.h>
#define MKDIR(d) mkdir(d, 0755)
#endif

namespace engine {

// -----------------------------------------------------------------------
// Run one episode
// -----------------------------------------------------------------------
EpisodeResult run_episode(LOB& lob,
                           uint64_t seed,
                           const std::string& collusion_type,
                           int n_noise) {
    std::mt19937_64 rng(seed);

    // Seed the book
    lob.submit("seed", "buy", "limit", 100, 0.0, 99.95);
    lob.submit("seed", "sell", "limit", 100, 0.0, 100.05);

    // Create noise traders
    std::vector<NoiseTrader> noise;
    noise.reserve(n_noise);
    for (int i = 0; i < n_noise; ++i) {
        std::string name = "N" + std::to_string(i);
        noise.emplace_back(name);
    }

    // Market maker
    MarketMaker mm;

    // Colluder pair
    ColluderPair pair("CA", "CB", collusion_type);

    double scheme_t_start = std::numeric_limits<double>::quiet_NaN();
    double scheme_t_end = std::numeric_limits<double>::quiet_NaN();

    if (collusion_type != "none") {
        // Per-mode duration ranges (matching Python)
        double scheme_dur;
        if (collusion_type == "wash") {
            std::uniform_real_distribution<double> dur_dist(60, 180);
            scheme_dur = dur_dist(rng);
        } else {
            std::uniform_real_distribution<double> dur_dist(120, 300);
            scheme_dur = dur_dist(rng);
        }

        double max_t0 = SIM_DURATION - scheme_dur - 10;
        std::uniform_real_distribution<double> t0_dist(0.1 * SIM_DURATION, max_t0);
        double t0 = t0_dist(rng);
        double t1 = t0 + scheme_dur;

        pair.schedule(t0, t1, rng);
        scheme_t_start = t0;
        scheme_t_end = t1;
    }

    // Simulation loop
    double anchor = 100.0;
    std::vector<double> history;
    history.reserve(6500);
    history.push_back(100.0);
    const int LOOKBACK = 50;

    double t = 0.0;
    std::normal_distribution<double> anchor_step(0, 0.02);

    while (t < SIM_DURATION) {
        anchor += anchor_step(rng);
        history.push_back(anchor);

        int lookback_idx = std::max(0, (int)history.size() - LOOKBACK);
        double drift = anchor - history[lookback_idx];

        for (auto& n : noise) {
            n.act(lob, t, anchor, drift, rng);
        }
        mm.act(lob, t, anchor);
        pair.step(lob, t, anchor, drift, rng);

        t = std::round((t + SIM_DT) * 1e6) / 1e6;  // round to avoid FP drift
    }

    EpisodeResult result;
    result.episode_id = 0;
    result.collusion_type = collusion_type;
    result.n_orders = (int)lob.order_log.size();
    result.n_trades = (int)lob.trades.size();
    result.scheme_t_start = scheme_t_start;
    result.scheme_t_end = scheme_t_end;

    return result;
}

// -----------------------------------------------------------------------
// Batch generation
// -----------------------------------------------------------------------
void batch_generate(const std::string& output_dir,
                    int episodes_per_class,
                    uint64_t base_seed) {
    MKDIR(output_dir.c_str());

    std::string orders_dir = output_dir + "/orders";
    std::string trades_dir = output_dir + "/trades";
    MKDIR(orders_dir.c_str());
    MKDIR(trades_dir.c_str());

    // Metadata CSV
    std::string meta_path = output_dir + "/episodes.csv";
    std::ofstream meta(meta_path);
    meta << "episode_id,collusion_type,seed,n_orders,n_trades,"
         << "scheme_t_start,scheme_t_end\n";

    int total = N_COLLUSION_TYPES * episodes_per_class;
    int done = 0;

    for (int ci = 0; ci < N_COLLUSION_TYPES; ++ci) {
        const std::string& ct = COLLUSION_TYPES[ci];
        for (int ep = 0; ep < episodes_per_class; ++ep) {
            int episode_id = ci * episodes_per_class + ep;
            uint64_t seed = base_seed + episode_id;

            LOB lob;
            EpisodeResult result = run_episode(lob, seed, ct);
            result.episode_id = episode_id;

            // Export order log
            std::ostringstream opath;
            opath << orders_dir << "/" << episode_id << ".csv";
            MarketTape::write_orders_csv(opath.str(), lob.order_log);

            // Export trades
            std::ostringstream tpath;
            tpath << trades_dir << "/" << episode_id << ".csv";
            MarketTape::write_trades_csv(tpath.str(), lob.trades);

            // Metadata row
            meta << episode_id << ","
                 << ct << ","
                 << seed << ","
                 << result.n_orders << ","
                 << result.n_trades << ",";
            if (std::isnan(result.scheme_t_start)) {
                meta << ",";
            } else {
                meta << result.scheme_t_start;
            }
            meta << ",";
            if (std::isnan(result.scheme_t_end)) {
                meta << "";
            } else {
                meta << result.scheme_t_end;
            }
            meta << "\n";

            ++done;
            if (done % 100 == 0 || done == total) {
                std::cout << "  " << done << "/" << total
                          << " (" << ct << " ep " << ep << ")"
                          << "  orders=" << result.n_orders
                          << "  trades=" << result.n_trades
                          << std::endl;
            }
        }
    }

    meta.flush();
    std::cout << "\nBatch complete. Output in: " << output_dir << std::endl;
    std::cout << "  Episodes CSV: " << meta_path << std::endl;
    std::cout << "  Order logs:   " << orders_dir << "/<id>.csv" << std::endl;
    std::cout << "  Trade logs:   " << trades_dir << "/<id>.csv" << std::endl;
}

} // namespace engine
