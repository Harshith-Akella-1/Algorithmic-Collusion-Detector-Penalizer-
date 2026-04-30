// main.cpp -- CLI driver for the C++ LOB simulator.
//
// Modes:
//   ./engine --mode smoke
//       Run one episode per collusion type, print stats.
//
//   ./engine --mode batch --episodes 1000 --output cpp_dataset
//       Generate episodes_per_class episodes of each type.
//
//   ./engine --mode single --type wash --seed 42
//       Run one specific episode, export to stdout-friendly format.
//
// Build:
//   g++ -std=c++14 -O2 -Wall -Wextra lob.cpp market_tape.cpp
//       participants.cpp simulation.cpp main.cpp -o engine
//
// Or just: make

#include "simulation.hpp"
#include "lob.hpp"
#include "market_tape.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

using namespace engine;

// -----------------------------------------------------------------------
// Smoke test: one episode per type
// -----------------------------------------------------------------------
void run_smoke_test() {
    std::cout << "=== C++ LOB Engine — Smoke Test ===" << std::endl;
    std::cout << "Running one episode per collusion type..." << std::endl;
    std::cout << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N_COLLUSION_TYPES; ++i) {
        const std::string& ct = COLLUSION_TYPES[i];

        auto t0 = std::chrono::high_resolution_clock::now();
        LOB lob;
        EpisodeResult result = run_episode(lob, 42 + i, ct);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Count CA/CB orders (for CNN compatibility check)
        int ca_cb_orders = 0;
        for (const auto& e : lob.order_log) {
            if (e.trader == "CA" || e.trader == "CB") {
                ca_cb_orders++;
            }
        }

        std::cout << "  " << ct;
        // Pad to 8 chars
        for (int p = (int)ct.size(); p < 8; ++p) std::cout << " ";
        std::cout << " | orders=" << result.n_orders
                  << "  trades=" << result.n_trades
                  << "  CA/CB_events=" << ca_cb_orders;
        if (!std::isnan(result.scheme_t_start)) {
            std::cout << "  window=[" << result.scheme_t_start
                      << " -> " << result.scheme_t_end << "]";
        }
        std::cout << "  (" << ms << " ms)" << std::endl;
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << std::endl;
    std::cout << "Total: " << total_ms << " ms" << std::endl;

    // Export the last episode's CSV as a sample
    LOB sample_lob;
    run_episode(sample_lob, 42, "wash");
    MarketTape::write_orders_csv("sample_orders.csv", sample_lob.order_log);
    MarketTape::write_trades_csv("sample_trades.csv", sample_lob.trades);
    std::cout << std::endl;
    std::cout << "Sample wash episode exported:" << std::endl;
    std::cout << "  sample_orders.csv (" << sample_lob.order_log.size() << " events)" << std::endl;
    std::cout << "  sample_trades.csv (" << sample_lob.trades.size() << " trades)" << std::endl;

    // Print first 10 order log entries as format sample
    std::cout << std::endl;
    std::cout << "=== Order Log Sample (first 10 rows) ===" << std::endl;
    std::cout << "ts,oid,trader,side,type,price,qty" << std::endl;
    int n_show = std::min(10, (int)sample_lob.order_log.size());
    for (int i = 0; i < n_show; ++i) {
        const auto& e = sample_lob.order_log[i];
        std::cout << e.ts << "," << e.oid << "," << e.trader << ","
                  << e.side << "," << e.type << "," << e.price << ","
                  << e.qty << std::endl;
    }

    std::cout << std::endl;
    std::cout << "=== Trade Log Sample (first 10 rows) ===" << std::endl;
    std::cout << "ts,buyer,seller,price,qty,aggressor,mid_at_trade" << std::endl;
    n_show = std::min(10, (int)sample_lob.trades.size());
    for (int i = 0; i < n_show; ++i) {
        const auto& t = sample_lob.trades[i];
        std::cout << t.ts << "," << t.buyer << "," << t.seller << ","
                  << t.price << "," << t.qty << "," << t.aggressor << ","
                  << t.mid_at_trade << std::endl;
    }
}

// -----------------------------------------------------------------------
// Single episode mode
// -----------------------------------------------------------------------
void run_single(const std::string& type, uint64_t seed, const std::string& output) {
    LOB lob;
    EpisodeResult result = run_episode(lob, seed, type);

    std::string orders_path = output.empty() ? "orders.csv" : output + "_orders.csv";
    std::string trades_path = output.empty() ? "trades.csv" : output + "_trades.csv";

    MarketTape::write_orders_csv(orders_path, lob.order_log);
    MarketTape::write_trades_csv(trades_path, lob.trades);

    std::cout << "Episode complete:" << std::endl;
    std::cout << "  Type:    " << result.collusion_type << std::endl;
    std::cout << "  Seed:    " << seed << std::endl;
    std::cout << "  Orders:  " << result.n_orders << std::endl;
    std::cout << "  Trades:  " << result.n_trades << std::endl;
    std::cout << "  Output:  " << orders_path << ", " << trades_path << std::endl;
}

// -----------------------------------------------------------------------
// Usage
// -----------------------------------------------------------------------
void print_usage(const char* prog) {
    std::cout << "Usage:" << std::endl;
    std::cout << "  " << prog << " --mode smoke" << std::endl;
    std::cout << "      Run one episode per type, print stats + sample CSVs" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << prog << " --mode batch --episodes N [--output DIR] [--seed S]" << std::endl;
    std::cout << "      Generate N episodes per class (5*N total)" << std::endl;
    std::cout << std::endl;
    std::cout << "  " << prog << " --mode single --type TYPE [--seed S] [--output PREFIX]" << std::endl;
    std::cout << "      Run one episode of TYPE {none,wash,paint,spoof,mirror}" << std::endl;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::string mode = "smoke";
    std::string type = "wash";
    std::string output = "cpp_dataset";
    int episodes = 1000;
    uint64_t seed = 42;

    // Parse args
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            mode = argv[++i];
        else if (std::strcmp(argv[i], "--type") == 0 && i + 1 < argc)
            type = argv[++i];
        else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc)
            output = argv[++i];
        else if (std::strcmp(argv[i], "--episodes") == 0 && i + 1 < argc)
            episodes = std::stoi(argv[++i]);
        else if (std::strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = std::stoull(argv[++i]);
        else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (mode == "smoke") {
        run_smoke_test();
    } else if (mode == "batch") {
        std::cout << "=== Batch Generation ===" << std::endl;
        std::cout << "Episodes per class: " << episodes << std::endl;
        std::cout << "Total episodes:     " << episodes * N_COLLUSION_TYPES << std::endl;
        std::cout << "Output directory:   " << output << std::endl;
        std::cout << "Base seed:          " << seed << std::endl;
        std::cout << std::endl;

        auto t0 = std::chrono::high_resolution_clock::now();
        batch_generate(output, episodes, seed);
        auto t1 = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Batch time: " << secs << " s" << std::endl;
    } else if (mode == "single") {
        run_single(type, seed, output);
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
