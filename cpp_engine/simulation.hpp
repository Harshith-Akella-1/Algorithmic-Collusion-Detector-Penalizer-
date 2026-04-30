// simulation.hpp -- Episode simulation runner.
//
// Port of Python run_episode(). Runs one 600s episode with noise traders,
// market maker, and an optional colluder pair. Exports CNN-compatible data.

#pragma once

#include "lob.hpp"
#include "market_tape.hpp"
#include "participants.hpp"

#include <string>
#include <random>

namespace engine {

// Simulation time constants (matching Python)
constexpr double SIM_DT = 0.1;
constexpr double SIM_DURATION = 600.0;

// Collusion type names
const std::string COLLUSION_TYPES[] = {"none", "wash", "paint", "spoof", "mirror"};
constexpr int N_COLLUSION_TYPES = 5;

// Result of running one episode
struct EpisodeResult {
    int    episode_id;
    std::string collusion_type;
    int    n_orders;
    int    n_trades;
    double scheme_t_start;   // NaN if no scheme
    double scheme_t_end;     // NaN if no scheme
};

// -----------------------------------------------------------------------
// Run one episode. Returns the LOB (with order_log and trades populated)
// and metadata about the episode.
// -----------------------------------------------------------------------
EpisodeResult run_episode(LOB& lob,
                           uint64_t seed,
                           const std::string& collusion_type = "none",
                           int n_noise = 15);

// -----------------------------------------------------------------------
// Batch run: generates N episodes, exports to output_dir
// -----------------------------------------------------------------------
void batch_generate(const std::string& output_dir,
                    int episodes_per_class,
                    uint64_t base_seed = 42);

} // namespace engine
