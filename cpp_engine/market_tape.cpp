// market_tape.cpp -- CSV export implementation.
//
// Outputs match Python simulator.py's DataFrame column names and types
// exactly, so downstream tools (prepare_sequences.py, features.py) work
// without modification.

#include "market_tape.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <direct.h>
#define MKDIR_P(d) _mkdir(d)
#else
#include <sys/stat.h>
#define MKDIR_P(d) mkdir(d, 0755)
#endif

namespace engine {

// Helper: format double with fixed precision, or empty for NaN
static std::string fmt_price(double p) {
    if (std::isnan(p)) return "";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << p;
    return oss.str();
}

static std::string fmt_ts(double ts) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << ts;
    return oss.str();
}

static std::string fmt_mid(double m) {
    if (std::isnan(m)) return "";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << m;
    return oss.str();
}

// -----------------------------------------------------------------------
// Order log CSV
// -----------------------------------------------------------------------
void MarketTape::write_orders_csv(const std::string& path,
                                   const std::vector<OrderLogEntry>& log) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Header matching Python DataFrame columns
    f << "ts,oid,trader,side,type,price,qty\n";

    for (const auto& e : log) {
        f << fmt_ts(e.ts) << ","
          << e.oid << ","
          << e.trader << ","
          << e.side << ","
          << e.type << ","
          << fmt_price(e.price) << ","
          << e.qty << "\n";
    }
    f.flush();
}

// -----------------------------------------------------------------------
// Trade log CSV
// -----------------------------------------------------------------------
void MarketTape::write_trades_csv(const std::string& path,
                                   const std::vector<Trade>& trades) {
    std::ofstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    // Header matching Python DataFrame columns
    f << "ts,buyer,seller,price,qty,aggressor,mid_at_trade\n";

    for (const auto& t : trades) {
        f << fmt_ts(t.ts) << ","
          << t.buyer << ","
          << t.seller << ","
          << fmt_price(t.price) << ","
          << t.qty << ","
          << t.aggressor << ","
          << fmt_mid(t.mid_at_trade) << "\n";
    }
    f.flush();
}

// -----------------------------------------------------------------------
// Export episode data to a directory
// -----------------------------------------------------------------------
void MarketTape::export_episode(const std::string& dir, int episode_id,
                                 const LOB& lob) {
    // Create directory (platform-specific, best-effort)
    MKDIR_P(dir.c_str());

    std::ostringstream orders_path, trades_path;
    orders_path << dir << "/" << episode_id << "_orders.csv";
    trades_path << dir << "/" << episode_id << "_trades.csv";

    write_orders_csv(orders_path.str(), lob.order_log);
    write_trades_csv(trades_path.str(), lob.trades);
}

} // namespace engine
