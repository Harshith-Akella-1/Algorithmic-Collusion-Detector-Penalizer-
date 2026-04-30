// market_tape.hpp -- Market tape recorder for CNN-compatible CSV output.
//
// Writes order logs and trade logs to CSV files in the same format
// that Python's simulator.py produces, so prepare_sequences.py can
// consume them directly.
//
// Order log columns: ts,oid,trader,side,type,price,qty
// Trade log columns: ts,buyer,seller,price,qty,aggressor,mid_at_trade

#pragma once

#include "lob.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace engine {

class MarketTape {
public:
    // Export order log to CSV (matches Python order_log DataFrame columns)
    static void write_orders_csv(const std::string& path,
                                  const std::vector<OrderLogEntry>& log);

    // Export trade log to CSV (matches Python trades DataFrame columns)
    static void write_trades_csv(const std::string& path,
                                  const std::vector<Trade>& trades);

    // Export both to a directory (creates <dir>/orders.csv and <dir>/trades.csv)
    static void export_episode(const std::string& dir, int episode_id,
                                const LOB& lob);
};

} // namespace engine
