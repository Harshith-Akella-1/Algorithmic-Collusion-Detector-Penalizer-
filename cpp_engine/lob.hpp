// lob.hpp -- Limit Order Book with price-time priority matching.
//
// Direct port of Python simulator.py LOB class. Same semantics,
// same matching rules, same order/trade log format.
//
// C++14 compatible (no std::optional, no if constexpr).
//
// Usage:
//   LOB book;
//   book.submit("trader1", Side::BUY, OrderType::LIMIT, 50, 0.1, 99.95);
//   book.submit("trader2", Side::SELL, OrderType::MARKET, 50, 0.2);
//   // book.trades now has the fill

#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// -----------------------------------------------------------------------
// Enums
// -----------------------------------------------------------------------
enum class Side : uint8_t { BUY, SELL };
enum class OrderType : uint8_t { LIMIT, MARKET, CANCEL };

// -----------------------------------------------------------------------
// Order: lives in the book's sorted vector. Matches Python Order dataclass.
// -----------------------------------------------------------------------
struct Order {
    int64_t     oid;
    std::string trader;
    Side        side;
    double      price;      // NaN for market orders
    int64_t     qty;
    double      ts;         // simulation time (seconds)
    int64_t     remaining;

    Order() : oid(0), side(Side::BUY), price(0.0), qty(0), ts(0.0), remaining(0) {}
    Order(int64_t id, const std::string& t, Side s, double p, int64_t q, double time)
        : oid(id), trader(t), side(s), price(p), qty(q), ts(time), remaining(q) {}
};

// -----------------------------------------------------------------------
// OrderLogEntry: one row in the order log (matches Python order_log dict)
// -----------------------------------------------------------------------
struct OrderLogEntry {
    double      ts;
    int64_t     oid;
    std::string trader;
    std::string side;    // "buy" or "sell"
    std::string type;    // "limit", "market", or "cancel"
    double      price;
    int64_t     qty;
};

// -----------------------------------------------------------------------
// Trade: one fill (matches Python trades dict)
// -----------------------------------------------------------------------
struct Trade {
    double      ts;
    std::string buyer;
    std::string seller;
    double      price;
    int64_t     qty;
    std::string aggressor;   // "buy" or "sell"
    double      mid_at_trade; // mid price at time of trade
};

// -----------------------------------------------------------------------
// LOB: Limit Order Book
// -----------------------------------------------------------------------
class LOB {
public:
    LOB();

    // Submit an order. Returns the assigned order ID.
    // For market orders, price is ignored (set to NaN internally).
    // For limit orders, price is required.
    int64_t submit(const std::string& trader, const std::string& side_str,
                   const std::string& otype, int64_t qty, double ts,
                   double price = std::numeric_limits<double>::quiet_NaN());

    // Cancel an order by ID. Returns true if found and cancelled.
    bool cancel(int64_t oid, double ts);

    // Best bid/ask. Returns NaN if side is empty.
    double best_bid() const;
    double best_ask() const;

    // Mid price. Returns NaN if either side is empty.
    double mid() const;

    // Public data — matching Python LOB attributes
    std::vector<Order>         bids;       // sorted: highest price first, then by ts
    std::vector<Order>         asks;       // sorted: lowest price first, then by ts
    std::unordered_map<int64_t, Order*> by_id; // oid -> pointer into bids/asks
    std::vector<OrderLogEntry> order_log;
    std::vector<Trade>         trades;

private:
    int64_t next_oid_;

    // Match incoming order against opposite side of the book.
    void match(Order& incoming, double ts, bool is_market);

    // Insert a limit order into the correct position in the book.
    void insert_limit(Order& o);

    // Remove an order from its side of the book.
    void remove_from_book(int64_t oid);

    // Rebuild by_id pointers after vector reallocation.
    void rebuild_by_id();
};

// -----------------------------------------------------------------------
// Utility: side string conversion
// -----------------------------------------------------------------------
inline std::string side_to_str(Side s) { return s == Side::BUY ? "buy" : "sell"; }

inline Side str_to_side(const std::string& s) {
    return (s == "buy" || s == "BUY") ? Side::BUY : Side::SELL;
}

} // namespace engine
