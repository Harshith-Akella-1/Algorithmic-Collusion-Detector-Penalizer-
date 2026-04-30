// lob.cpp -- LOB matching implementation.
//
// Matches Python simulator.py LOB semantics exactly:
//   - Price-time priority
//   - Market orders match against opposite book at best price
//   - Limit orders cross if price allows, then rest
//   - by_id map tracks live resting orders for O(1) cancel lookup

#include "lob.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace engine {

// -----------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------
LOB::LOB() : next_oid_(1) {
    // Reserve reasonable capacity to avoid early reallocations
    bids.reserve(256);
    asks.reserve(256);
    order_log.reserve(4096);
    trades.reserve(512);
}

// -----------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------
double LOB::best_bid() const {
    return bids.empty() ? std::numeric_limits<double>::quiet_NaN() : bids.front().price;
}

double LOB::best_ask() const {
    return asks.empty() ? std::numeric_limits<double>::quiet_NaN() : asks.front().price;
}

double LOB::mid() const {
    double bb = best_bid();
    double ba = best_ask();
    if (std::isnan(bb) || std::isnan(ba)) return std::numeric_limits<double>::quiet_NaN();
    return (bb + ba) / 2.0;
}

// -----------------------------------------------------------------------
// Rebuild by_id pointers (needed after vector reallocation)
// -----------------------------------------------------------------------
void LOB::rebuild_by_id() {
    by_id.clear();
    for (auto& o : bids) by_id[o.oid] = &o;
    for (auto& o : asks) by_id[o.oid] = &o;
}

// -----------------------------------------------------------------------
// Insert a limit order into sorted position
// -----------------------------------------------------------------------
void LOB::insert_limit(Order& o) {
    if (o.side == Side::BUY) {
        // Bids: sorted by (-price, ts) i.e. highest price first, then earliest ts
        auto pos = std::lower_bound(bids.begin(), bids.end(), o,
            [](const Order& a, const Order& b) {
                if (a.price != b.price) return a.price > b.price;  // descending price
                return a.ts < b.ts;  // ascending time
            });
        bids.insert(pos, o);
    } else {
        // Asks: sorted by (price, ts) i.e. lowest price first, then earliest ts
        auto pos = std::lower_bound(asks.begin(), asks.end(), o,
            [](const Order& a, const Order& b) {
                if (a.price != b.price) return a.price < b.price;  // ascending price
                return a.ts < b.ts;  // ascending time
            });
        asks.insert(pos, o);
    }
    // Rebuild all pointers since vector may have reallocated
    rebuild_by_id();
}

// -----------------------------------------------------------------------
// Remove an order from the book
// -----------------------------------------------------------------------
void LOB::remove_from_book(int64_t oid) {
    // Check bids
    for (auto it = bids.begin(); it != bids.end(); ++it) {
        if (it->oid == oid) {
            bids.erase(it);
            rebuild_by_id();
            return;
        }
    }
    // Check asks
    for (auto it = asks.begin(); it != asks.end(); ++it) {
        if (it->oid == oid) {
            asks.erase(it);
            rebuild_by_id();
            return;
        }
    }
}

// -----------------------------------------------------------------------
// Matching engine core
// -----------------------------------------------------------------------
void LOB::match(Order& incoming, double ts, bool is_market) {
    // Determine opposite book
    std::vector<Order>& opp = (incoming.side == Side::BUY) ? asks : bids;

    while (!opp.empty() && incoming.remaining > 0) {
        Order& top = opp.front();

        // Price check for limit orders
        if (!is_market) {
            if (incoming.side == Side::BUY && top.price > incoming.price) break;
            if (incoming.side == Side::SELL && top.price < incoming.price) break;
        }

        int64_t fill_qty = std::min(incoming.remaining, top.remaining);

        // Determine buyer/seller
        std::string buyer, seller;
        if (incoming.side == Side::BUY) {
            buyer = incoming.trader;
            seller = top.trader;
        } else {
            buyer = top.trader;
            seller = incoming.trader;
        }

        // Record trade
        Trade trade;
        trade.ts = ts;
        trade.buyer = buyer;
        trade.seller = seller;
        trade.price = top.price;
        trade.qty = fill_qty;
        trade.aggressor = side_to_str(incoming.side);
        trade.mid_at_trade = mid();  // current mid before removing
        trades.push_back(trade);

        incoming.remaining -= fill_qty;
        top.remaining -= fill_qty;

        if (top.remaining == 0) {
            int64_t dead_oid = top.oid;
            by_id.erase(dead_oid);
            opp.erase(opp.begin());
            // Don't rebuild_by_id here; we'll do it after the matching loop
            // since we may erase multiple orders. But we need to keep by_id consistent.
            // Actually since we erased from front, pointers to remaining orders shifted.
            // Rebuild now to be safe.
            rebuild_by_id();
        }
    }
}

// -----------------------------------------------------------------------
// Submit an order
// -----------------------------------------------------------------------
int64_t LOB::submit(const std::string& trader, const std::string& side_str,
                     const std::string& otype, int64_t qty, double ts, double price) {
    int64_t oid = next_oid_++;
    Side side = str_to_side(side_str);
    bool is_market = (otype == "market");

    double order_price = is_market ? std::numeric_limits<double>::quiet_NaN() : price;

    Order o(oid, trader, side, order_price, qty, ts);

    // Log the order
    OrderLogEntry log;
    log.ts = ts;
    log.oid = oid;
    log.trader = trader;
    log.side = side_str;
    log.type = otype;
    log.price = order_price;
    log.qty = qty;
    order_log.push_back(log);

    // Match against opposite book
    match(o, ts, is_market);

    // If limit order has remaining quantity, insert into book
    if (otype == "limit" && o.remaining > 0) {
        insert_limit(o);
    }

    return oid;
}

// -----------------------------------------------------------------------
// Cancel an order
// -----------------------------------------------------------------------
bool LOB::cancel(int64_t oid, double ts) {
    auto it = by_id.find(oid);
    if (it == by_id.end()) return false;

    Order* o = it->second;

    // Log the cancellation
    OrderLogEntry log;
    log.ts = ts;
    log.oid = o->oid;
    log.trader = o->trader;
    log.side = side_to_str(o->side);
    log.type = "cancel";
    log.price = o->price;
    log.qty = o->remaining;
    order_log.push_back(log);

    // Remove from book
    remove_from_book(oid);

    return true;
}

} // namespace engine
