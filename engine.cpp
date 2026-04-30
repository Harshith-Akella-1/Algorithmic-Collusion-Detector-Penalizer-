#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <string>

enum class Side { Buy, Sell };

struct Order {
    uint64_t id;
    uint32_t trader_id;
    Side side;
    double price;
    uint32_t qty;
    uint64_t entry_time; // Nanoseconds

    Order* next = nullptr;
    Order* prev = nullptr;
};

// --- Refined Market Tape for NN Training ---
class MarketTape {
    std::ofstream tape_file;
public:
    MarketTape(std::string filename) {
        tape_file.open(filename);
        // We log BOTH the entry time of the order and the time of the actual transaction/event
        tape_file << "event_timestamp_ns,event_type,order_id,trader_id,side,price,qty,counterparty_id,order_resting_time_ns\n";
    }

    void record(uint64_t event_ts, std::string type, uint64_t oid, uint32_t tid, Side s, double p, uint32_t q, uint32_t cid = 0, uint64_t resting_ts = 0) {
        tape_file << event_ts << "," << type << "," << oid << "," << tid << "," 
                 << (s == Side::Buy ? "B" : "S") << "," << p << "," << q << "," << cid << "," << resting_ts << "\n";
    }
};

struct Limit {
    double price;
    uint32_t total_volume = 0;
    Order *head = nullptr, *tail = nullptr;

    void add_order(Order* order) {
        if (!head) head = tail = order;
        else { tail->next = order; order->prev = tail; tail = order; }
        total_volume += order->qty;
    }

    void remove_order(Order* order) {
        if (order->prev) order->prev->next = order->next;
        if (order->next) order->next->prev = order->prev;
        if (order == head) head = order->next;
        if (order == tail) tail = order->prev;
        total_volume -= order->qty;
    }
};

class OrderBook {
private:
    std::map<double, Limit*, std::greater<double>> bids;
    std::map<double, Limit*, std::less<double>> asks;
    std::unordered_map<uint64_t, Order*> order_map;
    MarketTape& tape;

    uint64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    }

public:
    OrderBook(MarketTape& t) : tape(t) {}

    void process_limit_order(uint64_t id, uint32_t trader_id, Side side, double price, uint32_t qty) {
        uint64_t ts = now_ns();
        Order* incoming = new Order{id, trader_id, side, price, qty, ts};
        
        if (side == Side::Buy) match(incoming, asks, bids);
        else match(incoming, bids, asks);
    }

    template<typename T1, typename T2>
    void match(Order* incoming, T1& opposite_side, T2& same_side) {
        while (incoming->qty > 0 && !opposite_side.empty()) {
            auto it = opposite_side.begin();
            Limit* limit = it->second;

            if ((incoming->side == Side::Buy && incoming->price >= it->first) ||
                (incoming->side == Side::Sell && incoming->price <= it->first)) {
                
                while (limit->head && incoming->qty > 0) {
                    Order* resting = limit->head;
                    uint32_t trade_qty = std::min(incoming->qty, resting->qty);
                    uint64_t match_time = now_ns();

                    // TRACE: Every trade gets a timestamp and the resting time of the matched order
                    tape.record(match_time, "TRADE", incoming->id, incoming->trader_id, incoming->side, it->first, trade_qty, resting->trader_id, (match_time - resting->entry_time));

                    incoming->qty -= trade_qty;
                    resting->qty -= trade_qty;
                    limit->total_volume -= trade_qty;

                    if (resting->qty == 0) {
                        limit->remove_order(resting);
                        order_map.erase(resting->id);
                        delete resting;
                    }
                }
                if (!limit->head) { opposite_side.erase(it); delete limit; }
            } else break;
        }

        if (incoming->qty > 0) {
            if (same_side.find(incoming->price) == same_side.end()) same_side[incoming->price] = new Limit{incoming->price};
            same_side[incoming->price]->add_order(incoming);
            order_map[incoming->id] = incoming;
            // TRACE: Log the entry of the new resting order
            tape.record(incoming->entry_time, "NEW", incoming->id, incoming->trader_id, incoming->side, incoming->price, incoming->qty);
        } else delete incoming;
    }

    void cancel_order(uint64_t id) {
        if (order_map.find(id) == order_map.end()) return;
        uint64_t cancel_time = now_ns();
        Order* o = order_map[id];
        
        // Determine the side
        if (o->side == Side::Buy) {
            bids[o->price]->remove_order(o);
            if (bids[o->price]->total_volume == 0) bids.erase(o->price);
        } else {
            asks[o->price]->remove_order(o);
            if (asks[o->price]->total_volume == 0) asks.erase(o->price);
        }

        // TRACE: Log the cancellation with resting time (vital for Spoofing detection)
        tape.record(cancel_time, "CANCEL", o->id, o->trader_id, o->side, o->price, o->qty, 0, (cancel_time - o->entry_time));
        
        order_map.erase(id);
        delete o;
    }
};

