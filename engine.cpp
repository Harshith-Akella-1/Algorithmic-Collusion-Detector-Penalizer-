#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

struct Order {
    int agent_id;
    double price;
    int quantity;
    bool is_buy;
    long long timestamp;
};

class MatchingEngine {
public:
    void process_order(Order new_order) {
        new_order.timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
        if (new_order.is_buy) match_buy(new_order);
        else match_sell(new_order);
    }

private:
    std::vector<Order> buy_book;
    std::vector<Order> sell_book;

    void match_buy(Order& buy) {
        std::sort(sell_book.begin(), sell_book.end(), [](const Order& a, const Order& b) {
            return (a.price != b.price) ? (a.price < b.price) : (a.timestamp < b.timestamp);
        });

        for (auto it = sell_book.begin(); it != sell_book.end() && buy.quantity > 0; ) {
            if (buy.price >= it->price) {
                int matched = std::min(buy.quantity, it->quantity);
                std::cout << "TRADE: Agent " << buy.agent_id << " bought " << matched
                          << " from Agent " << it->agent_id << " @ " << it->price << std::endl;
                buy.quantity -= matched;
                it->quantity -= matched;
                if (it->quantity == 0) it = sell_book.erase(it); else ++it;
            } else break;
        }
        if (buy.quantity > 0) buy_book.push_back(buy);
    }

    void match_sell(Order& sell) {
        std::sort(buy_book.begin(), buy_book.end(), [](const Order& a, const Order& b) {
            return (a.price != b.price) ? (a.price > b.price) : (a.timestamp < b.timestamp);
        });
        // (Similar logic for selling to the highest bidder...)
        for (auto it = buy_book.begin(); it != buy_book.end() && sell.quantity > 0; ) {
            if (sell.price <= it->price) {
                int matched = std::min(sell.quantity, it->quantity);
                std::cout << "TRADE: Agent " << sell.agent_id << " sold " << matched
                          << " to Agent " << it->agent_id << " @ " << it->price << std::endl;
                sell.quantity -= matched;
                it->quantity -= matched;
                if (it->quantity == 0) it = sell_book.erase(it); else ++it;
            } else break;
        }
        if (sell.quantity > 0) sell_book.push_back(sell);
    }
};