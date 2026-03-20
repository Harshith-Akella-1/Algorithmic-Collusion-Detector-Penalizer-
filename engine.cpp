#include <iostream>
#include <vector>
#include <algorithm>

struct Order {
    int agent_id;
    double price;
    int quantity;
    bool is_buy;
};

class MatchingEngine {
public:
    // Core Logic: Price-Time Priority
    void process_order(Order new_order) {
        if (new_order.is_buy) {
            match_against_sells(new_order);
        } else {
            match_against_buys(new_order);
        }
    }

private:
    std::vector<Order> buy_book;
    std::vector<Order> sell_book;

    void match_against_sells(Order& buy) {
        // Sort sells: lowest price first
        std::sort(sell_book.begin(), sell_book.end(), [](const Order& a, const Order& b) {
            return a.price < b.price;
        });
        // Matching logic...
    }
   
    void match_against_buys(Order& sell) {
        // Sort buys: highest price first
        std::sort(buy_book.begin(), buy_book.end(), [](const Order& a, const Order& b) {
            return a.price > b.price;
        });
        // Matching logic...
    }
};