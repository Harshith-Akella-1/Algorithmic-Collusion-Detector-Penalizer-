#include <vector>
#include <queue>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for automatic std::vector -> Python List conversion

namespace py = pybind11;

// Global counter to guarantee absolute price-time priority without clock collisions
static long long global_order_id = 0;

struct Order {
    int agent_id;
    double price;
    int quantity;
    bool is_buy;
    long long order_id;
};

// Struct to return execution details to Python
struct ExecutionReport {
    int buyer_id;
    int seller_id;
    double price;
    int quantity;
};

// Max-heap for Bids: Highest price first. If prices match, lowest order_id first.
struct BuyComparator {
    bool operator()(const Order& a, const Order& b) const {
        if (a.price != b.price) return a.price < b.price;
        return a.order_id > b.order_id;
    }
};

// Min-heap for Asks: Lowest price first. If prices match, lowest order_id first.
struct SellComparator {
    bool operator()(const Order& a, const Order& b) const {
        if (a.price != b.price) return a.price > b.price;
        return a.order_id > b.order_id;
    }
};

class MatchingEngine {
public:
    // Takes explicit primitive types instead of an Order struct from Python
    // to streamline the Python interface. Returns a list of executions.
    std::vector<ExecutionReport> process_order(int agent_id, double price, int quantity, bool is_buy) {
        Order new_order = {agent_id, price, quantity, is_buy, ++global_order_id};
        std::vector<ExecutionReport> reports;

        if (is_buy) {
            match_buy(new_order, reports);
        } else {
            match_sell(new_order, reports);
        }
       
        return reports;
    }

private:
    std::priority_queue<Order, std::vector<Order>, BuyComparator> buy_book;
    std::priority_queue<Order, std::vector<Order>, SellComparator> sell_book;

    void match_buy(Order& buy, std::vector<ExecutionReport>& reports) {
        while (buy.quantity > 0 && !sell_book.empty()) {
            Order top_sell = sell_book.top();
           
            // If the buy price is lower than the best ask, no match
            if (buy.price < top_sell.price) break;

            // Pop top to modify it
            sell_book.pop();

            int matched_qty = std::min(buy.quantity, top_sell.quantity);
           
            // Trade always executes at the resting order's price (maker price)
            reports.push_back({buy.agent_id, top_sell.agent_id, top_sell.price, matched_qty});

            buy.quantity -= matched_qty;
            top_sell.quantity -= matched_qty;

            // If sell order is partially filled, push it back
            if (top_sell.quantity > 0) {
                sell_book.push(top_sell);
            }
        }
        // If buy order still has quantity, add to book
        if (buy.quantity > 0) {
            buy_book.push(buy);
        }
    }

    void match_sell(Order& sell, std::vector<ExecutionReport>& reports) {
        while (sell.quantity > 0 && !buy_book.empty()) {
            Order top_buy = buy_book.top();
           
            // If the sell price is higher than the best bid, no match
            if (sell.price > top_buy.price) break;

            // Pop top to modify it
            buy_book.pop();

            int matched_qty = std::min(sell.quantity, top_buy.quantity);
           
            // Trade always executes at the resting order's price (maker price)
            reports.push_back({top_buy.agent_id, sell.agent_id, top_buy.price, matched_qty});

            sell.quantity -= matched_qty;
            top_buy.quantity -= matched_qty;

            // If buy order is partially filled, push it back
            if (top_buy.quantity > 0) {
                buy_book.push(top_buy);
            }
        }
        // If sell order still has quantity, add to book
        if (sell.quantity > 0) {
            sell_book.push(sell);
        }
    }
};

PYBIND11_MODULE(trading_engine, m) {
    py::class_<ExecutionReport>(m, "ExecutionReport")
        .def_readonly("buyer_id", &ExecutionReport::buyer_id)
        .def_readonly("seller_id", &ExecutionReport::seller_id)
        .def_readonly("price", &ExecutionReport::price)
        .def_readonly("quantity", &ExecutionReport::quantity);

    py::class_<MatchingEngine>(m, "MatchingEngine")
        .def(py::init<>())
        .def("process_order", &MatchingEngine::process_order,
             py::arg("agent_id"), py::arg("price"), py::arg("quantity"), py::arg("is_buy"));
}