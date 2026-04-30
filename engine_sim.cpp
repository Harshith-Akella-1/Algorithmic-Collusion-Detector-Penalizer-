#include <random>

// --- Market Initializer ---
class MarketInitializer {
public:
    static void seed_market(OrderBook& engine, double mid_price, int depth_levels) {
        std::cout << "Seeding market depth around price: " << mid_price << "...\n";
        
        uint64_t init_oid = 9000000; // Start with a high offset for Init IDs
        uint32_t exchange_id = 0;    // Trader ID 0 represents the "Initial Liquidity"

        for (int i = 1; i <= depth_levels; ++i) {
            // Create a spread of 0.05 per level
            double bid_price = mid_price - (i * 0.05);
            double ask_price = mid_price + (i * 0.05);
            
            // Randomize quantity at each level to simulate a real LOB
            uint32_t bid_qty = 100 + (rand() % 400);
            uint32_t ask_qty = 100 + (rand() % 400);

            // Inject into engine
            engine.process_limit_order(init_oid++, exchange_id, Side::Buy, bid_price, bid_qty);
            engine.process_limit_order(init_oid++, exchange_id, Side::Sell, ask_price, ask_qty);
        }
        std::cout << "Market Seeded. LOB is now 'Warm'.\n";
    }
};

// --- Updated Main Function ---
int main() {
    // 1. Setup Infrastructure
    MarketTape tape("market_data.csv");
    OrderBook engine(tape);

    // 2. WARM-UP PHASE (The "Initial Trades/Orders")
    // We seed the market at $100.00 with 20 levels of depth.
    // This gives RL bots plenty of orders to 'hit' immediately.
    double initial_mid_price = 100.00;
    MarketInitializer::seed_market(engine, initial_mid_price, 20);

    // 3. TRADING PHASE (RL and ZI Bots)
    std::cout << "Starting Simulation Loop...\n";
    
    for (int i = 0; i < 500; ++i) {
        uint32_t trader_id = (i % 10) + 1; // 10 active bots
        
        // Simple ZI logic: trade near the mid_price
        Side side = (rand() % 2 == 0) ? Side::Buy : Side::Sell;
        
        // Bots will 'jitter' their price around the initial seed
        double price_offset = ((rand() % 200) - 100) / 100.0; // -1.00 to +1.00
        double price = initial_mid_price + price_offset;
        uint32_t qty = (rand() % 50) + 10;

        engine.process_limit_order(i, trader_id, side, price, qty);

        // Random Cancellation (important for detector training)
        if (i > 20 && i % 5 == 0) {
            engine.cancel_order(i - 15);
        }
    }

    std::cout << "Simulation complete. Check market_data.csv for the detector's training data.\n";
    return 0;
}