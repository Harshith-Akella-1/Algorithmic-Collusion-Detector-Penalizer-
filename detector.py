import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_market_behavior(csv_path="market_tape.csv", marginal_cost=20.0):
    try:
        # Loading data from your simulation tape
        df = pd.read_csv(csv_path, names=['p1', 'p2', 'r1', 'r2'])
    except Exception:
        print("Tape not found. Run your main.py training first.")
        return

    # 1. Feature: Price Synchronicity (Correlation)
    # Are they moving together?
    df['sync'] = df['p1'].rolling(window=20).corr(df['p2']).fillna(0)

    # 2. Feature: Price Magnitude (The 'Greed' Factor)
    # Collusion requires prices to be significantly above marginal cost (20.0)
    avg_price = (df['p1'] + df['p2']) / 2
    df['price_premium'] = (avg_price - marginal_cost).clip(lower=0)

    # 3. Feature: Joint Profitability
    # In competition, joint profit is near 0. In collusion, it is high.
    df['total_profit'] = df['r1'] + df['r2']

    # 4. THE COLLUSION SCORE (0.0 to 1.0)
    # Logic: Sync * Premium * Profit
    # If any of these are 0 (e.g. price is 20), the score drops to 0.
    scaler = MinMaxScaler()
    metrics = df[['sync', 'price_premium', 'total_profit']].fillna(0)
    scaled_metrics = scaler.fit_transform(metrics)
    
    # Weighted average: 40% Sync, 30% Price level, 30% Profit
    df['collusion_score'] = (scaled_metrics[:, 0] * 0.4 + 
                             scaled_metrics[:, 1] * 0.3 + 
                             scaled_metrics[:, 2] * 0.3)

    # Threshold: Anything above 0.7 is highly suspicious
    suspicious_activity = df[df['collusion_score'] > 0.7]
    
    print(f"--- Analysis Results ---")
    print(f"Total steps analyzed: {len(df)}")
    print(f"Average Market Price: {avg_price.mean():.2f}")
    print(f"Collusive episodes detected: {len(suspicious_activity)}")
    
    if len(suspicious_activity) == 0 and avg_price.mean() <= marginal_cost + 1:
        print("CONCLUSION: Market is currently COMPETITIVE (Bots are undercutting to marginal cost).")
    elif len(suspicious_activity) > 0:
        print("CONCLUSION: TACIT COLLUSION DETECTED (Bots are maintaining high prices in sync).")

    return df

if __name__ == "__main__":
    analyze_market_behavior()