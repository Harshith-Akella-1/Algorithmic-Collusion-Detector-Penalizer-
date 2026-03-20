import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def run_sheriff():
    try:
        data = pd.read_csv("market_tape.csv", names=['p1', 'p2', 'r1', 'r2'])
    except FileNotFoundError:
        print("Error: market_tape.csv not found. Run main.py first.")
        return

    # Feature Engineering: Looking for lockstep price movements
    data['rolling_corr'] = data['p1'].rolling(window=10).corr(data['p2']).fillna(0)
    data['price_spread'] = np.abs(data['p1'] - data['p2'])
    
    # Detect Anomaly: Low spread + High correlation = Potential Collusion
    features = data[['rolling_corr', 'price_spread']]
    model = IsolationForest(contamination=0.05, random_state=42)
    data['is_collusive'] = model.fit_predict(features)
    
    collusions = data[data['is_collusive'] == -1]
    print(f"Analysis Finished. Found {len(collusions)} suspicious trading windows.")
    return collusions

if __name__ == "__main__":
    run_sheriff()