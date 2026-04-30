import matplotlib.pyplot as plt
import pandas as pd
from detector import run_sheriff

def evaluate_and_plot():
    # 1. Run Detection
    run_sheriff()
    
    # 2. Plot results from the Tape
    data = pd.read_csv("market_tape.csv", names=['p1', 'p2', 'r1', 'r2'])

    plt.figure(figsize=(10, 4))
    plt.plot(data['p1'], label="Agent 1 Price")
    plt.plot(data['p2'], label="Agent 2 Price", linestyle='--')
    plt.title("Post-Training Price Evolution")
    plt.legend()
    plt.savefig("evaluation_results.png")
    print("Evaluation graph saved.")

if __name__ == "__main__":
    evaluate_and_plot()



