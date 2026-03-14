import matplotlib.pyplot as plt
from main import run_simulation

def plot_results():
    comp_history = run_simulation("competitive", steps=100)
    coll_history = run_simulation("collusive", steps=100)

    plt.figure(figsize=(12, 5))

    # Plot Prices
    plt.subplot(1, 2, 1)
    plt.plot(comp_history[:, 0], label="Competitive P1", color='blue', linestyle='--')
    plt.plot(coll_history[:, 0], label="Collusive P1", color='red')
    plt.axhline(y=20, color='black', label="Marginal Cost")
    plt.title("Price Evolution: Competition vs Collusion")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.legend()

    # Plot Cumulative Rewards
    plt.subplot(1, 2, 2)
    plt.plot(comp_history[:, 2].cumsum(), label="Comp Profit", color='blue', linestyle='--')
    plt.plot(coll_history[:, 2].cumsum(), label="Coll Profit", color='red')
    plt.title("Cumulative Profit")
    plt.xlabel("Step")
    plt.ylabel("Total Profit")
    plt.legend()

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Graph saved as simulation_results.png. Put this in your README.")

if __name__ == "__main__":
    plot_results()
    