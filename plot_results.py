import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Plot Training Results")
    parser.add_argument("--results", type=str, required=True, help="Path to results CSV file")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.results)
    except FileNotFoundError:
        print(f"Error: Could not find {args.results}")
        return

    # Create a figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot 1: Total Reward over Episodes
    axs[0].plot(df['episode'], df['total_reward'], label='Total Reward', color='blue', alpha=0.7)
    # Add a smoothed trendline
    window = max(1, len(df) // 20)
    axs[0].plot(df['episode'], df['total_reward'].rolling(window=window).mean(), 
                label='Moving Average', color='red', linewidth=2)
    axs[0].set_title('Total Reward per Episode')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Average Zone Temperatures
    axs[1].plot(df['episode'], df['avg_temp_zone0'], label='Living Room', alpha=0.8)
    axs[1].plot(df['episode'], df['avg_temp_zone1'], label='Bedroom', alpha=0.8)
    axs[1].plot(df['episode'], df['avg_temp_zone2'], label='Office', alpha=0.8)
    # Add comfort band highlight
    axs[1].axhspan(21, 24, color='green', alpha=0.1, label='Comfort Band (21-24°C)')
    axs[1].set_title('Average Zone Temperatures per Episode')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Temperature (°C)')
    axs[1].legend()
    axs[1].grid(True)

    # Plot 3: Average Energy Consumption
    axs[2].plot(df['episode'], df['avg_energy'], label='Avg Energy per Step', color='orange')
    axs[2].set_title('Average Energy Consumption per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Energy')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plot_path = args.results.replace('.csv', '.png')
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    # plt.show() # Optional, disabled for automated running

if __name__ == "__main__":
    main()
