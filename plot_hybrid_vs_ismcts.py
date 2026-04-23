import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

def plot_results(log_dir='logs'):
    files = glob.glob(os.path.join(log_dir, 'ismcts_comp_*s.csv'))
    if not files:
        print("No log files found.")
        return

    all_data = []
    for f in files:
        # Skip empty files
        if os.path.getsize(f) == 0:
            continue
        try:
            df = pd.read_csv(f)
            if df.empty:
                continue
            budget = float(os.path.basename(f).split('_')[-1].replace('s.csv', ''))
            
            # Group by agent_type and calculate mean win and payoff
            summary = df.groupby('agent_type').agg({
                'win': 'mean',
                'payoff': 'mean',
                'sim_time': 'mean'
            }).reset_index()
            summary['budget'] = budget
            all_data.append(summary)
        except Exception as e:
            print(f"Error processing {f}: {e}")

    if not all_data:
        print("No valid data to plot.")
        return

    results_df = pd.concat(all_data).sort_values('budget')
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for agent in results_df['agent_type'].unique():
        agent_data = results_df[results_df['agent_type'] == agent]
        axes[0].plot(agent_data['budget'], agent_data['win'], marker='o', label=f'Agent: {agent}')
        axes[1].plot(agent_data['budget'], agent_data['payoff'], marker='s', label=f'Agent: {agent}')

    axes[0].set_title('Bid-Hit Accuracy (Win Rate) vs Time Budget')
    axes[0].set_xlabel('Time Budget (s)')
    axes[0].set_ylabel('Win Rate (%)')
    axes[0].set_xscale('log')
    axes[0].grid(True, which="both", ls="-", alpha=0.5)
    axes[0].legend()

    axes[1].set_title('Average payoff vs Time Budget')
    axes[1].set_xlabel('Time Budget (s)')
    axes[1].set_ylabel('Avg payoff')
    axes[1].set_xscale('log')
    axes[1].grid(True, which="both", ls="-", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(log_dir, 'hybrid_vs_ismcts_comparison.png')
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_results()
