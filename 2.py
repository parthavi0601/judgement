import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.read_csv('checkpoints2/training_metrics.csv')

def analyze_trends(df, x_col):
    """
    Calculates the slope of the trend for every numeric column 
    and plots the results.
    """
    # Filter for numeric columns only, excluding the X axis (episode)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove(x_col)
    
    trend_results = []

    # Setup plotting
    n_cols = 2
    n_rows = (len(numeric_cols) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        # Remove NaN values for calculation
        clean_df = df[[x_col, col]].dropna()
        
        # Calculate Linear Regression Slope (Trend Direction)
        # z[0] is the slope, z[1] is the intercept
        z = np.polyfit(clean_df[x_col], clean_df[col], 1)
        slope = z[0]
        
        direction = "Increasing" if slope > 0 else "Decreasing"
        trend_results.append({'Metric': col, 'Slope': slope, 'Trend': direction})

        # Plotting
        sns.regplot(x=x_col, y=col, data=df, ax=axes[i], line_kws={"color": "red", "alpha": 0.5})
        axes[i].set_title(f"{col}\nSlope: {slope:.2e} ({direction})")
        axes[i].grid(True, linestyle='--', alpha=0.6)

    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(trend_results)

# Run the analysis
summary = analyze_trends(df, 'episode')

# Print summary table
print("\n--- Trend Summary ---")
print(summary)
