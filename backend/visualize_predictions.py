import pandas as pd
import matplotlib.pyplot as plt

print("\n" + "="*70)
print("VISUALIZE LAS VEGAS 2025 PREDICTIONS")
print("="*70)

# Load the predictions
df = pd.read_csv("backend/data//las_vegas_2025_predictions.csv")

# Sort by win probability
df_sorted = df.sort_values('win_probability', ascending=False).reset_index(drop=True)

# Plot top 10 drivers by win probability
top_10 = df_sorted.head(10)
plt.figure(figsize=(14, 7))
bars = plt.bar(top_10['FullName'], top_10['win_probability'], color='#E10600')
plt.title('Top 10 Predicted Drivers for Las Vegas GP 2025', fontsize=16, fontweight='bold')
plt.xlabel('Driver Name', fontsize=14)
plt.ylabel('Predicted Win Probability', fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate bars with probabilities
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2%}",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0,3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("backend/las_vegas_2025_win_probabilities.png", dpi=150)
print("✓ Saved prediction bar chart as 'backend/las_vegas_2025_win_probabilities.png'")

plt.show()
