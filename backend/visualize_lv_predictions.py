import pandas as pd
import matplotlib.pyplot as plt

# Load the predictions CSV
df = pd.read_csv("backend/data/las_vegas_2025_predictions_general.csv")

# Sort by win probability
df_sorted = df.sort_values('win_probability', ascending=False)

# Select top 10 drivers
top_drivers = df_sorted.head(10)

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(top_drivers['FullName'], top_drivers['win_probability'], color='#E10600')
plt.title('Top 10 Predicted Drivers for Las Vegas GP 2025', fontsize=16, fontweight='bold')
plt.xlabel('Driver Name', fontsize=14)
plt.ylabel('Win Probability', fontsize=14)
plt.ylim(0, 1)
plt.xticks(rotation=45, ha='right')

# Add annotations with probabilities
for bar in bars:
    height = bar.get_height()
    plt.annotate(f"{height:.2%}", 
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig("backend/top_lv_drivers_probabilities.png", dpi=150)
print("✓ Saved bar chart as 'backend/top_lv_drivers_probabilities.png'")
plt.show()
