import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("backend/data/f1_all_races_combined.csv")

# Get top winners
winners = df[df['Position'] == 1.0]
win_counts = winners['FullName'].value_counts().head(12)

# Create bar chart
plt.figure(figsize=(14, 6))
colors = ['#E10600' if i == 0 else '#0082FA' if i == 1 else '#FFA800' if i == 2 else '#1E3050' 
          for i in range(len(win_counts))]
win_counts.plot(kind='bar', color=colors)
plt.title('Top F1 Drivers by Race Wins (2020-2025)', fontsize=16, fontweight='bold')
plt.xlabel('Driver Name', fontsize=12)
plt.ylabel('Number of Wins', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the chart
plt.savefig('src/top_drivers_wins.png', dpi=150, bbox_inches='tight')
print("✓ Chart saved as 'src/top_drivers_wins.png'")
plt.close()

# Create Las Vegas specific chart
vegas = df[df['RaceName'] == 'Las Vegas Grand Prix']
if len(vegas) > 0:
    vegas_winners = vegas[vegas['Position'] == 1.0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    vegas_data = vegas_winners.groupby('Season').size()
    
    ax.bar(vegas_data.index, vegas_data.values, color='#E10600', edgecolor='black', linewidth=2)
    ax.set_title('Las Vegas Grand Prix - Winners by Year', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Races', fontsize=12)
    ax.set_xticks(vegas_data.index)
    
    plt.tight_layout()
    plt.savefig('src/vegas_winners.png', dpi=150, bbox_inches='tight')
    print("✓ Las Vegas chart saved as 'src/vegas_winners.png'")
    plt.close()

print("\n✓ Visualizations complete!")
