import pandas as pd

# Load the combined dataset
df = pd.read_csv("backend/data/f1_all_races_combined.csv")

print("\n" + "="*70)
print("F1 DATASET OVERVIEW (2020-2025)")
print("="*70)

print(f"\nTotal race results: {len(df)}")
print(f"Seasons covered: {int(df['Season'].min())} to {int(df['Season'].max())}")
print(f"Total races: {df.groupby(['Season', 'Round']).ngroups}")
print(f"Unique drivers: {df['FullName'].nunique()}")
print(f"Unique teams: {df['TeamName'].nunique()}")

print("\n" + "="*70)
print("DATA QUALITY")
print("="*70)
print("\nMissing values per column:")
print(df.isnull().sum())

print("\n" + "="*70)
print("TOP WINNERS (2020-2025)")
print("="*70)

winners = df[df['Position'] == 1.0]
win_counts = winners['FullName'].value_counts()

print(f"\nTop 15 drivers by race wins:")
for i, (driver, wins) in enumerate(win_counts.head(15).items(), 1):
    print(f"  {i:2d}. {driver:20s} - {wins:3d} wins")

print("\n" + "="*70)
print("LAS VEGAS GRAND PRIX")
print("="*70)

vegas = df[df['RaceName'] == 'Las Vegas Grand Prix']
print(f"\nLas Vegas races in database: {len(vegas)}")

if len(vegas) > 0:
    vegas_winners = vegas[vegas['Position'] == 1.0].sort_values('Season')
    print("\nLas Vegas winners:")
    for _, row in vegas_winners.iterrows():
        print(f"  {int(row['Season'])}: {row['FullName']:20s} ({row['TeamName']})")
        
    print("\nLas Vegas podiums (all time):")
    vegas_podium = vegas[vegas['Position'] <= 3].sort_values(['Season', 'Position'])
    for _, row in vegas_podium.iterrows():
        pos_text = ['Winner', '2nd Place', '3rd Place']
        print(f"  {int(row['Season'])}: #{int(row['Position'])} {row['FullName']:20s}")
else:
    print("No Las Vegas data found!")

print("\n" + "="*70)
print("2024 SEASON TOP DRIVERS")
print("="*70)

season_2024 = df[df['Season'] == 2024]
if len(season_2024) > 0:
    driver_points_2024 = season_2024.groupby('FullName')['Points'].sum().sort_values(ascending=False)
    print("\nTop 10 drivers by points in 2024:")
    for i, (driver, points) in enumerate(driver_points_2024.head(10).items(), 1):
        print(f"  {i:2d}. {driver:20s} - {int(points):3d} points")

print("\n" + "="*70)
