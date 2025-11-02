import pandas as pd
import numpy as np
from datetime import datetime

# Load combined data
df = pd.read_csv("data/f1_all_races_combined.csv")

print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# Make a copy to work with
df_features = df.copy()

# Sort by season and round to maintain chronological order
df_features = df_features.sort_values(['Season', 'Round']).reset_index(drop=True)

print("\nOriginal columns:", list(df_features.columns))

# ============================================================
# FEATURE 1: Driver Recent Form (Last 5 races)
# ============================================================
print("\n[1/7] Creating Driver Recent Form feature...")

def calculate_driver_form(df, races_back=5):
    """Calculate average finishing position for last N races"""
    driver_form = {}
    
    for driver in df['FullName'].unique():
        driver_races = df[df['FullName'] == driver].sort_values(['Season', 'Round'])
        
        # Get positions from all races
        positions = driver_races['Position'].dropna().values
        
        if len(positions) > 0:
            # Average of last 5 races (or fewer if driver hasn't done 5 races yet)
            recent_avg = positions[-races_back:].mean()
            driver_form[driver] = recent_avg
        else:
            driver_form[driver] = np.nan
    
    return driver_form

driver_form = calculate_driver_form(df_features)
df_features['driver_recent_form'] = df_features['FullName'].map(driver_form)

print("✓ Added: driver_recent_form")

# ============================================================
# FEATURE 2: Driver Win Percentage
# ============================================================
print("\n[2/7] Creating Driver Win Percentage feature...")

driver_wins = df_features[df_features['Position'] == 1.0]['FullName'].value_counts()
driver_total_races = df_features['FullName'].value_counts()
driver_win_pct = (driver_wins / driver_total_races * 100).fillna(0)

df_features['driver_win_percentage'] = df_features['FullName'].map(driver_win_pct).fillna(0)

print("✓ Added: driver_win_percentage")

# ============================================================
# FEATURE 3: Team Performance (Win Rate)
# ============================================================
print("\n[3/7] Creating Team Performance feature...")

team_wins = df_features[df_features['Position'] == 1.0]['TeamName'].value_counts()
team_total_races = df_features['TeamName'].value_counts()
team_win_pct = (team_wins / team_total_races * 100).fillna(0)

df_features['team_win_percentage'] = df_features['TeamName'].map(team_win_pct).fillna(0)

print("✓ Added: team_win_percentage")

# ============================================================
# FEATURE 4: Qualifying to Race Performance (Grid position helps)
# ============================================================
print("\n[4/7] Creating Grid Position Impact feature...")

# If driver qualified well, they're more likely to win
df_features['starting_position_quality'] = df_features['GridPosition'].fillna(20)

print("✓ Added: starting_position_quality")

# ============================================================
# FEATURE 5: DNF (Did Not Finish) Rate
# ============================================================
print("\n[5/7] Creating DNF Rate feature...")

def calculate_dnf_rate(df):
    dnf_rates = {}
    for driver in df['FullName'].unique():
        driver_races = df[df['FullName'] == driver]
        total_races = len(driver_races)
        dnf_count = len(driver_races[driver_races['Status'] != 'Finished'])
        dnf_rate = (dnf_count / total_races * 100) if total_races > 0 else 0
        dnf_rates[driver] = dnf_rate
    return dnf_rates

dnf_rates = calculate_dnf_rate(df_features)
df_features['driver_dnf_rate'] = df_features['FullName'].map(dnf_rates).fillna(0)

print("✓ Added: driver_dnf_rate")

# ============================================================
# FEATURE 6: Podium Rate (Top 3 finishes)
# ============================================================
print("\n[6/7] Creating Podium Rate feature...")

def calculate_podium_rate(df):
    podium_rates = {}
    for driver in df['FullName'].unique():
        driver_races = df[df['FullName'] == driver]
        total_races = len(driver_races)
        podium_count = len(driver_races[driver_races['Position'] <= 3])
        podium_rate = (podium_count / total_races * 100) if total_races > 0 else 0
        podium_rates[driver] = podium_rate
    return podium_rates

podium_rates = calculate_podium_rate(df_features)
df_features['driver_podium_rate'] = df_features['FullName'].map(podium_rates).fillna(0)

print("✓ Added: driver_podium_rate")

# ============================================================
# FEATURE 7: Races Competed (Experience)
# ============================================================
print("\n[7/7] Creating Driver Experience feature...")

races_competed = df_features['FullName'].value_counts()
df_features['driver_races_competed'] = df_features['FullName'].map(races_competed).fillna(0)

print("✓ Added: driver_races_competed")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("NEW FEATURES CREATED")
print("="*70)

new_features = [
    'driver_recent_form',
    'driver_win_percentage',
    'team_win_percentage',
    'starting_position_quality',
    'driver_dnf_rate',
    'driver_podium_rate',
    'driver_races_competed'
]

print("\nNew columns added:")
for i, feat in enumerate(new_features, 1):
    print(f"  {i}. {feat}")

print(f"\nTotal columns now: {len(df_features.columns)}")

# Save the engineered features
output_file = "data/f1_features_engineered.csv"
df_features.to_csv(output_file, index=False)
print(f"\n✓ Saved engineered features to: {output_file}")

# Show sample with new features
print("\n" + "="*70)
print("SAMPLE DATA WITH FEATURES")
print("="*70)

sample_cols = ['FullName', 'TeamName', 'Position', 'GridPosition', 
               'driver_recent_form', 'driver_win_percentage', 
               'team_win_percentage', 'driver_podium_rate']
print(df_features[sample_cols].head(15))

print("\nFeature statistics:")
print(df_features[new_features].describe())
