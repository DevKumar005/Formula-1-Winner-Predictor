import pandas as pd
import pickle
import numpy as np

print("\n" + "="*70)
print("STEP 8: LAS VEGAS 2025 WINNER PREDICTION")
print("="*70)

# Load full cleaned data and features
df_all = pd.read_csv("data/f1_data_cleaned.csv")

# Filter data to 2025 races before Las Vegas GP (Round 21)
season_2025_before_lv = df_all[(df_all['Season'] == 2025) & (df_all['Round'] < 21)]

print(f"\n2025 races before Las Vegas: {season_2025_before_lv['Round'].nunique()} races")

# Calculate updated driver recent form (average finish in last 5 races before LV)
def driver_recent_form(season_data, races_back=5):
    form = {}
    for driver in season_data['FullName'].unique():
        driver_races = season_data[season_data['FullName'] == driver].sort_values(['Season', 'Round'])
        positions = driver_races['Position'].dropna().values
        if len(positions) > 0:
            form[driver] = np.mean(positions[-races_back:])
        else:
            form[driver] = 999  # Large number if no prior races
    return form

driver_form = driver_recent_form(season_2025_before_lv)

# Calculate updated driver win percentage and podium rate before LV
def calculate_rate(season_data, column, condition, default=0):
    rates = {}
    for driver in season_data['FullName'].unique():
        driver_races = season_data[season_data['FullName'] == driver]
        total = len(driver_races)
        if total > 0:
            count = len(driver_races[condition(driver_races[column])])
            rates[driver] = count / total * 100
        else:
            rates[driver] = default
    return rates

driver_win_pct = calculate_rate(season_2025_before_lv, 'Position', lambda pos: pos == 1)
driver_podium_pct = calculate_rate(season_2025_before_lv, 'Position', lambda pos: pos <= 3)

# Calculate team win percentage before LV
teams_2025 = season_2025_before_lv['TeamName'].unique()
team_win_pct = {}
for team in teams_2025:
    team_races = season_2025_before_lv[season_2025_before_lv['TeamName'] == team]
    total = len(team_races)
    wins = len(team_races[team_races['Position'] == 1])
    team_win_pct[team] = (wins / total * 100) if total > 0 else 0

# Calculate driver DNF rate for 2025 before LV
def driver_dnf_rate(season_data):
    dnf_rates = {}
    for driver in season_data['FullName'].unique():
        driver_races = season_data[season_data['FullName'] == driver]
        total = len(driver_races)
        dnfs = len(driver_races[driver_races['Status'] != 'Finished'])
        dnf_rates[driver] = (dnfs / total * 100) if total > 0 else 0
    return dnf_rates

driver_dnf = driver_dnf_rate(season_2025_before_lv)

# Count races competed for experience (2025 before LV)
driver_experience = season_2025_before_lv['FullName'].value_counts().to_dict()

# Manually provide grid positions (example data - update with actual if known)
manual_grid = {
    'Max Verstappen': 1,
    'Lewis Hamilton': 2,
    'Lando Norris': 3,
    'Charles Leclerc': 4,
    'Carlos Sainz': 5,
    'George Russell': 6,
    'Oscar Piastri': 7,
    'Sergio Perez': 8,
    'Fernando Alonso': 9,
    'Nico Hulkenberg': 10,
    # Add more drivers if needed
}

lv_qualifying = pd.DataFrame([
    {'FullName': driver, 'GridPosition': pos}
    for driver, pos in manual_grid.items()
])

print(f"Using manually provided grid positions for {len(lv_qualifying)} drivers.")

# Prepare feature dataframe for prediction
feature_data = []

for _, row in lv_qualifying.iterrows():
    driver = row['FullName']
    grid_pos = row['GridPosition']
    team = df_all.loc[df_all['FullName'] == driver, 'TeamName'].values
    team_name = team[0] if len(team) > 0 else None

    feature_data.append({
        'FullName': driver,
        'GridPosition': grid_pos if not pd.isna(grid_pos) else 20,
        'driver_recent_form': driver_form.get(driver, 999),
        'driver_win_percentage': driver_win_pct.get(driver, 0),
        'driver_podium_rate': driver_podium_pct.get(driver, 0),
        'team_win_percentage': team_win_pct.get(team_name, 0),
        'driver_dnf_rate': driver_dnf.get(driver, 0),
        'driver_races_competed': driver_experience.get(driver, 0)
    })

df_features_lv = pd.DataFrame(feature_data)

print(f"\nPrepared features for {len(df_features_lv)} drivers for Las Vegas GP 2025.")

# Load scaler and model
with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("data/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load feature columns used during training
with open("data/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Reorder columns to match training data exactly
X_lv = df_features_lv[feature_columns]

# Scale features
X_lv_scaled = scaler.transform(X_lv)

# Predict winner probabilities
win_probs = rf_model.predict_proba(X_lv_scaled)[:, 1]

df_features_lv['win_probability'] = win_probs

# Sort by predicted win probability
df_results = df_features_lv.sort_values('win_probability', ascending=False).reset_index(drop=True)

print("\nTop 10 Predicted Drivers for Las Vegas GP 2025:")
print(df_results[['FullName', 'win_probability']].head(10))

# Identify potential dark horses (win% < 5%, but prob > 0.1)
print("\nPotential Dark Horse Drivers (win% < 5%, prob > 0.1):")
dark_horses = df_results[(df_results['driver_win_percentage'] < 5.0) & (df_results['win_probability'] > 0.1)]
print(dark_horses[['FullName', 'driver_win_percentage', 'win_probability']])

# Save predictions
df_results.to_csv("data/las_vegas_2025_predictions.csv", index=False)
print("\n✓ Saved Las Vegas 2025 predictions to data/las_vegas_2025_predictions.csv")
