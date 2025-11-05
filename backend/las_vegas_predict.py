import pandas as pd
import numpy as np
import pickle

print("\n" + "="*70)
print("GENERALIZED LAS VEGAS 2025 WINNER PREDICTION")
print("="*70)

# Load cleaned data (race results 2020-2025)
df_all = pd.read_csv("data/f1_data_cleaned.csv")

# Identify all drivers who raced in 2025
drivers_2025 = df_all[df_all['Season'] == 2025]['FullName'].unique()
print(f"\nDrivers who raced in 2025: {len(drivers_2025)}")

# Extract all Las Vegas GP races in dataset
lv_races = df_all[df_all['RaceName'].str.contains("Las Vegas Grand Prix", case=False, na=False)]
print(f"Total Las Vegas GP race records found: {len(lv_races)}")

# Track specific features for drivers at Las Vegas
lv_stats = lv_races.groupby('FullName').agg(
    lv_race_count = ('Position', 'count'),
    lv_avg_finish = ('Position', 'mean')
).reset_index()
lv_stats['lv_avg_finish'] = lv_stats['lv_avg_finish'].fillna(999)  # Large number if no races

# Season 2025 data before Las Vegas (round 21)
season_2025_before_lv = df_all[(df_all['Season'] == 2025) & (df_all['Round'] < 21)]

# 2025 recent form (avg finish last 5 races)
def recent_form(df, races_back=5):
    form = {}
    for d in df['FullName'].unique():
        dr = df[df['FullName'] == d].sort_values(['Season','Round'])
        positions = dr['Position'].dropna().values
        form[d] = np.mean(positions[-races_back:]) if len(positions) > 0 else 999
    return form
driver_recent_form_2025 = recent_form(season_2025_before_lv)

# 2025 driver win % and podium rate before LV
def calc_rate(df, col, cond):
    rates = {}
    for d in df['FullName'].unique():
        dr = df[df['FullName'] == d]
        total = len(dr)
        if total > 0:
            count = len(dr[cond(dr[col])])
            rates[d] = count / total * 100
        else:
            rates[d] = 0
    return rates
driver_win_pct_2025 = calc_rate(season_2025_before_lv, 'Position', lambda x: x==1)
driver_podium_pct_2025 = calc_rate(season_2025_before_lv, 'Position', lambda x: x<=3)

# 2025 team win % before LV
team_wins = season_2025_before_lv[season_2025_before_lv['Position'] == 1].groupby('TeamName').size()
team_races = season_2025_before_lv.groupby('TeamName').size()
team_win_pct_2025 = (team_wins / team_races * 100).fillna(0)

# 2025 driver DNF rate
def dnf_rate(df):
    rates = {}
    for d in df['FullName'].unique():
        dr = df[df['FullName'] == d]
        total = len(dr)
        dnfs = len(dr[dr['Status'] != 'Finished'])
        rates[d] = (dnfs / total * 100) if total > 0 else 0
    return rates
driver_dnf_rate_2025 = dnf_rate(season_2025_before_lv)

# Driver races competed in 2025 before LV
driver_race_counts_2025 = season_2025_before_lv['FullName'].value_counts().to_dict()

# Estimate grid positions for Las Vegas GP for all 2025 drivers

# Try to get actual grid from Las Vegas race (round 21)
lv_round_21 = df_all[(df_all['Season'] == 2025) & (df_all['Round'] == 21)][['FullName','GridPosition']]

# Average grid positions in 2025 if actual LV grid missing
avg_grid_2025 = season_2025_before_lv.groupby('FullName')['GridPosition'].mean()

grid_positions = {}
for driver in drivers_2025:
    # First try actual LV round 21 grid pos
    val = lv_round_21[lv_round_21['FullName'] == driver]['GridPosition']
    if not val.empty and not pd.isna(val.values[0]):
        grid_positions[driver] = val.values[0]
    else:
        # else average grid position this season before LV
        avg_val = avg_grid_2025.get(driver, np.nan)
        if not pd.isna(avg_val):
            grid_positions[driver] = avg_val
        else:
            # else default to 20
            grid_positions[driver] = 20

# Build feature dataframe for all 2025 drivers
feature_rows = []
for driver in drivers_2025:
    team_row = df_all[df_all['FullName'] == driver]
    team = team_row['TeamName'].iloc[0] if not team_row.empty else None
    feature_rows.append({
        'FullName': driver,
        'GridPosition': grid_positions.get(driver,20),
        'driver_recent_form': driver_recent_form_2025.get(driver, 999),
        'driver_win_percentage': driver_win_pct_2025.get(driver, 0),
        'driver_podium_rate': driver_podium_pct_2025.get(driver, 0),
        'team_win_percentage': team_win_pct_2025.get(team, 0),
        'driver_dnf_rate': driver_dnf_rate_2025.get(driver, 0),
        'driver_races_competed': driver_race_counts_2025.get(driver,0),
        'lv_avg_finish': lv_stats.set_index('FullName')['lv_avg_finish'].get(driver, 999),
        'lv_race_count': lv_stats.set_index('FullName')['lv_race_count'].get(driver, 0)
    })

df_features = pd.DataFrame(feature_rows)

print(f"\nPrepared features for {len(df_features)} drivers for Las Vegas GP 2025.")

# Load scaler, model, and feature columns used during training
with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("data/random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("data/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# Add new track-specific features to the dataframe
# Note: Model was trained on original features only; we can keep lv features for future use
# So only pass features the model expects
X = df_features[feature_columns]

# Scale features
X_scaled = scaler.transform(X)

# Predict using model
win_probabilities = rf_model.predict_proba(X_scaled)[:,1]

df_features['win_probability'] = win_probabilities

# Sort by predicted probability
df_results = df_features.sort_values('win_probability', ascending=False).reset_index(drop=True)

print("\nTop 15 Predicted Drivers for Las Vegas GP 2025:")
print(df_results[['FullName', 'win_probability']].head(15))

# Potential dark horses: low historical win % but decent predicted probability
dark_horses = df_results[(df_results['driver_win_percentage'] < 5.0) & (df_results['win_probability'] > 0.1)]
print("\nPotential Dark Horse Drivers (win% < 5%, prob > 0.1):")
print(dark_horses[['FullName', 'driver_win_percentage', 'win_probability']])

# Save result
df_results.to_csv("data/las_vegas_2025_predictions_general.csv", index=False)
print("\n✓ Saved comprehensive Las Vegas 2025 predictions to data/las_vegas_2025_predictions_general.csv")
