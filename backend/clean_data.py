import pandas as pd
import numpy as np

# Load engineered features
df = pd.read_csv("data/f1_features_engineered.csv")

print("\n" + "="*70)
print("DATA CLEANING")
print("="*70)

print("\nBefore cleaning:")
print(f"  Total rows: {len(df)}")
print(f"  Missing values:\n{df.isnull().sum()}")

# ============================================================
# Handle Missing Values
# ============================================================

print("\n" + "="*70)
print("HANDLING MISSING VALUES")
print("="*70)

# For Position: NaN means DNF (Did Not Finish) - mark as position 999
df['Position'] = df['Position'].fillna(999)
print("✓ Filled Position NaN with 999 (DNF marker)")

# For GridPosition: fill with median (assume they started mid-grid if unknown)
median_grid = df['GridPosition'].median()
df['GridPosition'] = df['GridPosition'].fillna(median_grid)
print(f"✓ Filled GridPosition NaN with median: {median_grid}")

# For Points: fill with 0 (no points if DNF)
df['Points'] = df['Points'].fillna(0)
print("✓ Filled Points NaN with 0")

# ============================================================
# Handle Outliers
# ============================================================

print("\n" + "="*70)
print("HANDLING OUTLIERS")
print("="*70)

# Cap extreme values in rates (should be 0-100%)
feature_cols_to_cap = ['driver_win_percentage', 'team_win_percentage', 
                        'driver_dnf_rate', 'driver_podium_rate']

for col in feature_cols_to_cap:
    df[col] = df[col].clip(0, 100)
    print(f"✓ Capped {col} to 0-100%")

# ============================================================
# Create Binary Target Variable (for machine learning)
# ============================================================

print("\n" + "="*70)
print("CREATE TARGET VARIABLE")
print("="*70)

# Target: Is this driver the race winner? (1 = Yes, 0 = No)
df['is_winner'] = (df['Position'] == 1.0).astype(int)

print("✓ Created is_winner (1 = race winner, 0 = not winner)")
print(f"\n  Winner records: {df['is_winner'].sum()}")
print(f"  Non-winner records: {(df['is_winner'] == 0).sum()}")
print(f"  Class balance: {df['is_winner'].sum() / len(df) * 100:.2f}% winners")

# ============================================================
# Data Type Optimization
# ============================================================

print("\n" + "="*70)
print("DATA TYPE OPTIMIZATION")
print("="*70)

# Convert to appropriate types to save memory
df['Season'] = df['Season'].astype('int16')
df['Round'] = df['Round'].astype('int8')
df['Position'] = df['Position'].astype('float32')
df['GridPosition'] = df['GridPosition'].astype('float32')
df['Points'] = df['Points'].astype('float32')
df['is_winner'] = df['is_winner'].astype('int8')

print("✓ Optimized data types for memory efficiency")

# ============================================================
# Final Check
# ============================================================

print("\n" + "="*70)
print("AFTER CLEANING")
print("="*70)

print(f"\nTotal rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"\nMissing values:\n{df.isnull().sum().sum()} total missing values")

if df.isnull().sum().sum() == 0:
    print("✓ NO MISSING VALUES - Data is clean!")

# Save cleaned data
output_file = "data/f1_data_cleaned.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Saved cleaned data to: {output_file}")

print("\n" + "="*70)
print("DATA SUMMARY")
print("="*70)
print(df.info())
print("\nFirst few rows:")
print(df.head())
