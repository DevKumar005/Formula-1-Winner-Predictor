import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

print("\n" + "="*70)
print("STEP 4: PREPARE DATA FOR MACHINE LEARNING")
print("="*70)

# Load cleaned data
df = pd.read_csv("data/f1_data_cleaned.csv")

print(f"\nLoaded data: {len(df)} rows, {len(df.columns)} columns")

# ============================================================
# SELECT FEATURES FOR MACHINE LEARNING
# ============================================================

print("\n" + "="*70)
print("SELECTING FEATURES FOR MODELS")
print("="*70)

# Features we'll use to predict (exclude Position, Status, RaceName, etc.)
feature_columns = [
    'GridPosition',              # Starting position
    'driver_recent_form',        # Recent performance
    'driver_win_percentage',     # Historical wins
    'team_win_percentage',       # Team strength
    'driver_dnf_rate',           # Reliability
    'driver_podium_rate',        # Consistency
    'driver_races_competed',     # Experience
]

# Target variable
target_column = 'is_winner'

print(f"\nFeatures selected ({len(feature_columns)}):")
for i, feat in enumerate(feature_columns, 1):
    print(f"  {i}. {feat}")

print(f"\nTarget variable: {target_column}")

# ============================================================
# CREATE FEATURE MATRIX AND TARGET VECTOR
# ============================================================

print("\n" + "="*70)
print("CREATE FEATURE MATRIX & TARGET")
print("="*70)

X = df[feature_columns].copy()
y = df[target_column].copy()

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Target vector y shape: {y.shape}")

print("\nFeature matrix statistics:")
print(X.describe())

print("\nTarget distribution:")
print(f"  Winners (1): {(y == 1).sum()}")
print(f"  Non-winners (0): {(y == 0).sum()}")

# ============================================================
# TRAIN-TEST SPLIT (80-20)
# ============================================================

print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 20% for testing
    random_state=42,         # For reproducibility
    stratify=y               # Keep same class balance in train/test
)

print(f"\nTraining set:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Winners: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")

print(f"\nTesting set:")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_test shape: {y_test.shape}")
print(f"  Winners: {(y_test == 1).sum()} ({(y_test == 1).sum() / len(y_test) * 100:.1f}%)")

# ============================================================
# FEATURE SCALING (Normalize values to 0-1)
# ============================================================

print("\n" + "="*70)
print("FEATURE SCALING (Standardization)")
print("="*70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")
print(f"  Mean of scaled features: {X_train_scaled.mean():.6f} (should be ~0)")
print(f"  Std of scaled features: {X_train_scaled.std():.6f} (should be ~1)")

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_columns)

# ============================================================
# SAVE FOR LATER USE
# ============================================================

print("\n" + "="*70)
print("SAVING PREPARED DATA")
print("="*70)

# Save as CSV
X_train_scaled.to_csv("data/X_train_scaled.csv", index=False)
X_test_scaled.to_csv("data/X_test_scaled.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("✓ Saved training/testing data:")
print("  - X_train_scaled.csv")
print("  - X_test_scaled.csv")
print("  - y_train.csv")
print("  - y_test.csv")

# Save scaler object for later use (when making predictions)
with open("data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✓ Saved scaler.pkl (for scaling new predictions)")

# Also save feature list
with open("data/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_columns, f)

print("✓ Saved feature_columns.pkl")

print("\n" + "="*70)
print("READY FOR MODEL TRAINING!")
print("="*70)
print("\nNext step: Train machine learning models")
