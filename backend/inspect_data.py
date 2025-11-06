import pandas as pd
import numpy as np

# Load combined data
df = pd.read_csv("backend/data/f1_all_races_combined.csv")

print("\n" + "="*70)
print("INSPECT RAW DATA")
print("="*70)

# Show structure
print("\nData shape (rows, columns):", df.shape)
print("\nColumn names and types:")
print(df.dtypes)

print("\n" + "="*70)
print("SAMPLE DATA (First 10 rows)")
print("="*70)
print(df.head(10))

print("\n" + "="*70)
print("DATA STATISTICS")
print("="*70)
print(df.describe())

print("\n" + "="*70)
print("CHECK FOR ISSUES")
print("="*70)

# Missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Unique values per column
print("\nUnique values per column:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")

# Check Position column (should be 1-20 or so)
print("\nPosition values (should be 1, 2, 3, ... or NaN for DNF):")
print(df['Position'].value_counts().sort_index().head(20))

# Check Status column (tells us DNF reasons)
print("\nStatus values (Finished, DNF, etc.):")
print(df['Status'].value_counts())

print("\n" + "="*70)
