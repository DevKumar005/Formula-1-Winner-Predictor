import pandas as pd
import os
import glob

# Read all CSV files from the data folder
data_dir = "data"
all_files = glob.glob(os.path.join(data_dir, "f1_*.csv"))

print(f"Found {len(all_files)} race files")
print("Combining all races into one dataset...\n")

# Combine all files into one dataframe
dfs = []
for file in sorted(all_files):
    try:
        df = pd.read_csv(file)
        dfs.append(df)
        print(f"✓ Loaded: {os.path.basename(file)}")
    except Exception as e:
        print(f"✗ Error reading {os.path.basename(file)}: {e}")

if len(dfs) > 0:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Save the combined file
    output_file = f"{data_dir}/f1_all_races_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Combined all races: {len(combined_df)} total race results")
    print(f"✓ Saved to: {output_file}")
    
    # Show some statistics
    print(f"\nDataset Summary:")
    print(f"  Seasons: {int(combined_df['Season'].min())} to {int(combined_df['Season'].max())}")
    print(f"  Total races: {combined_df.groupby(['Season', 'Round']).ngroups}")
    print(f"  Unique drivers: {combined_df['FullName'].nunique()}")
    print(f"  Unique teams: {combined_df['TeamName'].nunique()}")
else:
    print("✗ No files to combine!")
