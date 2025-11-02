import fastf1
import pandas as pd
import os
import time
from f1_schedule import F1_SCHEDULE

# Set up data folder
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

print("Starting F1 data download using manual schedule...")
print("This will take 15-20 minutes...\n")

downloaded_count = 0
skipped_count = 0
current_season = None

for season, rnd, event_name in F1_SCHEDULE:
    try:
        # Print season header when it changes
        if season != current_season:
            print(f"\n{'='*60}")
            print(f"SEASON {season}")
            print(f"{'='*60}")
            current_season = season
        
        # Load race session data
        session = fastf1.get_session(season, rnd, 'R')
        session.load()
        
        # Extract race results
        results = session.results
        
        if results is not None and len(results) > 0:
            # Select important columns
            df = results[['Abbreviation', 'FullName', 'TeamName', 'Position', 'GridPosition', 'Points', 'Status']]
            df['Season'] = season
            df['Round'] = rnd
            df['RaceName'] = session.name
            
            # Save to CSV file
            file_path = f"{data_dir}/f1_{season}_race_{rnd:02d}.csv"
            df.to_csv(file_path, index=False)
            
            print(f"✓ Round {rnd:2d}: {session.name:40s} ({len(df)} drivers)")
            downloaded_count += 1
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.3)
        else:
            print(f"✗ Round {rnd:2d}: {event_name:40s} (No data)")
            skipped_count += 1
            
    except Exception as e:
        print(f"✗ Round {rnd:2d}: {event_name:40s} (Error: {str(e)[:30]})")
        skipped_count += 1

print(f"\n{'='*60}")
print(f"DOWNLOAD SUMMARY")
print(f"{'='*60}")
print(f"✓ Successfully downloaded: {downloaded_count} races")
print(f"✗ Skipped/Failed: {skipped_count} rounds")
print(f"✓ Total files in data folder: {len([f for f in os.listdir(data_dir) if f.endswith('.csv')])}")
print(f"✓ Download complete!")
