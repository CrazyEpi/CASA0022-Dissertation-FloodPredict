import pandas as pd
import glob
import os
from scipy.ndimage import median_filter


def house_mill_data_pipeline():

    # enter input file path
    folder_path = input("Please enter data file path: ").strip()

    if not os.path.exists(folder_path):
        print(f"ERROR: Can't Find '{folder_path}', Please check the path and try again.")
        return None

    # Read csv / excel file
    all_files = glob.glob(os.path.join(folder_path, "*.csv")) + \
                glob.glob(os.path.join(folder_path, "*.xlsx"))

    if not all_files:
        print("ERROR: NO CSV or Excel File")
        return None

    print(f"Loaded File Number: {len(all_files)} ...")

    li = []
    for filename in all_files:
        # Skip matadata
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, comment='#')
        else:
            df = pd.read_excel(filename)
        li.append(df)

    raw_df = pd.concat(li, axis=0, ignore_index=True)

    # Resolve Date
    print("Resolving Date...")
    raw_df['_time'] = pd.to_datetime(raw_df['_time'], format='mixed', utc=True)
    raw_df = raw_df.sort_values('_time')

    # Transfer data to columns
    pivot_df = raw_df.pivot_table(index='_time', columns='water-height', values='_value')

    # Names
    name_map = {
        'personal/ucjtdjw/housemill/sonar/distance': 'sonar_dist_mm',
        'personal/ucjtdjw/housemill/LeaBridgeRiver/height': 'lea_height_m',
        'personal/ucjtdjw/housemill/SilvertownTidal/height': 'silver_tidal_m',
        'personal/ucjtdjw/housemill/TowerPierTidal/height': 'tower_tidal_m'
    }
    pivot_df = pivot_df.rename(columns=name_map)

    # Exceptions
    total_anomalies = 0
    if 'sonar_dist_mm' in pivot_df.columns:
        # Sonar Readings > 4800mm
        anomaly_mask = pivot_df['sonar_dist_mm'] > 4800
        total_anomalies = anomaly_mask.sum()

        # Replaced anomalies to 5000 for future filtering
        pivot_df.loc[anomaly_mask, 'sonar_dist_mm'] = 5000

        # Filter glitches
        pivot_df['sonar_dist_mm_cleaned'] = median_filter(
            pivot_df['sonar_dist_mm'].ffill().bfill(), size=5
        )

        # Convert to distance to the ground (sensor range 5m)
        pivot_df['internal_water_m'] = (5000 - pivot_df['sonar_dist_mm_cleaned']) / 1000.0

    # 15min per data
    final_df = pivot_df.resample('15T').mean()
    final_df = final_df.interpolate(method='linear')  # interpolation

    print("-" * 30)
    print(f"FINISHED！")
    print(f"Total Processed Files: {len(all_files)}")
    print(f"Processed Anomolies: {total_anomalies}")
    print(f"Time: {final_df.index.min()} To {final_df.index.max()}")
    print("-" * 30)

    return final_df


if __name__ == "__main__":
    processed_data = house_mill_data_pipeline()
    if processed_data is not None:
        save_name = "house_mill_processed.csv"
        processed_data.to_csv(save_name)
        print(f"Result Saved as: {save_name}")