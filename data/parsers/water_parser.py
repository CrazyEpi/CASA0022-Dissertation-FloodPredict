# parsers/water_parser.py
import pandas as pd
import glob
import os


def parse_water_data(base_folder):
    """Parse Water Level Related Data from the Sonar in House Hill"""
    print("[PARSER] Extracting InfluxDB Water Level Data...")

    sonar_folder = os.path.join(base_folder, "sonar")

    all_files = glob.glob(os.path.join(sonar_folder, "*.csv")) + glob.glob(os.path.join(sonar_folder, "*.xlsx"))
    water_dfs = []

    for filename in all_files:
        if filename.endswith('.csv'):
            df = pd.read_csv(filename, comment='#')
        else:
            df = pd.read_excel(filename)
        water_dfs.append(df)

    if not water_dfs:
        print(f"[WARNING] No monthly water level files detected in {sonar_folder}.")
        return pd.DataFrame()

    raw_df = pd.concat(water_dfs, axis=0, ignore_index=True)
    raw_df["_time"] = pd.to_datetime(raw_df["_time"], format="mixed", utc=True)

    # Sort data by time
    pivot_df = raw_df.pivot_table(index="_time", columns="water-height", values="_value")

    name_map = {
        "personal/ucjtdjw/housemill/sonar/distance": "sonar_dist_mm",
        "personal/ucjtdjw/housemill/LeaBridgeRiver/height": "lea_height_m",
        "personal/ucjtdjw/housemill/SilvertownTidal/height": "silver_tidal_m",
        "personal/ucjtdjw/housemill/TowerPierTidal/height": "tower_tidal_m",
    }
    pivot_df = pivot_df.rename(columns=name_map).sort_index()
    return pivot_df