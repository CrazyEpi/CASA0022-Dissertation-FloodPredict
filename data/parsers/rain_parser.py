# parsers/rain_parser.py
import pandas as pd
import glob
import os


def parse_rain_data(base_folder):
    """Parse Rainfall Data from 4 Environment Agency Sites"""
    print("[PARSER] Extracting EA Rainfall Data...")

    rain_folder = os.path.join(base_folder, "rain")

    files = glob.glob(os.path.join(rain_folder, "*.csv"))
    rainfall_dfs = []

    for filename in files:
        base_name = os.path.basename(filename)
        # delete "rainfall" in filename
        station_name = base_name.split("-rainfall")[0].lower().replace("-", "_") + "_rain_mm"

        try:
            rain_df = pd.read_csv(filename)
            time_col = [c for c in rain_df.columns if "time" in c.lower() or "date" in c.lower()][0]
            val_col = [c for c in rain_df.columns if "value" in c.lower() or "val" in c.lower()][0]

            rain_clean = pd.DataFrame()
            rain_clean["_time"] = pd.to_datetime(rain_df[time_col], format="mixed", utc=True)
            rain_clean[station_name] = pd.to_numeric(rain_df[val_col], errors="coerce").fillna(0.0)

            # 15 min merge
            rain_clean.set_index("_time", inplace=True)
            rain_clean = rain_clean.resample("15min").sum()
            rainfall_dfs.append(rain_clean)
            print(f"  -> Extracted: {station_name}")

        except Exception as e:
            print(f"[WARNING] Skipping {base_name}: {e}")

    if not rainfall_dfs:
        print(f"[WARNING] No rainfall files detected in {rain_folder}.")
        return pd.DataFrame()

    # merge 4 sites together
    merged_rain = rainfall_dfs[0]
    for i in range(1, len(rainfall_dfs)):
        merged_rain = merged_rain.join(rainfall_dfs[i], how="outer")

    return merged_rain