# parsers/tide_parser.py
import pandas as pd
import glob
import os


def parse_bodc_tide_data(base_folder, start_date='2023-06-01', end_date='2025-12-31'):
    """Parse Tide Data from BODC Sheerness Site"""
    print("[PARSER] Extracting BODC Tide Data...")

    tide_folder = os.path.join(base_folder, "tide")

    tide_files = glob.glob(os.path.join(tide_folder, "*.txt"))

    if not tide_files:
        print(f"[WARNING] No BODC tide data files found in {tide_folder}.")
        return pd.DataFrame()

    dfs = []
    for file in tide_files:
        skip_idx = 0
        with open(file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if "1)" in line and "/" in line:
                    skip_idx = i
                    break

        df = pd.read_csv(file, skiprows=skip_idx, sep=r'\s+', header=None,
                         usecols=[1, 2, 3], names=['Date', 'Time', 'Elevation'])

        # Delete potential letter in the end of the data
        df['Elevation'] = df['Elevation'].astype(str).str.replace(r'[A-Za-z]', '', regex=True)
        df['Elevation'] = pd.to_numeric(df['Elevation'], errors='coerce')

        df['_time'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], utc=True)
        df.set_index('_time', inplace=True)
        df.drop(columns=['Date', 'Time'], inplace=True)
        dfs.append(df)

    combined_tide = pd.concat(dfs).sort_index()

    # Cut target time and resample
    mask = (combined_tide.index >= pd.to_datetime(start_date, utc=True)) & \
           (combined_tide.index <= pd.to_datetime(end_date, utc=True))

    combined_tide = combined_tide.loc[mask].resample('15min').mean().interpolate(method='linear')
    combined_tide.rename(columns={'Elevation': 'sheerness_tidal_m'}, inplace=True)

    return combined_tide