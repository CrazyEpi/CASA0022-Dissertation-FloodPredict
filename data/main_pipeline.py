# main_pipeline.py
import os
from parsers import parse_water_data, parse_rain_data, parse_bodc_tide_data
from processors import process_sonar_data, add_time_and_astro_features


def build_dataset(base_data_folder, min_water_level_m=0.0):
    print("=" * 50)
    print("DATA PIPELINE STARTED")
    print("=" * 50)

    # ---------------------------------------------------------
    # Read data sources
    # ---------------------------------------------------------
    water_df = parse_water_data(base_data_folder)
    rain_df = parse_rain_data(base_data_folder)
    tide_df = parse_bodc_tide_data(base_data_folder, start_date='2023-06-01', end_date='2025-12-31')

    if water_df.empty:
        print("[ERROR] Core water data missing. Aborting.")
        return None

    # ---------------------------------------------------------
    # Transform data and do cleaning
    # ---------------------------------------------------------
    water_df = process_sonar_data(water_df, min_water_level_m=min_water_level_m)

    # ---------------------------------------------------------
    # Merge Data
    # ---------------------------------------------------------
    print("[LOAD] Merging features onto primary timeline...")

    # Left join rainfall data
    if not rain_df.empty:
        water_df = water_df.join(rain_df, how="left").fillna(0.0)

    # Left join sea tide data
    if not tide_df.empty:
        water_df = water_df.join(tide_df, how="left")

    # Interpolation and frame time threshold
    final_df = water_df.interpolate(method="linear")
    final_df = final_df.loc['2023-06-01':'2025-12-31']

    # Add time and moon phase
    final_df = add_time_and_astro_features(final_df)

    print("=" * 50)
    print("PIPELINE EXECUTION SUCCESSFUL!!!!!!!!")
    print(f"Final Shape: {final_df.shape}")
    print(f"Time Range: {final_df.index.min()} to {final_df.index.max()}")
    print("Features Included:")
    for col in final_df.columns:
        print(f"  - {col}")
    print("=" * 50)

    return final_df


if __name__ == "__main__":
    # Water level filter
    TARGET_MIN_WATER_LEVEL = 1.0

    BASE_DATA_FOLDER = "data"

    if os.path.exists(BASE_DATA_FOLDER):
        dataset = build_dataset(BASE_DATA_FOLDER, min_water_level_m=TARGET_MIN_WATER_LEVEL)

        if dataset is not None:
            save_name = "house_mill_integrated_dataset.csv"
            dataset.to_csv(save_name)
            print(f"[SYSTEM] Dataset saved to root directory as: {save_name}")
    else:
        print(
            f"[ERROR] Could not find the relative data folder: '{BASE_DATA_FOLDER}'. Make sure you are running the script from C:\\UCL\\Dissertation\\data")