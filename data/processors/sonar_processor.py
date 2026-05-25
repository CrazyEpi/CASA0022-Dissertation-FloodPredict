import pandas as pd
from scipy.ndimage import median_filter


def process_sonar_data(df, min_water_level_m=0.0):
    """Denoise, Transform, and Filter Sonar Data"""
    print(f"[PROCESSOR] Cleaning Sonar Data (Min level cut-off: {min_water_level_m}m)...")

    SONAR_MAX_THRESHOLD_MM = 4800
    SONAR_DEFAULT_FALLBACK_MM = 5000
    SENSOR_BASE_HEIGHT_M = 5.0

    if "sonar_dist_mm" not in df.columns:
        return df

    # Sudden Jump
    anomaly_mask = df["sonar_dist_mm"] > SONAR_MAX_THRESHOLD_MM
    df.loc[anomaly_mask, 'sonar_dist_mm'] = SONAR_DEFAULT_FALLBACK_MM

    # Remove Glitches
    df["sonar_dist_mm_cleaned"] = median_filter(df["sonar_dist_mm"].ffill().bfill(), size=5)

    # Reverse Sonar Distance to Water Level
    df["internal_water_m"] = SENSOR_BASE_HEIGHT_M - (df["sonar_dist_mm_cleaned"] / 1000.0)

    # Make sure 15 min Interval
    df = df.resample("15min").mean()

    # Denoise with Distance Filter
    if "internal_water_m" in df.columns:
        df["internal_water_m"] = df["internal_water_m"].clip(lower=min_water_level_m)

    return df