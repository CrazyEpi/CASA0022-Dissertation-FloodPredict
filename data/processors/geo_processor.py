import numpy as np
import ephem
import pandas as pd


def add_time_and_astro_features(df):
    """Add Trigonometric Time and Moon Phase Data"""
    print("[PROCESSOR] Computing Cyclical Time Encodings and Moon Phases...")

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    moon = ephem.Moon()
    phases = []

    # Use Percentage to Resemble Moon Phase
    for dt in df.index:
        moon.compute(dt)
        phases.append(moon.phase / 100.0)

    df['moon_phase'] = phases

    return df