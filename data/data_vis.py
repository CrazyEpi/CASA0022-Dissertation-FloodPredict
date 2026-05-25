import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks
import numpy as np


def visualize_flood_events(file_path, window_days=2, top_n_events=2, flood_threshold=4.43):
    """
    Advanced visualization for long-term IoT flood data.
    :param window_days: The +/- X days window around the flood peak.
    :param top_n_events: Number of top flood events to visualize.
    """
    print("[INFO] Loading data for visualization...")
    df = pd.read_csv(file_path, parse_dates=['_time'])
    df.set_index('_time', inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # ==========================================
    # 1. Global Overview (Aggregated for massive data)
    # ==========================================
    print("[INFO] Generating Global Overview...")
    # Resample to Daily Max to prevent line clustering over 2 years
    df_daily_max = df[['internal_water_m', 'lea_height_m', 'silver_tidal_m']].resample('1D').max()

    plt.figure(figsize=(15, 5))
    plt.plot(df_daily_max.index, df_daily_max['internal_water_m'], color='red', label='Daily Max Internal Water (m)')
    plt.axhline(y=flood_threshold, color='black', linestyle=':', label='Flood Threshold (4.43m)')
    plt.title('Global Overview: Daily Maximum Water Levels (2 Years)', fontsize=14)
    plt.ylabel('Height (m)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 2. Feature Correlation Matrix
    # ==========================================
    print("[INFO] Generating Correlation Matrix...")
    plt.figure(figsize=(10, 8))
    # Using a mask to hide the upper triangle for a cleaner academic look
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Global Feature Correlation Matrix', fontsize=15)
    plt.tight_layout()
    plt.show()

    # ==========================================
    # 3. Micro View: Event-Based Peak Slicing
    # ==========================================
    print(f"[INFO] Extracting Top {top_n_events} flood events (+/- {window_days} days)...")

    # Extract water values and find independent peaks
    # distance=96 means peaks must be at least 24 hours apart (96 * 15mins)
    water_levels = df['internal_water_m'].values
    peaks, properties = find_peaks(water_levels, height=flood_threshold, distance=96)

    if len(peaks) == 0:
        print("[WARNING] No flood events found above the threshold!")
        return

    # Sort peaks by height (descending) and get the top N indices
    peak_heights = properties['peak_heights']
    top_peak_indices = peaks[np.argsort(peak_heights)[-top_n_events:]][::-1]

    # Dynamically find if any rainfall data exists in the dataframe
    rain_cols = [c for c in df.columns if 'rain' in c.lower()]

    for i, peak_idx in enumerate(top_peak_indices):
        peak_time = df.index[peak_idx]
        peak_val = water_levels[peak_idx]
        print(f"  -> Plotting Event {i + 1}: {peak_time.strftime('%Y-%m-%d %H:%M')} (Peak: {peak_val:.2f}m)")

        # Slice the window: [peak - X days, peak + X days]
        start_time = peak_time - pd.Timedelta(days=window_days)
        end_time = peak_time + pd.Timedelta(days=window_days)
        df_event = df.loc[start_time:end_time]

        # Create subplots dynamically based on available data
        num_subplots = 3 if not rain_cols else 4
        fig, axes = plt.subplots(num_subplots, 1, figsize=(15, 4 * num_subplots), sharex=True)

        # Plot 1: Tides
        axes[0].plot(df_event.index, df_event['silver_tidal_m'], label='Silvertown Tidal (m)', alpha=0.8)
        axes[0].plot(df_event.index, df_event['tower_tidal_m'], label='Tower Pier Tidal (m)', alpha=0.8, linestyle='--')
        axes[0].set_title(f'Event {i + 1}: External Tidal Levels', fontsize=12)
        axes[0].set_ylabel('Height (m)')
        axes[0].legend(loc='upper right')

        # Plot 2: River
        axes[1].plot(df_event.index, df_event['lea_height_m'], color='green', label='Lea Bridge River (m)')
        axes[1].set_title('Upstream River Flow', fontsize=12)
        axes[1].set_ylabel('Height (m)')
        axes[1].legend(loc='upper right')

        # Plot 3: Internal Water Level
        axes[2].plot(df_event.index, df_event['internal_water_m'], color='red', label='House Mill Level (m)')
        axes[2].axhline(y=flood_threshold, color='black', linestyle=':', label='Floor Level')
        axes[2].fill_between(df_event.index, df_event['internal_water_m'], flood_threshold,
                             where=(df_event['internal_water_m'] > flood_threshold),
                             color='red', alpha=0.3)
        axes[2].plot(peak_time, peak_val, 'ko', markersize=8, label='Flood Peak')  # Mark the peak
        axes[2].set_title('House Mill Internal Water Level', fontsize=12)
        axes[2].set_ylabel('Height (m)')
        axes[2].legend(loc='upper right')

        # Plot 4 (Optional): Rainfall (Bar plot is better for rain)
        if rain_cols:
            # Sum up all available rain stations for a total catchment rain proxy
            total_rain = df_event[rain_cols].sum(axis=1)
            axes[3].bar(df_event.index, total_rain, width=0.01, color='blue', alpha=0.6,
                        label='Aggregated Rainfall (mm)')
            axes[3].set_title('Catchment Rainfall (15-min aggregation)', fontsize=12)
            axes[3].set_ylabel('Rainfall (mm)')
            axes[3].set_xlabel('Time')
            axes[3].legend(loc='upper right')
        else:
            axes[2].set_xlabel('Time')

        plt.suptitle(f'High-Resolution Flood Event Analysis: {peak_time.strftime("%Y-%m-%d")}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Adjust window_days to zoom in/out, and top_n_events to see more floods
    visualize_flood_events('house_mill_integrated_dataset.csv', window_days=2, top_n_events=3)