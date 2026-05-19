import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_house_mill_data(file_path):
    # Data loader
    df = pd.read_csv(file_path, parse_dates=['_time'])
    df.set_index('_time', inplace=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    # Silvertown vs Tower Pier Tide
    axes[0].plot(df.index, df['silver_tidal_m'], label='Silvertown Tidal (m)', alpha=0.8)
    axes[0].plot(df.index, df['tower_tidal_m'], label='Tower Pier Tidal (m)', alpha=0.8, linestyle='--')
    axes[0].set_title('External Tidal Levels (Thames PLA Data)', fontsize=14)
    axes[0].set_ylabel('Height (m)')
    axes[0].legend(loc='upper right')

    # Lea Bridge River Flow
    axes[1].plot(df.index, df['lea_height_m'], color='green', label='Lea Bridge River Height (m)')
    axes[1].set_title('Upstream River Flow (Fluvial Component)', fontsize=14)
    axes[1].set_ylabel('Height (m)')
    axes[1].legend(loc='upper right')

    # House Mill Water Level V.S. Warning Level
    axes[2].plot(df.index, df['internal_water_m'], color='red', label='House Mill Internal Level (m)')

    flood_threshold = 4.43
    axes[2].axhline(y=flood_threshold, color='black', linestyle=':', label='Floor Level (Threshold)')

    axes[2].fill_between(df.index, df['internal_water_m'], flood_threshold,
                         where=(df['internal_water_m'] > flood_threshold),
                         color='red', alpha=0.3, label='Flooding Event')

    axes[2].set_title('House Mill Internal Water Level & Flood Threshold', fontsize=14)
    axes[2].set_ylabel('Height (m)')
    axes[2].set_xlabel('Time')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plt.show()

    # Heat map between features
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='RdBu_r', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()


if __name__ == "__main__":
    visualize_house_mill_data('house_mill_processed.csv')