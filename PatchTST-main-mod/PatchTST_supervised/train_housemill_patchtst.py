import os
import pandas as pd
import subprocess

# ==========================================
# Paths Configuration
# ==========================================
# Data Path
SOURCE_CSV = r"C:\UCL\Dissertation\data\house_mill_integrated_dataset.csv"

# PatchTST Supervised Path
PATCHTST_DIR = r"C:\UCL\Dissertation\PatchTST-main-mod\PatchTST_supervised"

# Prepare Data loader
TARGET_DATASET_DIR = os.path.join(PATCHTST_DIR, "dataset", "housemill")
os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
TARGET_CSV = os.path.join(TARGET_DATASET_DIR, "data.csv")

# ==========================================
# Data Format Adaptation
# ==========================================
print("[INFO] Adapting dataset for PatchTST standard loader...")
try:
    df = pd.read_csv(SOURCE_CSV)
except FileNotFoundError:
    print(f"[ERROR] Could not find {SOURCE_CSV}. Please ensure the ETL pipeline was run.")
    exit(1)

if '_time' in df.columns:
    df.rename(columns={'_time': 'date'}, inplace=True)
elif 'time' in df.columns:
    df.rename(columns={'time': 'date'}, inplace=True)

# Make Sure Target
cols = [c for c in df.columns if c != 'internal_water_m'] + ['internal_water_m']
df = df[cols]

df.to_csv(TARGET_CSV, index=False)

enc_in = len(df.columns) - 1
print(f"[INFO] Dataset adapted successfully. Detected {enc_in} input features.")

# ==========================================
# Command Construction
# ==========================================
cmd = [
    "python", "run_longExp.py",
    "--is_training", "1",
    "--root_path", "./dataset/housemill/",
    "--data_path", "data.csv",
    "--model_id", "HouseMill_336_96",
    "--model", "PatchTST",  # Model
    "--data", "custom",  # Custom Dataset
    "--features", "MS",  # MS = Multivariate input, Single target output
    "--target", "internal_water_m",  # Target

    # Use min instead of "t", so pandas can read it
    "--freq", "min",

    "--seq_len", "336",  # Lookback window (L)
    "--label_len", "96",  # Decoder start token length
    "--pred_len", "96",  # Prediction length (T)

    # PatchTST Configuration
    "--enc_in", str(enc_in),  # Input length
    "--e_layers", "2",  # Encoder layer
    "--n_heads", "4",  # multi-head number
    "--d_model", "32",  # dimension
    "--d_ff", "128",  # FFN dimension
    "--dropout", "0.3",
    "--fc_dropout", "0.3",
    "--head_dropout", "0.0",
    "--patch_len", "16",  # P: patch length
    "--stride", "8",  # S: slide length

    "--des", "Exp",
    "--train_epochs", "50",
    "--patience", "10",  # Early stop
    "--itr", "1",  # repeat number
    "--batch_size", "128",
    "--learning_rate", "0.0003"
]

# ==========================================
# Execution
# ==========================================
print("=" * 60)
print("[SYSTEM] Launching PatchTST Supervised Training...")
print("=" * 60)
print("Command:", " ".join(cmd))
print("=" * 60)

subprocess.run(cmd, cwd=PATCHTST_DIR)