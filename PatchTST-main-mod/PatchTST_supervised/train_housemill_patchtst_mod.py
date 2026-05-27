import os
import pandas as pd
import subprocess
from datetime import datetime

# ==========================================
# Paths Configuration
# ==========================================
SOURCE_CSV = r"C:\UCL\Dissertation\data\house_mill_integrated_dataset.csv"
PATCHTST_DIR = r"C:\UCL\Dissertation\PatchTST-main-mod\PatchTST_supervised"

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
    print(f"[ERROR] Could not find {SOURCE_CSV}.")
    exit(1)

if '_time' in df.columns:
    df.rename(columns={'_time': 'date'}, inplace=True)
elif 'time' in df.columns:
    df.rename(columns={'time': 'date'}, inplace=True)

cols = [c for c in df.columns if c != 'internal_water_m'] + ['internal_water_m']
df = df[cols]
df.to_csv(TARGET_CSV, index=False)
enc_in = len(df.columns) - 1

# ==========================================
# Command Construction (Hyperparameters)
# ==========================================
cmd = [
    "python", "-u", "run_longExp.py",
    "--is_training", "1",
    "--root_path", "./dataset/housemill/",
    "--data_path", "data.csv",
    "--model_id", "HouseMill_336_96",
    "--model", "PatchTST",
    "--data", "custom",
    "--features", "MS",
    "--target", "internal_water_m",
    "--freq", "min",

    "--seq_len", "672",
    "--label_len", "96",
    "--pred_len", "96",

    "--enc_in", str(enc_in),
    "--e_layers", "2",
    "--n_heads", "4",
    "--d_model", "32",
    "--d_ff", "128",

    "--dropout", "0.2",
    "--fc_dropout", "0.2",
    "--head_dropout", "0.0",
    "--patch_len", "16",
    "--stride", "8",
    "--revin", "0",

    "--des", "Exp",
    "--train_epochs", "60",
    "--patience", "8",
    "--itr", "1",
    "--batch_size", "128",

    "--learning_rate", "0.0002"
]

# ==========================================
# Execution & Logging
# ==========================================
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/patchtst_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

print("=" * 60)
print(f"[SYSTEM] Launching PatchTST... Log will be saved to {log_filename}")
print("Command:", " ".join(cmd))
print("=" * 60)

# generate log file
with open(log_filename, 'w', encoding='utf-8') as log_file:
    log_file.write("Command executed: " + " ".join(cmd) + "\n\n")

    process = subprocess.Popen(
        cmd, cwd=PATCHTST_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    # print and write to log
    for line in process.stdout:
        print(line, end='')
        log_file.write(line)
        log_file.flush()  # instant write into log

process.wait()
print("=" * 60)
print(f"[SYSTEM] Training Finished! Log completely saved to {log_filename}")