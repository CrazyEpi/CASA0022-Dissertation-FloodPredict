import os
import pandas as pd
import subprocess
from datetime import datetime

# ==========================================
# Paths Configuration
# ==========================================
SOURCE_CSV = r"C:\UCL\Dissertation\data\house_mill_integrated_dataset.csv"
PATCHTST_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET_DATASET_DIR = os.path.join(PATCHTST_DIR, "dataset", "housemill")
os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
TARGET_CSV = os.path.join(TARGET_DATASET_DIR, "data.csv")

# ==========================================
# Data Format Adaptation & Leakage Prevention
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

# 【核心】：一定要屏蔽声纳和外部河道，逼迫模型用降雨和潮汐推演！
GLOBAL_LEAKAGE_DROP = ["sonar", "lea_height", "lea_"]
cols_to_drop = [c for c in df.columns if any(leak in c for leak in GLOBAL_LEAKAGE_DROP)]
if cols_to_drop:
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"[INFO] Dropped leakage features: {cols_to_drop}")

cols = [c for c in df.columns if c != 'internal_water_m'] + ['internal_water_m']
df = df[cols]
df.to_csv(TARGET_CSV, index=False)
enc_in = len(df.columns) - 1
print(f"[INFO] Final input dimension (enc_in): {enc_in}")

# ==========================================
# Command Construction (Hyperparameters)
# ==========================================
cmd = [
    "python", "-u", "run_longExp.py",
    "--is_training", "1",
    "--root_path", "./dataset/housemill/",
    "--data_path", "data.csv",
    "--model_id", "HouseMill_Final_Baseline",
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
    "--des", "Final_Baseline",
    "--train_epochs", "60",
    "--patience", "5",
    "--itr", "1",
    "--batch_size", "128",
    "--learning_rate", "0.0001" # 建议用0.0001，更稳定
]

# ==========================================
# Execution & Logging
# ==========================================
os.makedirs('logs', exist_ok=True)
log_filename = f"logs/patchtst_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

print("=" * 60)
print(f"[SYSTEM] Launching PatchTST Final Baseline... Log will be saved to {log_filename}")
print("=" * 60)

with open(log_filename, 'w', encoding='utf-8') as log_file:
    log_file.write("Command executed: " + " ".join(cmd) + "\n\n")

    process = subprocess.Popen(
        cmd, cwd=PATCHTST_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace'
    )

    for line in process.stdout:
        print(line, end='')
        log_file.write(line)
        log_file.flush()

process.wait()
print("=" * 60)
print(f"[SYSTEM] Training Finished! Log saved to {log_filename}")