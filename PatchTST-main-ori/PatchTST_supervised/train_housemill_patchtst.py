import os
import pandas as pd
import subprocess

# ==========================================
# 1. 路径配置 (Paths Configuration)
# ==========================================
# 指向你刚刚用 ETL pipeline 生成的最终整合数据
SOURCE_CSV = r"C:\UCL\Dissertation\data\house_mill_integrated_dataset.csv"

# PatchTST 监督学习根目录 (当前脚本所在目录)
PATCHTST_DIR = r"C:\UCL\Dissertation\PatchTST-main\PatchTST_supervised"

# 为 PatchTST 内部 Dataloader 创建专属的 dataset 文件夹
TARGET_DATASET_DIR = os.path.join(PATCHTST_DIR, "dataset", "housemill")
os.makedirs(TARGET_DATASET_DIR, exist_ok=True)
TARGET_CSV = os.path.join(TARGET_DATASET_DIR, "data.csv")

# ==========================================
# 2. 数据格式适配 (Format Adaptation)
# ==========================================
print("[INFO] Adapting dataset for PatchTST standard loader...")
try:
    df = pd.read_csv(SOURCE_CSV)
except FileNotFoundError:
    print(f"[ERROR] Could not find {SOURCE_CSV}. Please ensure the ETL pipeline was run.")
    exit(1)

# 【核心适配】：PatchTST (Autoformer框架) 严格要求时间列的表头必须叫 'date'
if '_time' in df.columns:
    df.rename(columns={'_time': 'date'}, inplace=True)
elif 'time' in df.columns:
    df.rename(columns={'time': 'date'}, inplace=True)

# 确保 target 列 (internal_water_m) 在最后，虽然配置了 target 参数，但放最后更稳妥
cols = [c for c in df.columns if c != 'internal_water_m'] + ['internal_water_m']
df = df[cols]

# 保存给 PatchTST 读取
df.to_csv(TARGET_CSV, index=False)

# 动态计算 enc_in (输入特征数，需扣除 'date' 列)
enc_in = len(df.columns) - 1
print(f"[INFO] Dataset adapted successfully. Detected {enc_in} input features.")

# ==========================================
# 3. 构建命令行参数 (Command Construction)
# ==========================================
# 以下参数严格参考 PatchTST/42 及多变量预测的最佳实践
cmd = [
    "python", "run_longExp.py",
    "--is_training", "1",
    "--root_path", "./dataset/housemill/",  # 数据根目录
    "--data_path", "data.csv",  # 数据文件名
    "--model_id", "HouseMill_336_96",  # 实验 ID
    "--model", "PatchTST",  # 调用的模型
    "--data", "custom",  # 声明使用自定义数据集
    "--features", "MS",  # MS = Multivariate input, Single target output
    "--target", "internal_water_m",  # 目标预测列

    # 【修复核心】: 将 't' 改为 'min' 以适配高版本 Pandas
    "--freq", "min",

    # 视界配置
    "--seq_len", "336",  # 历史回溯窗口 (L)
    "--label_len", "96",  # Decoder 启动 Token 长度
    "--pred_len", "96",  # 预测未来步数 (T)

    # PatchTST 核心结构配置
    "--enc_in", str(enc_in),  # 输入通道数
    "--e_layers", "2",  # Encoder 层数
    "--n_heads", "4",  # 多头注意力头数
    "--d_model", "32",  # 潜空间维度
    "--d_ff", "128",  # FFN 维度
    "--dropout", "0.3",
    "--fc_dropout", "0.3",
    "--head_dropout", "0.0",
    "--patch_len", "16",  # P: 分块长度
    "--stride", "8",  # S: 分块滑动步长

    # 训练超参数
    "--des", "Exp",
    "--train_epochs", "50",
    "--patience", "10",  # Early stopping 耐心值
    "--itr", "1",  # 实验重复次数
    "--batch_size", "128",
    "--learning_rate", "0.0003"  # 官方推荐对于长序列使用较小学习率
]

# ==========================================
# 4. 执行训练 (Execution)
# ==========================================
print("=" * 60)
print("[SYSTEM] Launching PatchTST Supervised Training...")
print("=" * 60)
print("Command:", " ".join(cmd))
print("=" * 60)

# 通过 subprocess 在当前目录拉起训练
subprocess.run(cmd, cwd=PATCHTST_DIR)