import os
import torch
import numpy as np
import pandas as pd
import random
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from models.PatchTST import Model

class IoTConfig:
    enc_in = 31  # data dimension
    seq_len = 672  # go back 7 days
    pred_len = 96  # future 24h
    e_layers = 2
    n_heads = 4
    d_model = 32
    d_ff = 128
    dropout = 0.2
    fc_dropout = 0.2
    head_dropout = 0.0
    patch_len = 16
    stride = 8
    revin = 0
    decomposition = 0
    individual = 0
    padding_patch = 'end'
    affine = 0
    subtract_last = 0
    kernel_size = 25


def run_1year_simulation():
    # env
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configs = IoTConfig()

    FLOOD_THRESHOLD = 4.43
    CLF_THRESHOLD = 0.6  # classification threshold
    MODEL_PATH = "C:\\UCL\CASA0016\CASA0022-Dissertation-FloodPredict\PatchTST-main-mod\PatchTST_supervised\checkpoints\HouseMill_Final_Baseline_PatchTST_custom_ftMS_sl672_ll96_pl96_dm32_nh4_el2_dl1_df128_fc1_ebtimeF_dtTrue_Final_Baseline_0/checkpoint.pth"

    # Data Path
    if os.path.exists("C:\\UCL\Dissertation\data\house_mill_integrated_dataset.csv"):
        DATA_PATH = "C:\\UCL\Dissertation\data\house_mill_integrated_dataset.csv"
    else: print(f"[ERROR] Cannot find data file at {DATA_PATH}. Please check the path.")

    print(f"[SYSTEM] Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Cannot find data file at {DATA_PATH}. Please check the path.")
        return

    if '_time' in df.columns:
        df.rename(columns={'_time': 'date'}, inplace=True)
    elif 'time' in df.columns:
        df.rename(columns={'time': 'date'}, inplace=True)

    # remove sonar related features
    GLOBAL_LEAKAGE_DROP = ["sonar", "lea_height", "lea_"]
    cols_to_drop = [c for c in df.columns if any(leak in c for leak in GLOBAL_LEAKAGE_DROP)]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        print(f"[INFO] Dropped leakage features: {cols_to_drop}")

    cols = [c for c in df.columns if c not in ['date', 'internal_water_m']] + ['internal_water_m']
    df = df[['date'] + cols]

    cols_data = [col for col in df.columns if col != 'date']
    df_data = df[cols_data]

    if len(cols_data) != configs.enc_in:
        print(f"[WARNING] Dimension Mismatch! Config expects {configs.enc_in}, but got {len(cols_data)}")

    # standardize
    scaler = StandardScaler()
    fit_len = min(62368, len(df_data))
    scaler.fit(df_data.iloc[:fit_len].values)
    data_scaled = scaler.transform(df_data.values)

    # 1 year data: 365d * 24h * 4per hr = 35040 steps
    YEAR_STEPS = 35040
    max_start = len(data_scaled) - YEAR_STEPS - configs.seq_len - configs.pred_len

    if max_start <= 0:
        print("[WARNING] Dataset is shorter than 1 year, using maximum available length.")
        start_idx = 0
        YEAR_STEPS = len(data_scaled) - configs.seq_len - configs.pred_len
    else:
        start_idx = random.randint(0, max_start)

    print(f"[SYSTEM] Random 1-Year Window Dropped In: Step {start_idx} to {start_idx + YEAR_STEPS}")

    # load model
    print("[SYSTEM] Loading model weights...")
    model = Model(configs).float().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("[SYSTEM] Model loaded successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to load model from {MODEL_PATH}: {e}")
        print("[SYSTEM] Proceeding with untrained initialized weights for logic testing...")

    model.eval()

    # simulate inference in one year
    print(f"[SYSTEM] Running Continuous Cloud Inference for {YEAR_STEPS} steps... This may take a moment.")

    all_preds_raw = []
    all_clf_probs = []
    all_trues = []

    batch_size = 256
    x_batches = []
    y_batches = []

    # fetch one year samples
    for i in range(YEAR_STEPS):
        curr_start = start_idx + i
        curr_end = curr_start + configs.seq_len
        pred_end = curr_end + configs.pred_len

        x_batches.append(data_scaled[curr_start:curr_end])
        y_batches.append(df_data.iloc[curr_end:pred_end]['internal_water_m'].values)

        # inference
    with torch.no_grad():
        for i in range(0, YEAR_STEPS, batch_size):
            batch_x = torch.tensor(np.array(x_batches[i:i + batch_size])).float().to(device)
            outputs, clf_outputs = model(batch_x)

            # get regression result and back to meter
            pred_water_scaled = outputs[:, :, -1].cpu().numpy()
            pred_water_real = (pred_water_scaled * scaler.scale_[-1]) + scaler.mean_[-1]

            # ger classification result
            clf_probs = torch.sigmoid(clf_outputs[:, :, -1]).cpu().numpy()

            all_preds_raw.extend(pred_water_real)
            all_clf_probs.extend(clf_probs)
            all_trues.extend(y_batches[i:i + batch_size])

            if (i % 5000) < batch_size and i > 0:
                print(f"   ... Processed {i}/{YEAR_STEPS} steps ({(i / YEAR_STEPS):.1%})")

    all_preds_raw = np.array(all_preds_raw)
    all_clf_probs = np.array(all_clf_probs)
    all_trues = np.array(all_trues)

    all_preds_locked = np.where(
        all_clf_probs < CLF_THRESHOLD,
        np.minimum(all_preds_raw, FLOOD_THRESHOLD - 0.1),
        all_preds_raw
    )

    continuous_pred_locked = all_preds_locked[:, 0]
    continuous_true = all_trues[:, 0]

    # Evaluation

    print("\n" + "=" * 65)
    print("[SYSTEM] 1-YEAR SIMULATION: EVENT-BASED FLOOD EVALUATION")
    print("=" * 65)

    # make sure one flood are count as one
    true_peaks, _ = find_peaks(continuous_true, height=FLOOD_THRESHOLD, distance=96)
    pred_peaks, _ = find_peaks(continuous_pred_locked, height=FLOOD_THRESHOLD, distance=96)

    detected_events = 0
    missed_events = 0
    false_alarms = 0
    lead_times = []
    peak_errors = []

    # metrics
    for t_peak in true_peaks:
        earliest_alert_N = None
        search_start = max(0, t_peak - 96)

        # emergency time count
        for N in range(search_start, t_peak + 1):
            if N >= len(all_preds_locked): break
            forecast_window = all_preds_locked[N]

            if np.max(forecast_window) >= FLOOD_THRESHOLD:
                earliest_alert_N = N
                break

        if earliest_alert_N is not None:
            detected_events += 1
            lead_time_steps = t_peak - earliest_alert_N
            lead_times.append(lead_time_steps / 4.0)

            pred_max = np.max(all_preds_locked[earliest_alert_N])
            true_max = continuous_true[t_peak]
            peak_errors.append(pred_max - true_max)
        else:
            missed_events += 1

    # fake alarm
    for p_peak in pred_peaks:
        search_start = max(0, p_peak - 96)
        search_end = min(len(continuous_true), p_peak + 96)
        if np.max(continuous_true[search_start:search_end]) < FLOOD_THRESHOLD:
            false_alarms += 1

    total_actual = len(true_peaks)
    event_recall = detected_events / total_actual if total_actual > 0 else 1.0
    event_precision = detected_events / (detected_events + false_alarms) if (
                                                                                        detected_events + false_alarms) > 0 else 1.0
    avg_lead_time = np.mean(lead_times) if len(lead_times) > 0 else 0.0
    avg_peak_error = np.mean(peak_errors) if len(peak_errors) > 0 else 0.0

    print(f"FULL YEAR STRESS TEST RESULTS:")
    print(f"  - Total Actual Flood Events: {total_actual} storms over 365 days")
    print(f"  - Successfully Detected:  {detected_events} storms")
    print(f"  - Missed Events (FN):     {missed_events} storms")
    print(f"  - False Alarms (FP):      {false_alarms} phantom alerts")
    print("-" * 40)
    print(f"  - Robust Event Recall:     {event_recall:.2%}")
    print(f"  - Robust Event Precision:  {event_precision:.2%}")
    print(f"  - Avg Warning Lead Time:  {avg_lead_time:.1f} Hours before peak")
    print(f"  - Avg Peak Error:         {avg_peak_error:+.2f} meters")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_1year_simulation()