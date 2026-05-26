import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


# ------ Configs ------

class Config:
    CSV_PATH = 'C:\\UCL\\Dissertation\\data\\house_mill_integrated_dataset.csv'
    LOOKBACK = 336
    HORIZON = 96
    HIDDEN_DIM = 256
    NUM_LAYERS = 3
    DROPOUT = 0.3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 64
    EPOCHS = 50
    PATIENCE = 15
    FLOOD_THRESHOLD = 4.43
    LOSS_WEIGHT = 15.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------ Dataloader ------

class FloodDataset(Dataset):

    def __init__(self, data, lookback=336, horizon=96, threshold_scaled=0.0):
        self.lookback = lookback
        self.horizon = horizon
        self.data = torch.FloatTensor(data)
        self.threshold = threshold_scaled

        # Predict flood situation of each sample to help the sampler know which is which.
        self.has_flood = []
        for i in range(len(self.data) - lookback - horizon):
            target_seq = self.data[i + lookback: i + lookback + horizon, -1]
            # If any point in the future exceeds threshold (internal_water_m), be labeled as a flood window
            self.has_flood.append(1 if (target_seq > threshold_scaled).any() else 0)

        self.has_flood = np.array(self.has_flood)

    def __len__(self):
        return len(self.data) - self.lookback - self.horizon

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.lookback]
        y = self.data[idx + self.lookback: idx + self.lookback + self.horizon, -1]
        return x, y


# ------ Neural network ------

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Adding LayerNorm
        self.ln = nn.LayerNorm(hidden_dim)

        # Deep prediction head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        # We only need the last time step output from the LSTM
        last_out = self.ln(out[:, -1, :])
        return self.fc(last_out)


# ------ Asymmetric Weighted Huber Loss ------

class LossFunction(nn.Module):
    def __init__(self, threshold, weight=15.0):
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        base_loss = self.mse(pred, target)

        # if water level higher than threshold, triggers punishment
        mask = (target > self.threshold).float()

        # punishment
        weighted_loss = base_loss * (1 + mask * self.weight)

        return torch.mean(weighted_loss)


def calc_nse(obs, sim):
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else 0.0


# ------ Utilities ------

class Trainer:
    def __init__(self, model, device, threshold_scaled):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold_scaled
        # AdamW is industry standard now
        self.optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        self.criterion = LossFunction(threshold_scaled, weight=Config.LOSS_WEIGHT)
        # Cosine annealing to escape local minima
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10)

    def eval_metrics(self, output, target):
        # Numerical accuracy: error < 0.2 (regression tolerance)
        num_acc = (torch.abs(output - target) < 0.2).float().mean().item()

        # Flood classification metrics: Recall (Predicted flood / Real flood)
        pred_f = (output > self.threshold).float()
        target_f = (target > self.threshold).float()

        true_pos = (pred_f * target_f).sum()
        actual_pos = target_f.sum()

        # Add 1e-6 to avoid zero division error
        recall = (true_pos / (actual_pos + 1e-6)).item()

        return num_acc, recall

    def train_step(self, loader):
        self.model.train()
        total_loss, total_num_acc, total_recall = 0, 0, 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(x)
            loss = self.criterion(pred, y)

            loss.backward()
            # Clip gradients to 1.0 to ensure stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_acc, rec = self.eval_metrics(pred, y)
            total_num_acc += n_acc
            total_recall += rec

        return total_loss / len(loader), total_num_acc / len(loader), total_recall / len(loader)


# ------ Start Training ------

def run_training(csv_path):
    log_messages = []

    def log_print(message):
        print(message)
        log_messages.append(message + "\n")

    # Load data
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).dropna()

    # Remove sonar related columns, prevent data leakage
    cols_to_drop = [c for c in df.columns if 'sonar' in c.lower()]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Make sure target is last column
    cols = [c for c in df.columns if c != 'internal_water_m'] + ['internal_water_m']
    df = df[cols]
    features = list(df.columns)

    log_print(f"Detected {len(features)} features: {features}")

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df.values)

    # Calculate flood threshold after standardization
    dummy = np.zeros((1, len(features)))
    dummy[0, -1] = Config.FLOOD_THRESHOLD
    threshold_scaled = scaler.transform(dummy)[0, -1]

    # Split dataset
    split = int(len(data_scaled) * 0.8)
    full_dataset = FloodDataset(data_scaled[:split], lookback=Config.LOOKBACK, horizon=Config.HORIZON,
                                threshold_scaled=threshold_scaled)

    # Sampler
    class_sample_count = np.array([len(np.where(full_dataset.has_flood == t)[0]) for t in [0, 1]])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in full_dataset.has_flood])

    sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), len(samples_weight))

    train_loader = DataLoader(full_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(FloodDataset(data_scaled[split:], lookback=Config.LOOKBACK, horizon=Config.HORIZON,
                                         threshold_scaled=threshold_scaled), batch_size=Config.BATCH_SIZE)

    model = LSTM(input_dim=len(features), hidden_dim=Config.HIDDEN_DIM, num_layers=Config.NUM_LAYERS,
                 output_dim=Config.HORIZON, dropout=Config.DROPOUT)
    device = Config.DEVICE
    trainer = Trainer(model, device, threshold_scaled)

    log_print(f"Data balance mode activated. Device: {device}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'best_lstm_model.pth'

    # Training loop
    for epoch in range(Config.EPOCHS):
        t_loss, t_num_acc, t_recall = trainer.train_step(train_loader)
        trainer.scheduler.step()

        # [新增] 在验证集上计算 Validation Loss 用于早停判断
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val)
                loss = trainer.criterion(preds, y_val)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        log_print(
            f"Epoch {epoch + 1:02d} | Train Loss: {t_loss:.4f} | Val Loss: {val_loss:.4f} | Num Acc: {t_num_acc:.2%} | Flood Catch Rate (Recall): {t_recall:.2%}"
        )

        # Early stop
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            log_print(
                f"Best Performance, Saving Model..."
            )
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                log_print(f"Early stopping triggered at Epoch {epoch + 1}! Best Val Loss: {best_val_loss:.4f}")
                break

    # Validation Set Evaluation
    log_print(f"Training complete!!!!!!! Performing evaluation on verification dataset...")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        log_print("Loaded best model weights for final evaluation.")

    model.eval()
    val_preds, val_obs = [], []

    with torch.no_grad():
        for x_val, y_val in val_loader:
            x_val = x_val.to(device)
            preds = model(x_val).cpu().numpy()
            val_preds.append(preds)
            val_obs.append(y_val.numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_obs = np.concatenate(val_obs, axis=0)

    preds_continuous = val_preds[:, 0].reshape(-1, 1)
    obs_continuous = val_obs[:, 0].reshape(-1, 1)

    # Inverse transform
    preds_real = \
        scaler.inverse_transform(
            np.hstack([np.zeros((preds_continuous.shape[0], len(features) - 1)), preds_continuous]))[
            :, -1]
    obs_real = \
        scaler.inverse_transform(np.hstack([np.zeros((obs_continuous.shape[0], len(features) - 1)), obs_continuous]))[
            :, -1]

    nse_score = calc_nse(obs_real, preds_real)
    log_print(f"Final NSE: {nse_score:.4f}")

    # Generate Plots
    # Get timeline
    val_time_index = df.index[split + Config.LOOKBACK: split + Config.LOOKBACK + len(obs_real)]

    # Find sole flood peaks
    peaks, properties = find_peaks(obs_real, height=Config.FLOOD_THRESHOLD, distance=96)

    plot_filename = 'lstm_balanced_evaluation.png'

    if len(peaks) > 0:
        # get top 3 peaks
        peak_heights = obs_real[peaks]
        top_peaks = peaks[np.argsort(peak_heights)[-3:]][::-1]

        fig, axes = plt.subplots(len(top_peaks), 1, figsize=(18, 6 * len(top_peaks)), sharex=False)
        if len(top_peaks) == 1:
            axes = [axes]

        for i, peak_idx in enumerate(top_peaks):
            # before / after flood 96 hours to visualization
            start_idx = max(0, peak_idx - 384)
            end_idx = min(len(obs_real), peak_idx + 384)

            event_time = val_time_index[start_idx:end_idx]
            event_obs = obs_real[start_idx:end_idx]
            event_pred = preds_real[start_idx:end_idx]

            axes[i].plot(event_time, event_obs, label='Observation (Actual Water Level)', color='blue', alpha=0.7)
            axes[i].plot(event_time, event_pred, label='LSTM 1-Step-Ahead Prediction', color='red', alpha=0.8,
                         linestyle='--')
            axes[i].axhline(y=Config.FLOOD_THRESHOLD, color='black', linestyle=':', label='Floor Level (4.43m)')

            peak_time_str = val_time_index[peak_idx].strftime('%Y-%m-%d %H:%M')
            axes[i].set_title(f'Targeted Flood Event {i + 1} | Peak Time: {peak_time_str} | NSE: {nse_score:.3f}',
                              fontsize=14)
            axes[i].set_ylabel('Water Level Height (m)')
            axes[i].legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)
        log_print(f"Identified {len(top_peaks)} major flood events in validation set. Plot saved as: {plot_filename}")

    else:
        # if there's no flood
        log_print("No flood peaks above 4.43m found in validation set. Plotting continuous timeline instead.")
        plt.figure(figsize=(20, 6))
        plot_len = min(3000, len(obs_real))
        plt.plot(val_time_index[:plot_len], obs_real[:plot_len], label='Observation', color='blue', alpha=0.7)
        plt.plot(val_time_index[:plot_len], preds_real[:plot_len], label='Prediction', color='red', linestyle='--')
        plt.axhline(y=Config.FLOOD_THRESHOLD, color='black', linestyle=':', label='Floor Level (4.43m)')
        plt.title(f'House Mill Validation Evaluation (No Floods Detected) | NSE: {nse_score:.4f}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300)

    # Save Logs
    os.makedirs('logs', exist_ok=True)
    log_filename = f"logs/lstm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.writelines(log_messages)
    log_print(f"Runtime logs saved as: {log_filename}")


if __name__ == "__main__":
    run_training(Config.CSV_PATH)