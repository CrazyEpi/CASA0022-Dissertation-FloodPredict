import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        # Force flood samples to be penalized 15x more!
        super().__init__()
        self.threshold = threshold
        self.weight = weight
        # Huber loss
        self.huber = nn.HuberLoss(reduction='none', delta=1.0)

    def forward(self, pred, target):
        loss = self.huber(pred, target)

        # Apply heavy penalty to flood zones
        mask = (target > self.threshold).float()
        weighted_loss = loss * (1 + mask * self.weight)

        return weighted_loss.mean()


# ------ Utilities ------

class Trainer:
    def __init__(self, model, device, threshold_scaled):
        self.model = model.to(device)
        self.device = device
        self.threshold = threshold_scaled
        # AdamW is industry standard now
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.criterion = LossFunction(threshold_scaled)
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
    # Load data and simple feature engineering
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True).dropna()

    # Adding time features (sin/cos encoding for cyclicity)
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    features = ['hour_sin', 'hour_cos', 'lea_height_m', 'silver_tidal_m', 'tower_tidal_m', 'internal_water_m']

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features].values)

    # Floof line 4.43m
    dummy = np.zeros((1, len(features)))
    dummy[0, -1] = 4.43
    threshold_scaled = scaler.transform(dummy)[0, -1]

    # Split dataset (80% train, 20% val)
    split = int(len(data_scaled) * 0.8)
    full_dataset = FloodDataset(data_scaled[:split], threshold_scaled=threshold_scaled)

    # Balanced sampler will force model to focus on flooded events
    class_sample_count = np.array([len(np.where(full_dataset.has_flood == t)[0]) for t in [0, 1]])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in full_dataset.has_flood])

    sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight), len(samples_weight))

    train_loader = DataLoader(full_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(FloodDataset(data_scaled[split:], threshold_scaled=threshold_scaled), batch_size=64)

    model = LSTM(input_dim=len(features), hidden_dim=256, num_layers=3, output_dim=96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device, threshold_scaled)

    print(f"[SYSTEM] Data balance mode activated. Device: {device} | Flood sample weights forcefully boosted!")

    # Training loop
    for epoch in range(50):
        t_loss, t_num_acc, t_recall = trainer.train_step(train_loader)
        trainer.scheduler.step()

        print(
            f"Epoch {epoch + 1:02d} | Loss: {t_loss:.4f} | Num Acc: {t_num_acc:.2%} | Flood Catch Rate (Recall): {t_recall:.2%}"
        )


if __name__ == "__main__":
    run_training('C:\\UCL\\Dissertation\\data\\data.csv')