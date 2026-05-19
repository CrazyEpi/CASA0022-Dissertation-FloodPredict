import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime


# ------ Configs ------

class Config:
    CSV_PATH = 'C:\\UCL\\Dissertation\\data\\data.csv'
    SAVE_MODEL_PATH = 'kratzert_lstm_housemill.pth'
    SEQ_LENGTH = 336
    FORECAST_LEAD = 96
    HIDDEN_SIZE = 20
    NUM_LAYERS = 1
    DROPOUT_RATE = 0.4
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 256
    EPOCHS = 60
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------ Dataloader ------

class Dataloader(Dataset):
    def __init__(self, x_data, y_data, seq_length, forecast_lead):
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).view(-1, 1)
        self.seq_length = seq_length
        self.forecast_lead = forecast_lead

    def __len__(self):
        return len(self.x) - self.seq_length - self.forecast_lead

    def __getitem__(self, idx):
        x_seq = self.x[idx: idx + self.seq_length]
        y_val = self.y[idx + self.seq_length - 1 + self.forecast_lead]
        return x_seq, y_val


def load_and_preprocess_data(csv_path):
    print("正在加载数据...")
    df = pd.read_csv(csv_path, parse_dates=['_time'])
    df.set_index('_time', inplace=True)
    df = df.dropna()

    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)

    input_features = ['hour_sin', 'hour_cos', 'lea_height_m', 'silver_tidal_m', 'tower_tidal_m']
    target_feature = ['internal_water_m']

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    train_x = scaler_x.fit_transform(train_df[input_features].values)
    train_y = scaler_y.fit_transform(train_df[target_feature].values)
    test_x = scaler_x.transform(test_df[input_features].values)
    test_y = scaler_y.transform(test_df[target_feature].values)

    return train_x, train_y, test_x, test_y, scaler_y


# ==========================================
# 3. 模型定义
# ==========================================
class KratzertLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(KratzertLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        last_hidden_state = h_n[-1, :, :]
        return self.fc(self.dropout(last_hidden_state))


# ==========================================
# 4. 水文学评估指标
# ==========================================
def calc_nse(obs, sim):
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    return 1 - (numerator / denominator) if denominator != 0 else 0.0


# ==========================================
# 5. 主执行逻辑
# ==========================================
def main():
    log_messages = []

    def log_print(message):
        print(message)
        log_messages.append(message + "\n")

    train_x, train_y, test_x, test_y, scaler_y = load_and_preprocess_data(Config.CSV_PATH)

    train_loader = DataLoader(Dataloader(train_x, train_y, Config.SEQ_LENGTH, Config.FORECAST_LEAD),
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(Dataloader(test_x, test_y, Config.SEQ_LENGTH, Config.FORECAST_LEAD),
                             batch_size=Config.BATCH_SIZE, shuffle=False)

    flood_threshold_scaled = scaler_y.transform([[4.43]])[0, 0]

    model = KratzertLSTM(train_x.shape[1], Config.HIDDEN_SIZE, Config.NUM_LAYERS, Config.DROPOUT_RATE).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    log_print(f"Training Start | Threshold(Scaled): {flood_threshold_scaled:.4f}")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss, train_correct, train_tp, train_ap = 0.0, 0, 0, 0
        total_samples = 0

        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            preds = model(x)
            y = y.view_as(preds)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (torch.abs(preds - y) < 0.2).sum().item()
            pred_f, target_f = preds > flood_threshold_scaled, y > flood_threshold_scaled
            train_tp += (pred_f & target_f).sum().item()
            train_ap += target_f.sum().item()
            total_samples += len(y)

        t_loss = train_loss / len(train_loader)
        t_num_acc = train_correct / total_samples
        t_recall = train_tp / (train_ap + 1e-6)

        # Evaluation
        model.eval()
        val_preds, val_obs = [], []
        with torch.no_grad():
            for x, y in test_loader:
                preds = model(x.to(Config.DEVICE))
                val_preds.append(preds.cpu().numpy())
                val_obs.append(y.numpy())

        v_preds = np.concatenate(val_preds)
        v_obs = np.concatenate(val_obs)
        v_nse = calc_nse(v_obs, v_preds)

        log_print(
            f"Epoch {epoch + 1:02d} | MSE Loss: {t_loss:.4f} | Num Acc: {t_num_acc:.2%} | Flood Catch Rate (Recall): {t_recall:.2%} | Val NSE: {v_nse:.4f}")

    # Save Log
    os.makedirs('logs', exist_ok=True)
    log_filename = f"logs/log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.writelines(log_messages)
    print(f"Log Saved as: {log_filename}")


if __name__ == '__main__':
    main()