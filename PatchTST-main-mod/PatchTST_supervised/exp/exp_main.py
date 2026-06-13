from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, NLinear, PatchTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler

import os
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')


class AsymmetricFloodLoss(nn.Module):
    def __init__(self, delta=1.0, peak_penalty=1.2, under_predict_factor=2.8, over_predict_penalty=3.5, deadzone=0.15):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        # Reduced peak penalty to prevent gradients from exploding
        self.peak_penalty = peak_penalty

        # [MODIFIED]: Lowered under-prediction to stop the model from being TOO aggressive
        self.under_predict_factor = under_predict_factor

        # [MODIFIED]: Increased over-prediction penalty to punish Precision-killers
        self.over_predict_penalty = over_predict_penalty
        self.deadzone = deadzone

    def forward(self, pred, true):
        error = pred - true
        base_loss = self.huber(pred, true)
        severity_weights = torch.exp(self.peak_penalty * F.relu(true))

        # Precision-Focused: Only penalize under-prediction when it's genuinely high-risk
        under_predict_mask = (true > 1.2) & (error < 0)
        under_penalty = under_predict_mask.float() * self.under_predict_factor

        # Precision-Focused: Aggressive penalty for high-confidence False Positives
        # If the model predicts a flood (error > deadzone) while it's actually safe (true < 1.0)
        over_predict_mask = (true < 1.0) & (error > self.deadzone)
        over_penalty = over_predict_mask.float() * self.over_predict_penalty

        # Reduce penalty for "Safe Margin" (the model is just slightly cautious)
        safe_margin_mask = (error > 0) & (error <= self.deadzone)
        base_loss = torch.where(safe_margin_mask, base_loss * 0.1, base_loss)

        direction_multiplier = 1.0 + under_penalty + over_penalty

        weighted_loss = base_loss * severity_weights * direction_multiplier
        return torch.mean(weighted_loss)


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = AsymmetricFloodLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        vali_tp = 0
        vali_ap = 0
        vali_pp = 0
        vali_correct = 0
        total_vali_samples = 0

        bce_criterion = nn.BCEWithLogitsLoss()

        flood_threshold_real = 4.43
        try:
            flood_threshold_scaled = vali_data.scaler.transform(
                np.zeros((1, vali_data.data_x.shape[1] + 1)) + flood_threshold_real
            )[0, -1]
        except:
            flood_threshold_scaled = 1.0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, clf_outputs = self.model(batch_x)
                else:
                    outputs, clf_outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_reg = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                clf_outputs = clf_outputs[:, -self.args.pred_len:, :]

                reg_loss = criterion(outputs, batch_y_reg)
                clf_target = (batch_y_reg > flood_threshold_scaled).float()
                clf_loss = bce_criterion(clf_outputs, clf_target)

                loss = reg_loss + clf_loss
                total_loss.append(loss.item())

                preds_flat = outputs[..., -1]
                target_flat = batch_y_reg[..., -1]
                clf_probs = torch.sigmoid(clf_outputs[..., -1])

                vali_correct += (torch.abs(preds_flat - target_flat) < 0.2).sum().item()
                total_vali_samples += target_flat.numel()

                pred_f = (preds_flat > flood_threshold_scaled) & (clf_probs > 0.35)
                target_f = target_flat > flood_threshold_scaled

                vali_tp += (pred_f & target_f).sum().item()
                vali_ap += target_f.sum().item()
                vali_pp += pred_f.sum().item()

        total_loss = np.average(total_loss)
        vali_recall = vali_tp / (vali_ap + 1e-6)
        vali_precision = vali_tp / (vali_pp + 1e-6)

        # calculate num acc
        vali_num_acc = vali_correct / (total_vali_samples + 1e-6)

        self.model.train()

        return total_loss, vali_recall, vali_precision, vali_num_acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        pos_weight = torch.tensor([4.0]).to(self.device)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Dynamic threshold alignment for BCE target
        flood_threshold_real = 4.43
        try:
            flood_threshold_scaled = train_data.scaler.transform(
                np.zeros((1, train_data.data_x.shape[1] + 1)) + flood_threshold_real
            )[0, -1]
        except:
            flood_threshold_scaled = 1.5

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            train_correct = 0
            train_tp = 0
            train_ap = 0
            train_pp = 0
            total_train_samples = 0

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Multi-task output unpacking
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, clf_outputs = self.model(batch_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_reg = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        clf_outputs = clf_outputs[:, -self.args.pred_len:, :]

                        reg_loss = criterion(outputs, batch_y_reg)
                        clf_target = (batch_y_reg > flood_threshold_scaled).float()
                        clf_loss = bce_criterion(clf_outputs, clf_target)

                        loss = reg_loss + clf_loss
                        train_loss.append(loss.item())
                else:
                    outputs, clf_outputs = self.model(batch_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_reg = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    clf_outputs = clf_outputs[:, -self.args.pred_len:, :]

                    reg_loss = criterion(outputs, batch_y_reg)
                    clf_target = (batch_y_reg > flood_threshold_scaled).float()
                    clf_loss = bce_criterion(clf_outputs, clf_target)

                    loss = reg_loss + clf_loss
                    train_loss.append(loss.item())

                # Dual-Lock mechanism evaluation
                with torch.no_grad():
                    preds_flat = outputs[..., -1]
                    target_flat = batch_y_reg[..., -1]
                    clf_probs = torch.sigmoid(clf_outputs[..., -1])

                    train_correct += (torch.abs(preds_flat - target_flat) < 0.2).sum().item()

                    # Both constraints must be met to trigger the flood warning
                    pred_f = (preds_flat > flood_threshold_scaled) & (clf_probs > 0.35)
                    target_f = target_flat > flood_threshold_scaled

                    train_tp += (pred_f & target_f).sum().item()
                    train_ap += target_f.sum().item()
                    train_pp += pred_f.sum().item()
                    total_train_samples += target_flat.numel()

                if (i + 1) % 100 == 0:
                    rolling_acc = train_correct / total_train_samples
                    rolling_recall = train_tp / (train_ap + 1e-6)
                    rolling_precision = train_tp / (train_pp + 1e-6)

                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print(
                        "\t Num Acc: {0:.2%} | Recall: {1:.2%} | Precision: {2:.2%}".format(rolling_acc, rolling_recall,
                                                                                            rolling_precision))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss, vali_acc, vali_recall, vali_precision = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc, test_recall, test_precision = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}")
            print(
                f"    --> [Validation] Acc: {vali_acc:.2%} | Recall: {vali_recall:.2%} | Precision: {vali_precision:.2%}")
            print(f"    --> [Test] Acc: {test_acc:.2%} | Recall: {test_recall:.2%} | Precision: {test_precision:.2%}")

            # F1 Score
            if (vali_recall + vali_precision) > 0:
                vali_f1 = 2 * (vali_precision * vali_recall) / (vali_precision + vali_recall)
            else:
                vali_f1 = 0.0

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        clf_preds = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, clf_outputs = self.model(batch_x)
                else:
                    outputs, clf_outputs = self.model(batch_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                clf_outputs = clf_outputs[:, -self.args.pred_len:, :]

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                clf_probs = torch.sigmoid(clf_outputs).detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                clf_preds.append(clf_probs)
                inputx.append(batch_x.detach().cpu().numpy())

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd_arr = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd_arr, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        clf_preds = np.array(clf_preds)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        clf_preds = clf_preds.reshape(-1, clf_preds.shape[-2], clf_preds.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        f.write('\n\n')
        f.close()

        np.save(folder_path + 'pred.npy', preds)

        # Plotting & Physical Constraints
        try:
            print("[SYSTEM] Executing Multi-Task Dual-Lock Evaluation...")

            preds_1step = preds[:, 0, -1]
            obs_1step = trues[:, 0, -1]
            clf_probs_1step = clf_preds[:, 0, -1]

            scaler = test_data.scaler
            mean = scaler.mean_[-1]
            scale = scaler.scale_[-1]

            preds_real = (preds_1step * scale) + mean
            obs_real = (obs_1step * scale) + mean

            FLOOD_THRESHOLD = 4.43

            # Apply Dual-Lock Gating Constraint
            # If the classification network indicates a safe threshold (probability < 50%),
            # aggressively flatten any phantom harmonic surges created by the regression head.
            preds_real = np.where(clf_probs_1step < 0.5, np.minimum(preds_real, FLOOD_THRESHOLD - 0.1), preds_real)

            csv_path = os.path.join(self.args.root_path, self.args.data_path)
            df_raw = pd.read_csv(csv_path)
            time_col = 'date' if 'date' in df_raw.columns else df_raw.columns[0]
            full_time_index = pd.to_datetime(df_raw[time_col].values)

            val_time_index = full_time_index[-len(obs_real):]

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(folder_path, f'patchtst_balanced_evaluation_{timestamp}.png')

            peaks, properties = find_peaks(obs_real, height=FLOOD_THRESHOLD, distance=96)

            if len(peaks) > 0:
                peak_heights = obs_real[peaks]
                top_peaks = peaks[np.argsort(peak_heights)[-3:]][::-1]

                fig, axes = plt.subplots(len(top_peaks), 1, figsize=(18, 6 * len(top_peaks)), sharex=False)
                if len(top_peaks) == 1: axes = [axes]

                for i, peak_idx in enumerate(top_peaks):
                    start_window = max(0, peak_idx - 384)
                    end_window = min(len(obs_real), peak_idx + 384)

                    event_time = val_time_index[start_window:end_window]
                    event_obs = obs_real[start_window:end_window]
                    event_pred = preds_real[start_window:end_window]

                    axes[i].plot(event_time, event_obs, label='Observation (Actual Water Level)', color='blue',
                                 alpha=0.7)
                    axes[i].plot(event_time, event_pred, label='PatchTST (Dual-Lock) 1-Step Prediction', color='red',
                                 alpha=0.8, linestyle='--')
                    axes[i].axhline(y=FLOOD_THRESHOLD, color='black', linestyle=':', label='Floor Level (4.43m)')

                    peak_time_str = event_time.values[peak_idx - start_window]
                    if isinstance(peak_time_str, np.datetime64):
                        peak_time_str = pd.Timestamp(peak_time_str).strftime('%Y-%m-%d %H:%M')
                    else:
                        peak_time_str = str(peak_time_str)

                    axes[i].set_title(f'Targeted Flood Event {i + 1} | Peak Time: {peak_time_str}', fontsize=14)
                    axes[i].set_ylabel('Water Level Height (m)')
                    axes[i].legend(loc='upper right')

                plt.tight_layout()
                plt.savefig(plot_filename, dpi=300)
                print(f"[SYSTEM] Identified {len(top_peaks)} major flood events. Plot saved as: {plot_filename}")
            else:
                print("[SYSTEM] No floods > 4.43m found. Plotting continuous timeline.")
                plt.figure(figsize=(20, 6))
                plot_len = min(3000, len(obs_real))
                plt.plot(val_time_index[:plot_len], obs_real[:plot_len], label='Observation', color='blue', alpha=0.7)
                plt.plot(val_time_index[:plot_len], preds_real[:plot_len], label='PatchTST (Dual-Lock) Prediction',
                         color='red', linestyle='--')
                plt.axhline(y=FLOOD_THRESHOLD, color='black', linestyle=':', label='Floor Level (4.43m)')
                plt.title('PatchTST Test Evaluation (No Floods Detected)')
                plt.legend(loc='upper right')
                plt.tight_layout()
                plt.savefig(plot_filename, dpi=300)

        except Exception as e:
            print(f"[WARNING] Failed to generate flood evaluation plot: {e}")

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, clf_outputs = self.model(batch_x)
                else:
                    outputs, clf_outputs = self.model(batch_x)

                preds.append(outputs.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)
        return