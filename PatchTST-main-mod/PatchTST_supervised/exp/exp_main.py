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
    def __init__(self, delta=1.0, peak_penalty=1.2, under_predict_factor=3.5, over_predict_penalty=3.0, deadzone=0.15):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='none', delta=delta)
        self.peak_penalty = peak_penalty
        self.under_predict_factor = under_predict_factor
        self.over_predict_penalty = over_predict_penalty
        self.deadzone = deadzone

    def forward(self, pred, true):
        error = pred - true
        base_loss = self.huber(pred, true)
        critical_level = torch.clamp(torch.maximum(pred, true), max=3.0)
        severity_weights = torch.exp(self.peak_penalty * F.relu(critical_level))

        under_predict_mask = (true > 1.2) & (error < 0)
        under_penalty = under_predict_mask.float() * self.under_predict_factor

        over_predict_mask = (true < 1.0) & (error > self.deadzone)
        over_penalty = over_predict_mask.float() * self.over_predict_penalty

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
        return AsymmetricFloodLoss()

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

                # 完美平衡阈值
                pred_f = (preds_flat > flood_threshold_scaled) & (clf_probs > 0.45)
                target_f = target_flat > flood_threshold_scaled

                vali_tp += (pred_f & target_f).sum().item()
                vali_ap += target_f.sum().item()
                vali_pp += pred_f.sum().item()

        total_loss = np.average(total_loss)
        vali_recall = vali_tp / (vali_ap + 1e-6)
        vali_precision = vali_tp / (vali_pp + 1e-6)
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

        # 保留原有的 EarlyStopping 用于打印，但拦截它的保存逻辑
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 分类器权重调优为 6.0，既不过度敏感也不麻木
        pos_weight = torch.tensor([6.0]).to(self.device)
        bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

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

        # 【F1 早停机制初始化】
        self.best_vali_f1 = -1.0
        self.f1_patience_counter = 0

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

                with torch.no_grad():
                    preds_flat = outputs[..., -1]
                    target_flat = batch_y_reg[..., -1]
                    clf_probs = torch.sigmoid(clf_outputs[..., -1])

                    train_correct += (torch.abs(preds_flat - target_flat) < 0.2).sum().item()

                    pred_f = (preds_flat > flood_threshold_scaled) & (clf_probs > 0.45)
                    target_f = target_flat > flood_threshold_scaled

                    train_tp += (pred_f & target_f).sum().item()
                    train_ap += target_f.sum().item()
                    train_pp += pred_f.sum().item()
                    total_train_samples += target_flat.numel()

                if (i + 1) % 100 == 0:
                    rolling_acc = train_correct / total_train_samples
                    rolling_recall = train_tp / (train_ap + 1e-6)
                    rolling_precision = train_tp / (train_pp + 1e-6)
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
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

            train_loss = np.average(train_loss)

            vali_loss, vali_recall, vali_precision, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_recall, test_precision, test_acc = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1} | Train Loss: {train_loss:.7f} | Vali Loss: {vali_loss:.7f} | Test Loss: {test_loss:.7f}")
            print(
                f"    --> [Validation] Acc: {vali_acc:.2%} | Recall: {vali_recall:.2%} | Precision: {vali_precision:.2%}")
            print(f"    --> [Test] Acc: {test_acc:.2%} | Recall: {test_recall:.2%} | Precision: {test_precision:.2%}")

            # ==========================================================
            # 【终极防汛 F1 早停机制】
            # ==========================================================
            if (vali_recall + vali_precision) > 0:
                vali_f1 = 2 * (vali_precision * vali_recall) / (vali_precision + vali_recall)
            else:
                vali_f1 = 0.0

            # 如果 F1 分数破纪录，立刻强制保存模型覆盖旧权重
            if vali_f1 > self.best_vali_f1:
                self.best_vali_f1 = vali_f1
                self.f1_patience_counter = 0
                best_model_path = path + '/' + 'checkpoint.pth'
                torch.save(self.model.state_dict(), best_model_path)
                print(f"    [🌟 BEST MODEL SAVED] New Best Vali F1: {vali_f1:.2%}!")
            else:
                self.f1_patience_counter += 1
                print(f"    [Patience] F1 unchanged. Counter: {self.f1_patience_counter} / {self.args.patience}")

            if self.f1_patience_counter >= self.args.patience:
                print(f"Early stopping triggered by F1-Score bottleneck! Best F1 locked at {self.best_vali_f1:.2%}")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        # 训练全部结束后，加载那个具有最高 F1 的极品权重
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

        # ==========================================================
        # 【物理蒙眼与双重锁门画图逻辑】：10天大视野下，N时刻对未来 24h 预测能力
        # ==========================================================
        try:
            print("[SYSTEM] Generating 10-Day Context & 24-Hour Forecasting Evaluation Plots...")

            scaler = test_data.scaler
            mean = scaler.mean_[-1]
            scale = scaler.scale_[-1]

            FLOOD_THRESHOLD = 4.43
            try:
                # 尝试获取你定义在文件顶部的全局阈值
                CLF_THRESHOLD = GLOBAL_CLF_THRESHOLD
            except NameError:
                CLF_THRESHOLD = 0.5

            # 提取最后一列水位 (还原量纲)
            target_trues = (trues[:, :, -1] * scale) + mean
            target_preds_raw = (preds[:, :, -1] * scale) + mean
            target_clf_probs = clf_preds[:, :, -1]

            # 【核心重构】：利用滑动窗口的第 1 步，拼合出一条连续的全局时间轴
            # 突破模型一次只能预测 24 小时的限制，用来绘制 10 天的全局背景能力线
            continuous_true = np.concatenate([target_trues[:, 0], target_trues[-1, 1:]])
            continuous_pred_raw = np.concatenate([target_preds_raw[:, 0], target_preds_raw[-1, 1:]])

            # 双重锁门机制：在未来 96 步的每一步上，如果分类概率低于阈值，强制压制到安全水位
            target_preds_locked = np.where(
                target_clf_probs < CLF_THRESHOLD,
                np.minimum(target_preds_raw, FLOOD_THRESHOLD - 0.1),
                target_preds_raw
            )
            # 生成锁门后的 10 天连续滚动线
            continuous_pred_locked = np.concatenate([target_preds_locked[:, 0], target_preds_locked[-1, 1:]])

            # 寻找测试集中最危险的事件样本（按波峰高度排序）
            max_water_levels = target_trues.max(axis=1)
            sorted_indices = np.argsort(max_water_levels)[::-1]

            top_indices = []
            # 独立样本间隔至少 7 天 (7天 * 24h * 4个15min = 672步)
            min_distance = 672

            for idx in sorted_indices:
                if len(top_indices) >= 3:
                    break
                # 检查距离冲突
                if not any(abs(idx - selected_idx) < min_distance for selected_idx in top_indices):
                    # 确保这个样本逼近或超过洪水线，有展现预警的价值
                    if max_water_levels[idx] > FLOOD_THRESHOLD - 0.5:
                        top_indices.append(idx)

            # 如果没发生严重洪水导致凑不够 3 个，就放宽洪水线要求，只要满足 7 天间隔即可
            if len(top_indices) < 3:
                top_indices = []
                for idx in sorted_indices:
                    if len(top_indices) >= 3:
                        break
                    if not any(abs(idx - selected_idx) < min_distance for selected_idx in top_indices):
                        top_indices.append(idx)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_filename = os.path.join(folder_path, f'forecast_10days_evaluation_{timestamp}.png')

            # 画幅横向拉长 (18, 4.5) 方便 10 天横轴“挤一挤”
            fig, axes = plt.subplots(len(top_indices), 1, figsize=(18, 4.5 * len(top_indices)), sharex=False)
            if len(top_indices) == 1: axes = [axes]

            for i, idx in enumerate(top_indices):
                pred_len = target_trues.shape[1]  # 96 步 (24小时)

                # 设定 10 天的展示窗口: N时刻的前 7 天 (672步) 到 后 3 天 (288步) -> 共 960 步
                plot_start = max(0, idx - 672)
                plot_end = min(len(continuous_true), idx + 288)

                # 构造相对时间轴 (设当前评估时刻 N 为 0)
                time_axis = np.arange(plot_start - idx, plot_end - idx)
                pred_axis = np.arange(0, pred_len)

                # 1. 铺底：画出全局 10 天的无缝真实水位线 (粗蓝线)
                axes[i].plot(time_axis, continuous_true[plot_start:plot_end],
                             label='Global True Water Level', color='#1f77b4', alpha=0.8,
                             linewidth=2)

                # 2. 预测背景：画出全局 10 天的连续滚动预测 (反映模型的综合追踪实力，细线)
                axes[i].plot(time_axis, continuous_pred_raw[plot_start:plot_end],
                             label='Global Raw Regression (Rolling)', color='#ff7f0e', alpha=0.4, linestyle='-',
                             linewidth=1.2)
                axes[i].plot(time_axis, continuous_pred_locked[plot_start:plot_end],
                             label='Global Dual-Lock (Rolling)', color='#d62728', alpha=0.4, linestyle='-',
                             linewidth=1.2)

                # 3. 核心大招：画出在 N 时刻触发的、针对未来 24 小时的长程完整预测 (粗虚线)
                axes[i].plot(pred_axis, target_preds_raw[idx],
                             label='24h Horizon Raw Forecast (96-step)', color='#ff7f0e', alpha=1.0, linestyle='-.',
                             linewidth=2.5)
                axes[i].plot(pred_axis, target_preds_locked[idx],
                             label='24h Horizon Dual-Lock (Final Alert)', color='#d62728', alpha=1.0, linestyle='--',
                             linewidth=3)

                # 视觉聚光灯：用淡黄色背景块高亮标出模型正在推演的这 24 小时
                axes[i].axvspan(0, pred_len - 1, color='yellow', alpha=0.15, label='24h Inference Window')

                # 辅助线：当前时刻 N
                axes[i].axvline(x=0, color='black', linestyle='-', alpha=0.8, label='Current Moment (N)')
                # 辅助线：洪水警戒线
                axes[i].axhline(y=FLOOD_THRESHOLD, color='red', linestyle=':', alpha=0.8,
                                label=f'Flood Threshold ({FLOOD_THRESHOLD}m)')

                axes[i].set_title(
                    f'10-Day Context & 24-Hour Forecast Horizon | Event Peak: {max_water_levels[idx]:.2f}m',
                    fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Time Steps (15-min intervals, 0 = Current Moment N)', fontsize=12)
                axes[i].set_ylabel('Water Level (m)', fontsize=12)

                # 强制锁定横轴边界，确保 10 天视野铺满
                axes[i].set_xlim([time_axis[0], time_axis[-1]])
                axes[i].grid(True, alpha=0.3)

                # 调整图例位置并分成 3 列，避免遮挡重要曲线
                axes[i].legend(loc='upper right', fontsize=10, ncol=3)

            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            print(f"[SYSTEM] 10-Day Horizon Forecast evaluation plot saved to: {plot_filename}")

        except Exception as e:
            print(f"[WARNING] Failed to generate 10-day forecast evaluation plot: {e}")

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