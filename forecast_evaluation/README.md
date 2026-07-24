# 洪水预测与实测对比工具

这个工具把服务器每 30 分钟保存的完整 24 小时预测，与之后到达的 10 分钟声纳实测值按时间匹配。默认使用与服务器相同的 15 分钟窗口均值 `mean`，并等待一个窗口完整结束后再评分，避免使用未完成时间窗的数据。

## 服务器现在保存什么

- `sonar_history.sqlite3`：声纳原始实测，每个 MQTT 上行一条。
- `prediction_history.sqlite3`：每次预测运行及其 96 个未来 15 分钟预测点。
- ESP32 的 MQTT topic：只保留当前风险摘要，不保存完整预测曲线。

旧版本没有 `prediction_history.sqlite3`。旧 MQTT 消息也不包含完整的 `forecast` 数组，因此无法反推出过去的 96 点预测；完整评估从部署新版 `server.py` 后开始。

## 1. 部署预测归档功能

把更新后的 `cloud_flood_server/server.py` 和 `.env.example` 上传到 DigitalOcean。不要用 `.env.example` 覆盖服务器中已经填好密钥的 `.env`，只需确认 `.env` 有：

```text
PREDICTION_ARCHIVE_ENABLED=1
PREDICTION_DATABASE=data/prediction_history.sqlite3
```

重建并仅启动 2GB 服务器需要的两个服务：

```bash
cd /root/cloud_flood_server
docker compose build
docker compose up -d --force-recreate water_collector predictor_worker
docker compose logs --since=5m predictor_worker
```

预测日志仍会每 30 分钟出现。首次成功预测后检查：

```bash
docker compose exec -T predictor_worker python -c "import sqlite3; c=sqlite3.connect('/app/data/prediction_history.sqlite3'); print('runs, points =', c.execute('SELECT COUNT(*) FROM forecast_runs').fetchone()[0], c.execute('SELECT COUNT(*) FROM forecast_points').fetchone()[0]); c.close()"
```

正常情况下，第一次预测应显示 `runs, points = 1 96`。每半小时增加 1 个 run 和 96 个 points。

## 2. 安全导出正在使用的 SQLite

不要在两个容器运行时直接只复制 `.sqlite3` 主文件，因为最新事务可能仍在 WAL 文件中。先在服务器上用 SQLite backup API 创建一致性快照：

```bash
cd /root/cloud_flood_server
docker compose exec -T predictor_worker python -c "import sqlite3; s=sqlite3.connect('/app/data/prediction_history.sqlite3'); d=sqlite3.connect('/app/data/prediction_history_export.sqlite3'); s.backup(d); d.close(); s.close()"
docker compose exec -T water_collector python -c "import sqlite3; s=sqlite3.connect('/app/data/sonar_history.sqlite3'); d=sqlite3.connect('/app/data/sonar_history_export.sqlite3'); s.backup(d); d.close(); s.close()"
```

然后在本地 PowerShell 下载：

```powershell
New-Item -ItemType Directory -Force ".\forecast_evaluation\downloaded_data"
scp root@你的服务器IP:/root/cloud_flood_server/data/prediction_history_export.sqlite3 ".\forecast_evaluation\downloaded_data\prediction_history.sqlite3"
scp root@你的服务器IP:/root/cloud_flood_server/data/sonar_history_export.sqlite3 ".\forecast_evaluation\downloaded_data\sonar_history.sqlite3"
```

## 3. 运行评估

在项目根目录：

```powershell
cd C:\UCL\CASA0016\CASA0022-Dissertation-FloodPredict
py -m pip install -r .\forecast_evaluation\requirements.txt
py .\forecast_evaluation\evaluate_forecasts.py `
  --prediction-db .\forecast_evaluation\downloaded_data\prediction_history.sqlite3 `
  --sonar-db .\forecast_evaluation\downloaded_data\sonar_history.sqlite3
```

如需检查每个预测目标与最近一条原始声纳值，可额外使用：

```powershell
py .\forecast_evaluation\evaluate_forecasts.py `
  --prediction-db .\forecast_evaluation\downloaded_data\prediction_history.sqlite3 `
  --sonar-db .\forecast_evaluation\downloaded_data\sonar_history.sqlite3 `
  --actual-resample-method nearest `
  --match-tolerance-minutes 8
```

也可以直接评估本机服务器目录中的数据库：

```powershell
py .\forecast_evaluation\evaluate_forecasts.py
```

## 输出文件

- `summary.json`：总体样本数、覆盖率、最大/平均/中位数绝对误差、最大误差点详情、Bias、RMSE、提前时间统计、相关系数、R²、风险命中统计。
- `matched_forecast_points.csv`：每个预测点、对应实测、提前量、误差。
- `pending_forecast_points.csv`：目标时间尚未到达，暂不能评分的预测点。
- `lead_time_metrics.csv`：0-1h、1-3h、3-6h、6-12h、12-24h 分段误差。
- `water_source_metrics.csv`：分别统计 API warm-up、声纳 ready、降级回退等输入模式。
- `run_metrics.csv`：每次完整 24h 预测的峰值、ETA、风险等级和误差。
- `risk_confusion_matrix.csv`：0/1/2/3 四级风险混淆矩阵。
- `predicted_vs_actual.png`、`error_by_lead_time.png`、`risk_confusion_matrix.png`：论文可用的基础检查图。

## 指标解释

- Bias 为正：模型总体预测偏高；为负：总体偏低。
- Mean absolute error / MAE：平均绝对水位误差，单位米，最直观。
- Median absolute error：绝对误差中位数，比平均值更不容易被少数异常点影响。
- Maximum absolute error：所有已匹配预测点中最差的一次绝对水位误差。
- RMSE：更重罚少数大误差，通常高于 MAE。
- `archived_lead_time_statistics` 统计全部归档预测点的最短、最长、平均和中位提前时间；`evaluated_lead_time_statistics` 只统计已经匹配到实测值的预测点。
- R² 和 Pearson r：描述曲线变化是否一致，但不能替代 MAE/RMSE。
- 风险等级只评估目标窗口已经结束且声纳匹配覆盖率不低于 90% 的预测运行，避免拿不完整的未来 24h 实测做结论。
- Warning miss 表示实际达到 Warning/Severe、预测却低于 Warning，是最需要优先检查的错误。

建议至少积累 7 天预测后看初步趋势，积累 30 天后再报告稳定的分提前量指标；洪水事件很少时，风险准确率会被大量无风险样本抬高，应同时报告 Warning miss、false alarm 和混淆矩阵。
