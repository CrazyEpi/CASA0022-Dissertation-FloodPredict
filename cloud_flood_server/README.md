# House Mill Cloud Flood Server

这是 House Mill 洪水预测的云端运行包。它尽量使用 API/实时 MQTT 数据：

- 水位：`WATER_MQTT_*` 订阅实时声纳水位，默认写入支持并发读写和去重的 `data/sonar_history.sqlite3`。
- 降雨：Environment Agency Hydrology API 的 15 分钟 rainfall measures。
- 潮汐：Admiralty UK Tidal API 的 Sheerness station / tidal events。
- 预测：15 分钟 PatchTST checkpoint，输入过去 7 天，输出未来 24h。
- 发布：`ALERT_MQTT_*` 发布给 ESP32 可视化端。

密码、API key 和 MQTT secret 都留在 `.env`，不要写入代码。

## 风险等级

- `0 No risk`: 未来 24h 最高水位低于 `4.20m`，ESP32 显示 `#`。
- `1 Watch`: 最高水位 `4.20m <= level < 4.43m`，表示接近洪水线但还没到。
- `2 Warning`: 最高水位达到 `4.43m`，或连续超过 `4.43m` 达到 1 小时。
- `3 Severe`: 最高水位达到 `4.70m`，或第一次达到 `4.43m` 的 ETA 小于等于 2 小时，或连续超过 `4.43m` 达到 2 小时。

`eta_minutes` 和 `next_flood_utc` 只按真正洪水线 `4.43m` 计算。Watch 只给 `watch_eta_minutes`。

## 本地命令

```powershell
cd C:\UCL\Dissertation\cloud_flood_server
copy .env.example .env
```

检查 API/MQTT 配置是否足够：

```powershell
.\run_local.bat --mode check-apis
```

只跑一次完整预测：

```powershell
.\run_local.bat --mode once
```

如果你只想用训练 CSV 离线检查代码链路，可以临时设置：

```powershell
$env:USE_LIVE_API="1"
$env:ALLOW_HISTORY_CSV_FALLBACK="1"
$env:REQUIRE_RAIN_API="0"
$env:REQUIRE_TIDE_API="0"
$env:LIVE_END_UTC="2025-12-31T23:45:00Z"
.\run_local.bat --mode once
```

启动 HTTP server，手动访问 `/predict`：

```powershell
.\run_local.bat --mode server
```

本机不会自动常驻，除非你显式运行 `server`、`worker` 或 `collect-water-mqtt`。

## 云端进程

DigitalOcean 上建议分两个进程：

```bash
python server.py --mode collect-water-mqtt
python server.py --mode worker
```

`collect-water-mqtt` 持续积累声纳水位。`worker` 每半小时重新预测；如果存在风险，它每分钟更新 ETA 并发布到 `ALERT_MQTT_TOPIC`。

模型采用延迟加载。水位采集进程不会加载 PyTorch；只有 `worker`、`once` 和 HTTP 预测服务加载模型。2GB DigitalOcean Droplet 建议只常驻 `water_collector` 和 `predictor_worker`，不要同时常驻可选 `api` 服务，并配置 2GB swap 防止预测峰值触发 OOM。

也可以用 HTTP 服务：

```bash
python server.py --mode server
curl http://127.0.0.1:8080/predict
```

## 需要填写的配置

`WATER_MQTT_*`: 水位 MQTT 数据源。

`ALERT_MQTT_*`: ESP32 订阅的预测发布 MQTT。

`ADMIRALTY_API_KEY`: Admiralty Developer Portal 订阅 UK Tidal API 后得到。默认 endpoint 使用：

```text
https://admiraltyapi.azure-api.net/uktidalapi/api/V1/Stations/{station_id}/TidalEvents
```

我已经验证 `Stations`、`Stations/{id}` 和 `Stations/{id}/TidalEvents` 路径存在；无 key 时返回 `401 Access Denied`。是否有 `Sheerness` 必须在你填入 key 后用 `--mode check-apis` 确认。

`EA_RAIN_MEASURE_IDS`: Environment Agency Hydrology API 的 15min rainfall measure id，逗号分隔。可以先用：

```text
https://environment.data.gov.uk/hydrology/id/stations.json?search=Deptford&observedProperty=rainfall&_limit=10
https://environment.data.gov.uk/hydrology/id/measures.json?station={stationGuid}&observedProperty=rainfall&periodName=15min
```

找到 measure 后填入 `.env`。

`RIVERLEVELS_LOCATION`: 可选。RiverLevels.uk 下载接口是：

```text
https://riverlevels.uk/[location]/data/[csv|json]/[days]
```

但它只提供每天 `min/avg/max`，不保存小时级历史，所以不能作为 15 分钟预测主链路。

## Docker

```bash
docker build -t housemill-flood .
docker run --env-file .env -p 8080:8080 housemill-flood
```

Docker 默认启动 HTTP server。自动预测建议另开 worker 容器：

```bash
docker run --env-file .env housemill-flood python server.py --mode worker
docker run --env-file .env housemill-flood python server.py --mode collect-water-mqtt
```

## 输出 payload

发布到 ESP32 的 MQTT payload 会去掉完整 forecast 数组，保留核心字段：

```json
{
  "site": "house_mill",
  "risk_level": 2,
  "risk_label": "Warning",
  "display": "6h30m",
  "eta_minutes": 390,
  "next_flood_utc": "2026-07-08T18:30:00Z",
  "max_predicted_m": 4.51,
  "forecast_generated_utc": "2026-07-08T12:00:00Z",
  "valid_until_utc": "2026-07-08T12:30:00Z"
}
```

## RiverLevels API 替代方案

RiverLevels.uk 只适合作为每日背景数据，不适合作为 15 分钟在线预测主链路。服务器现在增加了两个官方替代源：

1. Environment Agency Hydrology API

   用途：历史和近期 15 分钟水位。推荐作为 RiverLevels 的主替代。

   House Mill 附近推荐：

   ```text
   Lee Bridge:
   f463f458-8115-42d8-834a-c6872116737d-level-i-900-m-qualified
   ```

   填入：

   ```text
   EA_WATER_LEVEL_MEASURE_IDS=f463f458-8115-42d8-834a-c6872116737d-level-i-900-m-qualified
   ```

2. Environment Agency Flood Monitoring API

   用途：近实时水位、最新值、官方 Check for Flooding/telemetry 数据校验。推荐用于可靠性检查。

   House Mill 附近推荐：

   ```text
   Lea Bridge:
   5390TH-level-stage-i-15_min-mASD

   Tower Pier tidal level:
   0007-level-tidal_level-i-15_min-mAOD

   Silvertown tidal level:
   0001-level-tidal_level-i-15_min-mAOD
   ```

   填入：

   ```text
   FLOOD_MONITORING_LEVEL_MEASURE_IDS=5390TH-level-stage-i-15_min-mASD,0007-level-tidal_level-i-15_min-mAOD,0001-level-tidal_level-i-15_min-mAOD
   ```

查找附近候选：

```powershell
.\run_local.bat --mode check-river-alternatives
```

检查所有 API 和 MQTT 配置：

```powershell
.\run_local.bat --mode check-apis
```

这些外部水位目前只进入 `data_quality.external_levels` 诊断，不进入模型特征列。原因是当前 checkpoint 是 15 个输入特征训练的，直接新增水位特征会导致模型权重维度不匹配。要把这些水位真正用于预测，需要下一步重新训练一个包含官方外部水位特征的新模型。

## 下一步 TODO

1. 在 `.env` 填入 `WATER_MQTT_*`，启动 `python server.py --mode collect-water-mqtt`，连续积累声纳数据。默认存储是 `sonar_history.sqlite3`；已有 `sonar_history.csv` 会自动兼容导入。

2. 在 `.env` 填入 `EA_RAIN_MEASURE_IDS`。先用 EA Hydrology API 搜索 Deptford、Hornsey、Wanstead、Chalk Bridge Sluice 的 15 分钟 rainfall measures。

3. 在 Admiralty Developer Portal 申请 `ADMIRALTY_API_KEY`，然后运行 `python server.py --mode check-apis`，确认 Sheerness station id 是否就是 `Sheerness`。如果不是，把正确 id 写入 `ADMIRALTY_STATION_ID`。

4. 填入 `EA_WATER_LEVEL_MEASURE_IDS` 和 `FLOOD_MONITORING_LEVEL_MEASURE_IDS`，用 `check-apis` 确认官方外部水位诊断可用。

5. 本地或云端运行 `python server.py --mode once`，确认真实 API 模式下输出 `risk_level`、`display`、`eta_minutes`、`data_quality`。

6. 填入 `ALERT_MQTT_*`，设置 `PUBLISH_ENABLED=1`，让 ESP32 订阅 `ALERT_MQTT_TOPIC`。

7. 在 DigitalOcean 用 systemd 或 Docker Compose 常驻两个进程：`collect-water-mqtt` 和 `worker`。

8. 等第一版跑稳后，重新训练一个新模型，把 Lee Bridge、Tower Pier、Silvertown 等官方外部水位加入训练数据，再更新 `FEATURE_COLUMNS` 和 checkpoint。

## 水位源自动切换策略

默认配置：

```text
PRIMARY_WATER_LEVEL_SOURCE=auto
SONAR_STORAGE=sqlite
SONAR_READY_MIN_COVERAGE=0.90
SONAR_READY_MIN_SPAN_HOURS=167.0
SONAR_READY_MAX_AGE_MINUTES=30
SONAR_ALLOW_DEGRADED_FALLBACK=1
RAIN_SOURCE=auto
TIDE_SOURCE=auto
ALLOW_SHORT_HISTORY_PAD=1
REQUIRE_RAIN_API=0
REQUIRE_TIDE_API=0
```

`auto` 模式不会因为声纳文件刚刚出现就立即切换。只有同时满足以下条件，声纳才成为模型主水位输入：

- 过去 7 天重采样后至少覆盖 90%，即至少约 605/672 个 15 分钟点。
- 实际时间跨度至少 167 小时。
- 最新声纳记录距预测时刻不超过 30 分钟。

实际状态机：

- 声纳尚未达标：继续存储声纳，预测使用 Flood Monitoring Lea Bridge；失败后尝试 EA Hydrology Lee Bridge。
- 声纳达标：预测改用 SQLite 中的真实声纳历史。
- 公开 API 故障且声纳尚未达标：允许声纳作为 `sonar_degraded` 应急源，并明确标记填充预测。
- API 和声纳都不可用：本轮预测失败；只有显式打开 `ALLOW_HISTORY_CSV_FALLBACK=1` 才使用训练参考历史。
- 降雨：EA Hydrology rainfall measures -> Open-Meteo precipitation。
- 潮位：Admiralty Sheerness -> Flood Monitoring Silvertown/Tower Pier tide。

运行 `python server.py --mode check-apis` 可以查看 `sonar.status`：`not_collected`、`warming_up`、`ready` 或 `error`。预测输出的 `data_quality.water_selection_mode` 会显示本轮实际使用 `api_primary`、`api_warmup`、`sonar_ready` 或 `sonar_degraded`。

当前 checkpoint 仍然是旧 15 特征模型，所以实时源会被整理进既有列名，不会新增模型输入维度。

## 10 分钟声纳重采样

你的声纳大约每 10 分钟更新一次，模型仍然需要 15 分钟输入。服务器现在会自动把原始水位重采样到 15 分钟：

```text
WATER_MQTT_TOPIC=v3/water-height-lora@ttn/devices/eui-a8610a3233458c03/up
WATER_RAW_INTERVAL_MINUTES=10
WATER_RESAMPLE_METHOD=mean
WATER_RECOMPUTE_FROM_DISTANCE=1
WATER_DISTANCE_FIELDS=distance,distance_mm,sonar_dist_mm,sonar_distance_mm
WATER_DISTANCE_UNIT=mm
SONAR_APPLY_MEDIAN_FILTER=1
SONAR_MEDIAN_FILTER_SIZE=5
```

默认 `mean` 与本地训练管线的 `resample("15min").mean()` 保持一致。可选：

- `mean`: 与训练最一致，默认推荐。
- `last`: 更贴近当前最新读数。
- `max`: 更保守，容易提前预警，但会比训练分布偏高。
- `median`: 更抗噪。

每次预测的 `data_quality` 会显示：

- `water_selection_mode`
- `water_selection_reason`
- `water_raw_points`
- `water_raw_interval_minutes`
- `water_resample_method`
- `water_source_points`
- `sonar_coverage_ratio`
- `sonar_observed_span_hours`
- `sonar_latest_age_minutes`
- `sonar_reasons`

TTN 报文会按以下结构解析：

```text
uplink_message.decoded_payload.distance -> distance_mm
received_at -> date
end_device_ids.device_id -> device_id
uplink_message.f_cnt -> f_cnt
uplink_message.decoded_payload.battery_* -> battery fields
```

水位换算：

```text
internal_water_m = SONAR_SENSOR_HEIGHT_M - distance_mm / 1000
```

详细后续步骤见 [TODO.md](</C:/UCL/Dissertation/cloud_flood_server/TODO.md>)。
