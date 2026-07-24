# House Mill 云端预测 TODO

## 目前服务器已经支持的数据源策略

默认策略是“声纳达标后切换”：

- 水位主输入：`PRIMARY_WATER_LEVEL_SOURCE=auto`
  - MQTT 声纳默认写入 `data/sonar_history.sqlite3`；SQLite 使用 WAL 和重复报文去重。
  - 声纳必须达到 90% 的 7 天覆盖率、至少 167 小时时间跨度，且最新点不超过 30 分钟，才会成为主输入。
  - 声纳积累期间会自动用 Environment Agency Flood Monitoring API 的 Lea Bridge 15 分钟水位：
    `5390TH-level-stage-i-15_min-mASD`
  - 如果 Flood Monitoring 不可用，会尝试 EA Hydrology 的 Lee Bridge 15 分钟水位：
    `f463f458-8115-42d8-834a-c6872116737d-level-i-900-m-qualified`
  - 两个 API 都不可用时，未达标但有数据的声纳会作为 `sonar_degraded` 应急输入。

- 降雨：`RAIN_SOURCE=auto`
  - 如果配置了 `EA_RAIN_MEASURE_IDS`，用 EA Hydrology rainfall measures。
  - 如果没有配置，自动用 Open-Meteo 15 分钟 precipitation。

- 潮位：`TIDE_SOURCE=auto`
  - 如果配置了 `ADMIRALTY_API_KEY`，优先用 Admiralty UK Tidal API。
  - 如果没有 key，自动用 Flood Monitoring 的 Silvertown 潮位：
    `0001-level-tidal_level-i-15_min-mAOD`

- 短历史：`ALLOW_SHORT_HISTORY_PAD=1`
  - 允许公开 API 或应急声纳在数据不完整时补齐 672 个 15 分钟点。
  - 正常切换到声纳仍必须通过覆盖率、跨度和新鲜度门槛。

## 第 1 步：准备云端环境

1. 在 DigitalOcean droplet 安装 Python 3.11、git、systemd 或 Docker。
2. 上传整个 `cloud_flood_server` 文件夹。
3. 上传或保留模型文件：
   - `HISTORY_CSV`
   - `MODEL_CHECKPOINT`
4. 复制配置文件：

   ```bash
   cp .env.example .env
   ```

5. 不要把 MQTT 密码、Admiralty key 写进代码，只写进 `.env`。

## 第 2 步：先跑 API 检查

```bash
python server.py --mode check-apis
```

需要看到：

- `primary_water_level.status = ok`
- `rain.status = ok`
- `tide.status = ok`

如果 Admiralty 还没 key，`admiralty.status = needs_key` 可以暂时忽略，因为 `TIDE_SOURCE=auto` 会用 Flood Monitoring tide fallback。

## 第 3 步：跑一次完整预测

```bash
python server.py --mode once
```

检查输出里是否有：

- `risk_level`
- `risk_label`
- `display`
- `eta_minutes`
- `max_predicted_m`
- `data_quality.water_source`

如果 `display = #`，表示未来 24h 没有达到真正洪水线。若有风险，`display` 会是类似 `15m`、`2h30m`。

## 第 4 步：接入你的水位 MQTT

填入：

```text
WATER_MQTT_HOST=
WATER_MQTT_PORT=
WATER_MQTT_TLS=1
WATER_MQTT_USERNAME=
WATER_MQTT_PASSWORD=
WATER_MQTT_TOPIC=v3/water-height-lora@ttn/devices/eui-a8610a3233458c03/up
```

声纳约 10 分钟更新一次，保持下面配置即可：

```text
WATER_RAW_INTERVAL_MINUTES=10
WATER_RESAMPLE_METHOD=mean
WATER_RECOMPUTE_FROM_DISTANCE=1
WATER_DISTANCE_FIELDS=distance,distance_mm,sonar_dist_mm,sonar_distance_mm
WATER_DISTANCE_UNIT=mm
SONAR_APPLY_MEDIAN_FILTER=1
```

服务器会把 10 分钟原始点聚合到 15 分钟模型网格。默认 `mean` 与训练管线一致；如果你希望预警更保守，可以把 `WATER_RESAMPLE_METHOD` 改成 `max`。

你给的 TTN 报文里，声纳距离在：

```text
uplink_message.decoded_payload.distance
```

单位按毫米处理，例如 `4309` 会存为 `distance_mm=4309`，并换算：

```text
internal_water_m = SONAR_SENSOR_HEIGHT_M - 4309 / 1000
```

然后运行：

```bash
python server.py --mode collect-water-mqtt
```

它会把实时水位写入：

```text
data/sonar_history.sqlite3
```

积累未达标时使用公开 API 预测；满足覆盖率、跨度和新鲜度条件后自动切换为真实 MQTT 声纳。已有 `data/sonar_history.csv` 会在首次 SQLite 写入时自动导入。

## 第 5 步：接入 ESP32 发布 MQTT

填入：

```text
PUBLISH_ENABLED=1
ALERT_MQTT_HOST=
ALERT_MQTT_PORT=
ALERT_MQTT_TLS=1
ALERT_MQTT_USERNAME=
ALERT_MQTT_PASSWORD=
ALERT_MQTT_TOPIC=housemill/flood/forecast
```

ESP32 订阅 `ALERT_MQTT_TOPIC`，收到 JSON 后：

- `risk_level = 0`：显示 `#`
- `risk_level > 0`：显示 `display` 和风险等级
- 每分钟用 `next_flood_utc` 在 ESP32 本地更新倒计时

## 第 6 步：云端常驻运行

建议两个进程：

```bash
python server.py --mode collect-water-mqtt
python server.py --mode worker
```

`worker` 会：

- 每半小时重新预测一次。
- 如果有风险，每分钟更新 ETA 并发布 MQTT。

生产环境建议用 systemd 或 Docker Compose 管理这两个进程。

## 第 7 步：后续模型改进

当前服务器把不同来源的数据强行整理成现有 15 特征输入，以保持 checkpoint 可用。下一轮训练建议：

1. 把 `flood_monitoring:5390TH` Lea Bridge 水位加入训练集。
2. 把 `0001` Silvertown 或 `0007` Tower Pier tide 加入训练集。
3. 把 Open-Meteo precipitation 与 EA rainfall 同时保留，做消融比较。
4. 重新训练 PatchTST，保存新的 `FEATURE_COLUMNS`、scaler、checkpoint。
5. 用新 checkpoint 替换 `.env` 里的 `MODEL_CHECKPOINT`。

## 第 8 步：最终演示检查

演示前运行：

```bash
python server.py --mode check-apis
python server.py --mode once
```

确认：

- API 有数据。
- MQTT 发布成功。
- ESP32 能收到 payload。
- `risk_level`、`display`、`eta_minutes` 展示符合预期。
