# DigitalOcean Deployment

这个文件夹现在可以作为一个独立部署包复制到 DigitalOcean。

必须包含：

- `server.py`
- `.env`
- `.env.example`
- `requirements.txt`
- `Dockerfile`
- `docker-compose.yml`
- `patchtst_runtime/`
- `model/history_reference_15min.csv`
- `model/checkpoint_15min.pth`
- `data/`
- `logs/`

`.env` 里可以继续保留你本地的 Windows 模型路径；Linux 上程序检测到 `C:\...` 后会自动回退到 `model/` 目录里的随包模型文件。

## 方法 A：Docker Compose

```bash
cd /opt/housemill-flood/cloud_flood_server
docker compose build
docker compose up -d water_collector predictor_worker
```

可选启动 HTTP API：

```bash
docker compose up -d api
curl http://127.0.0.1:8080/health
```

检查日志：

```bash
docker compose logs -f water_collector
docker compose logs -f predictor_worker
```

## 方法 B：普通 Python

```bash
cd /opt/housemill-flood/cloud_flood_server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py --mode check-apis
python server.py --mode once
```

然后用 systemd 或 tmux 常驻：

```bash
python server.py --mode collect-water-mqtt
python server.py --mode worker
```

## 部署前检查

```bash
python server.py --mode check-apis
```

需要看到：

- `primary_water_level.status = ok`
- `rain.status = ok`
- `tide.status = ok`
- `mqtt.water_source_configured = true`
- `mqtt.alert_publish_configured = true`
- 初次部署时 `sonar.status = not_collected`；开始采集后为 `warming_up`；达到门槛后为 `ready`

再跑：

```bash
PUBLISH_ENABLED=0 python server.py --mode once
```

确认预测能输出：

- `risk_level`
- `display`
- `eta_minutes`
- `data_quality`
- `data_quality.water_selection_mode`，初期通常为 `api_primary` 或 `api_warmup`

最后把 `.env` 里的：

```text
PUBLISH_ENABLED=1
```

再启动 worker，让它正式发布到 ESP32 MQTT。

## 安全

`.dockerignore` 已排除 `.env`，Docker 镜像不会把密钥打进去。运行时通过 `env_file: .env` 注入密钥。

`data/` 通过 Docker Compose 绑定到宿主机，因此 `sonar_history.sqlite3` 会在容器重建后保留。SQLite 已启用 WAL，可安全支持采集进程写入和预测进程读取。备份时应暂停 `water_collector` 后复制数据库，或使用 SQLite 的 `.backup` 命令，避免遗漏尚在 WAL 中的数据。

## 2GB Droplet 内存配置

服务器已采用模型延迟加载：`water_collector` 和 `check-apis` 不加载 PyTorch，只有 `predictor_worker`、`once` 和可选 HTTP API 加载模型。2GB 服务器不要常驻启动可选 `api` 服务，也不要在 worker 运行时并行执行多个 `--mode once`。

建议添加 2GB swap 作为峰值保护：

```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

检查内存和 OOM：

```bash
free -h
swapon --show
docker stats --no-stream
journalctl -k --since "60 minutes ago" --no-pager | grep -Ei 'oom|out of memory|killed process'
```

如果修改了 `.env`，必须重新创建容器才能载入新环境变量：

```bash
docker compose up -d --build --force-recreate water_collector predictor_worker
```
