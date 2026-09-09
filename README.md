# CASA0022 Dissertation - FloodPredict

FloodPredict is a localised 24-hour water-level forecasting and warning system for the House Mill, a Grade I listed tidal mill in London. The project combines on-site sonar measurements, rainfall and tidal data, deep-learning forecasting, cloud deployment, and physical Internet of Things (IoT) devices.

The main research question is whether a site-specific system can provide useful flood warnings for a single heritage building. The final workflow uses seven days of observations at 15-minute intervals to predict the following 24 hours of internal water level. Forecasts are converted into risk levels and an estimated time to flooding (ETA), then published to a physical warning console through MQTT.

> **Research prototype:** This system is an academic prototype. It is not a certified flood-warning service and should not be used as the only basis for safety decisions.

![FloodPredict system workflow](figure/workflow.png)

## Project Components

The repository contains five connected parts:

1. **Historical data pipeline** - cleans House Mill sonar measurements and combines them with rainfall, tide, river-level, time, and moon-phase features.
2. **Forecasting models** - includes LSTM developmental baselines and a modified PatchTST model for 24-hour sequence forecasting.
3. **Real-time cloud pipeline** - collects live sonar data through MQTT, obtains environmental API data, runs PatchTST inference every 30 minutes, and publishes warning messages.
4. **Forecast evaluation** - matches archived 24-hour forecasts with later sonar observations and reports regression, lead-time, and risk metrics.
5. **Physical communication** - uses two ESP32-S3 devices to display live warnings and demonstrate predicted flooding with a water-based House Mill model.

## Data Pipeline

The historical dataset covers 6 June 2023 to 31 December 2025 and is resampled to 15-minute intervals. The main target is the internal House Mill water level derived from a MaxBotix MB7389 ultrasonic sonar sensor.

The preprocessing pipeline:

- treats sonar distances above 4,800 mm as unreliable and replaces them with the calibrated 5,000 mm reference;
- applies a five-point median filter to reduce short sensor spikes;
- converts distance into internal water level;
- groups levels below 1.0 m into a shared low-water modelling state;
- adds rainfall totals and 1-hour, 6-hour, 24-hour, and 7-day rainfall accumulations;
- adds tidal, river-level, cyclical time, and moon-phase features.

The offline experimental dataset contains 31 model channels. The later deployment pipeline uses a reduced 15-feature configuration because some historical inputs are not reliably available from live APIs. The two pipelines and checkpoints should therefore be treated as related but separate model versions.

## Forecasting Models

### LSTM baselines

The project began with an LSTM baseline based on established rainfall-runoff forecasting research. A second LSTM was developed to produce a complete 96-step forecast and to place more attention on rare high-water windows through weighted sampling and loss weighting. These models are retained as developmental comparisons.

### Modified PatchTST

PatchTST is the main forecasting model. It divides each input variable into overlapping temporal patches before applying Transformer encoding. The final offline configuration uses:

| Setting | Value |
| --- | ---: |
| Input history | 672 steps / 7 days |
| Forecast horizon | 96 steps / 24 hours |
| Sampling interval | 15 minutes |
| Offline input channels | 31 |
| Patch length / stride | 16 / 8 |
| Encoder layers | 2 |
| Attention heads | 4 |
| Model dimension | 32 |
| Feed-forward dimension | 128 |
| Dropout | 0.2 |

The modified model uses an asymmetric Huber-based regression loss to give more attention to high-water errors and underprediction. A separate classification head acts as a confidence check for predicted threshold crossings without replacing the continuous water-level forecast.

![Modified PatchTST architecture](figure/patchtst.png)

## Risk Communication

The server converts each 24-hour forecast into four operational risk levels:

| Risk level | Main water-level condition |
| --- | --- |
| No risk | Forecast maximum below 4.20 m |
| Caution | Forecast maximum from 4.20 m to below 4.43 m |
| Warning | Forecast reaches the calibrated 4.43 m flood threshold |
| Severe | Forecast reaches 4.70 m or meets additional duration/ETA rules |

The current server payload uses the label `Watch` for the Caution level. ETA is calculated from the first predicted crossing of the 4.43 m Warning threshold.

## Physical Devices

The project includes two ESP32-S3 devices:

- **Warning console:** a TN-73 analogue VU meter displays flood ETA from 24 hours to zero. A 16-pixel WS2812B NeoPixel strip displays risk severity and the countdown state.
- **Exhibition device:** a pump circulates water between two tanks containing a 3D-printed House Mill model. Pump duration represents different risk scenarios, while a continuously open 6 mm drainage hole returns water to the lower reservoir.

The warning console can receive real forecasts directly from the cloud server through the `forecast` MQTT topic. In exhibition mode, the model controller publishes simulated pump state, risk, and ETA through the `status` topic. Remote commands affect only the exhibition mode and do not modify cloud forecast results.

![Completed warning console and exhibition device](figure/device.JPG)

Hardware designs and firmware are available in [`3DModels/`](3DModels/) and [`Code/`](Code/).

## Evaluation Results

The following results are those reported in the dissertation experiments.

### Offline evaluation

- PatchTST test MAE: **0.235 m**
- PatchTST test MSE: **0.131 m^2**
- Mean event recall across ten random year-long windows: **86.19%**
- Mean event precision: **81.50%**
- Mean warning lead time: **16.2 hours before the observed peak**
- Mean peak error: **-0.07 m**

The latest basic LSTM result had an MSE of 0.335 m^2. This is a developmental comparison rather than a controlled architecture benchmark because the LSTM and PatchTST experiments used different forecasting outputs.

### Live evaluation

The DigitalOcean deployment generated 817 forecast runs. After incomplete future windows were removed, 76,091 forecast-observation pairs were evaluated:

| Metric | Result |
| --- | ---: |
| MAE | 0.213 m |
| RMSE | 0.262 m |
| Pearson correlation | 0.974 |
| R-squared | 0.928 |
| MAE reduction against persistence | 79.2% |

No observed flood crossed the 4.20 m Caution threshold during the evaluated live period. The live results therefore demonstrate continuous water-level forecasting under non-flood conditions, but not live flood-event detection. Historical event testing provides the current evidence for recall, precision, and warning lead time.

![Normal and rapid-rise forecast comparison](figure/result_normal_vs_rapid_rise.png)

## Repository Structure

```text
CASA0022-Dissertation-FloodPredict/
|-- data/                         Historical parsers and preprocessing pipeline
|-- lstm/                         LSTM baselines and saved experimental results
|-- PatchTST-main-mod/
|   `-- PatchTST_supervised/      Modified PatchTST training and year testing
|-- cloud_flood_server/           Real-time collection, inference, API, and MQTT server
|-- forecast_evaluation/          Live forecast evaluation and plotting tools
|-- Code/                         ESP32-S3 firmware and MQTT control documentation
|-- 3DModels/                     Printable warning-console and exhibition models
|-- figure/                       Dissertation figures and diagrams
`-- latex/                        Dissertation LaTeX source and references
```

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/CrazyEpi/CASA0022-Dissertation-FloodPredict.git
cd CASA0022-Dissertation-FloodPredict
```

### 2. Build the historical dataset

Run the data pipeline from the `data` directory so its relative paths resolve correctly:

```powershell
cd data
python main_pipeline.py
```

The generated dataset is written to `data/house_mill_integrated_dataset.csv`.

### 3. Train the modified PatchTST model

```powershell
cd ..\PatchTST-main-mod\PatchTST_supervised
python -m pip install -r requirements.txt
python train_housemill_patchtst_mod.py
```

Before training on another computer, update the `SOURCE_CSV` constant in `train_housemill_patchtst_mod.py`. Some experimental scripts retain local absolute paths from the original dissertation environment.

### 4. Run the repeated historical event test

After updating `DATA_PATH` and `MODEL_PATH` in `YearTest.py` when necessary:

```powershell
python YearTest.py --runs 10 --seed 2026
```

The ten windows are selected from a dataset covering less than three years and may overlap. They test different temporal sections rather than representing ten independent years.

### 5. Run the cloud server locally

```powershell
cd ..\..\cloud_flood_server
Copy-Item .env.example .env
python -m pip install -r requirements.txt
.\run_local.bat --mode check-apis
.\run_local.bat --mode once
```

Fill in the required API and MQTT values in `.env` before using live sources. Detailed local, Docker, and DigitalOcean instructions are provided in [`cloud_flood_server/README.md`](cloud_flood_server/README.md).

### 6. Evaluate archived live forecasts

From the repository root:

```powershell
python -m pip install -r .\forecast_evaluation\requirements.txt
python .\forecast_evaluation\evaluate_forecasts.py
python .\forecast_evaluation\evaluate_high_water.py --threshold-m 2.0
```

See [`forecast_evaluation/README.md`](forecast_evaluation/README.md) for database export, matching, and output details.

## Configuration and Secrets

- Copy `cloud_flood_server/.env.example` to `.env`; never commit the completed `.env` file.
- Do not publish MQTT passwords, Wi-Fi credentials, API keys, or private server addresses.
- Check the ESP32 sketches for local credentials before making a fork public.
- Model checkpoints and scalers must match the feature order used during their training.

## Known Limitations

- Low-water sonar measurements in the historical dataset contain unstable long-distance readings. The applied clipping improves high-water modelling but removes detailed low-water variation.
- A 30-60 minute phase lag was observed during one rapid-rise period, although the model followed the general tide and peak.
- The live evaluation period contained no threshold-crossing flood event.
- The manually calibrated 4.43 m threshold requires further site validation.
- Informer and Autoformer are included as related experimental code, but controlled final comparisons were not completed.

## Further Documentation

- [Cloud server deployment](cloud_flood_server/README.md)
- [DigitalOcean deployment notes](cloud_flood_server/DEPLOY_DIGITALOCEAN.md)
- [Forecast evaluation](forecast_evaluation/README.md)
- [MQTT exhibition control](Code/MQTT_REMOTE_CONTROL.md)
- [Dissertation LaTeX source](latex/housemill_flood_prediction_with_figures.tex)

## Academic References

- Kratzert, F. et al. (2018). *Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks*. Hydrology and Earth System Sciences, 22, 6005-6022. https://doi.org/10.5194/hess-22-6005-2018
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*. International Conference on Learning Representations. https://arxiv.org/abs/2211.14730

## Author

Haoyu Hu<br>
CASA0022 Dissertation, The Bartlett Centre for Advanced Spatial Analysis, University College London
