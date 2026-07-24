from __future__ import annotations

import json
import math
import os
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import argparse
import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT.parent
RUNTIME_ROOT = APP_ROOT / "patchtst_runtime"
DATA_ROOT = APP_ROOT / "data"
MODEL_ROOT = APP_ROOT / "model"

DEFAULT_HISTORY_CSV = MODEL_ROOT / "history_reference_15min.csv"
DEFAULT_CHECKPOINT = MODEL_ROOT / "checkpoint_15min.pth"

TARGET_COLUMN = "internal_water_m"
FEATURE_COLUMNS = [
    "sheerness_tidal_m",
    "hour_sin",
    "hour_cos",
    "year_sin",
    "year_cos",
    "month_sin",
    "month_cos",
    "moon_phase",
    "catchment_rain_mm",
    "catchment_rain_mm_1h_sum",
    "catchment_rain_mm_6h_sum",
    "catchment_rain_mm_24h_sum",
    "catchment_rain_mm_7d_sum",
    "runoff_conversion_index",
    TARGET_COLUMN,
]

WATCH_LEVEL_M = 4.20
WARNING_LEVEL_M = 4.43
SEVERE_LEVEL_M = 4.70
INTERVAL_MINUTES = 15
SEQ_LEN = 672
PRED_LEN = 96
LABEL_LEN = 96


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(APP_ROOT / ".env")


def env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default.expanduser()
    if os.name != "nt" and len(raw) >= 3 and raw[1] == ":" and raw[2] in {"\\", "/"}:
        return default.expanduser()
    configured = Path(raw).expanduser()
    if not configured.is_absolute():
        configured = APP_ROOT / configured
    if configured.exists():
        return configured
    if default.exists():
        return default.expanduser()
    return configured


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime | pd.Timestamp | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (datetime, pd.Timestamp)):
        return iso_utc(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def http_get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 20) -> Any:
    req_headers = {
        "User-Agent": os.getenv("USER_AGENT", "UCL-HouseMill-FloodPredictor/0.1")
    }
    if headers:
        req_headers.update(headers)
    retries = max(1, int(os.getenv("HTTP_RETRIES", "3")))
    backoff = max(0.0, float(os.getenv("HTTP_RETRY_BACKOFF_SECONDS", "1")))
    last_error: Exception | None = None
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < retries - 1:
            time.sleep(backoff * (2**attempt))
    raise RuntimeError(f"HTTP request failed after {retries} attempts: {last_error}")


def floor_to_interval(dt: datetime, minutes: int = INTERVAL_MINUTES) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    minute = (dt.minute // minutes) * minutes
    return dt.replace(minute=minute)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def split_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def env_datetime_utc(name: str) -> datetime | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    ts = pd.to_datetime(raw, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"{name} is not a valid datetime: {raw}")
    return ts.to_pydatetime()


def ensure_data_root() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)


def normalize_measure_ref(measure: str) -> str:
    measure = measure.strip()
    if not measure:
        return measure
    return measure.rstrip("/").split("/")[-1]


def extract_datetime_and_value(row: dict[str, Any]) -> tuple[pd.Timestamp | None, float | None]:
    time_keys = [
        "dateTime",
        "datetime",
        "time",
        "date",
        "timestamp",
        "eventDateTime",
        "eventTime",
    ]
    value_keys = [
        "value",
        "height",
        "Height",
        "tidalHeight",
        "waterLevel",
        "level",
        "Level",
    ]

    ts = None
    for key in time_keys:
        if key in row and row[key] not in (None, ""):
            ts = pd.to_datetime(row[key], utc=True, errors="coerce")
            break

    value = None
    for key in value_keys:
        if key in row and row[key] not in (None, ""):
            try:
                value = float(row[key])
                break
            except (TypeError, ValueError):
                pass

    if ts is None or pd.isna(ts):
        return None, value
    return ts, value


def list_from_api_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("items", "value", "readings", "data", "stations", "events", "tidalEvents"):
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def approx_moon_phase(ts: pd.Timestamp) -> float:
    """Return approximate lunar phase in [0, 1). Good enough for online parity."""
    if ts.tzinfo is None:
        dt = ts.to_pydatetime().replace(tzinfo=timezone.utc)
    else:
        dt = ts.to_pydatetime().astimezone(timezone.utc)
    known_new_moon = datetime(2000, 1, 6, 18, 14, tzinfo=timezone.utc)
    synodic_month_days = 29.53058867
    days = (dt - known_new_moon).total_seconds() / 86400.0
    return float((days % synodic_month_days) / synodic_month_days)


def recompute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    df["year_sin"] = np.sin(2 * np.pi * idx.dayofyear / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * idx.dayofyear / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * idx.day / 31.0)
    df["month_cos"] = np.cos(2 * np.pi * idx.day / 31.0)
    return df


def recompute_rain_features(df: pd.DataFrame) -> pd.DataFrame:
    if "catchment_rain_mm" not in df.columns:
        df["catchment_rain_mm"] = 0.0
    windows = {
        "1h": 4,
        "6h": 24,
        "24h": 96,
        "7d": 672,
    }
    for suffix, steps in windows.items():
        df[f"catchment_rain_mm_{suffix}_sum"] = (
            df["catchment_rain_mm"].rolling(window=steps, min_periods=1).sum()
        )
    df["runoff_conversion_index"] = (
        df["catchment_rain_mm_1h_sum"] * df["catchment_rain_mm_7d_sum"]
    )
    return df


def prepare_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "date" not in df.columns:
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "date"})
        else:
            raise ValueError("Input data must include a date column.")

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last").set_index("date")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = recompute_time_features(df)
    if "moon_phase" not in df.columns or df["moon_phase"].isna().any():
        df["moon_phase"] = [approx_moon_phase(ts) for ts in df.index]
    df = recompute_rain_features(df)

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[FEATURE_COLUMNS]
    df = df.interpolate(method="linear").ffill().bfill()
    return df


def fetch_ea_measure_readings(
    measure: str, start_utc: datetime, end_utc: datetime
) -> pd.DataFrame:
    measure_ref = normalize_measure_ref(measure)
    if measure.lower().startswith("http") and "readings" in measure.lower():
        base_url = measure
    else:
        base_url = (
            "https://environment.data.gov.uk/hydrology/id/measures/"
            f"{urllib.parse.quote(measure_ref, safe='')}/readings.json"
        )

    params = {
        "mineq-date": iso_utc(start_utc),
        "maxeq-date": iso_utc(end_utc),
        "_view": "full",
        "_limit": os.getenv("EA_READINGS_LIMIT", "100000"),
    }
    separator = "&" if "?" in base_url else "?"
    url = base_url + separator + urllib.parse.urlencode(params)
    rows = list_from_api_payload(http_get_json(url))

    records = []
    for row in rows:
        ts, value = extract_datetime_and_value(row)
        if ts is None or value is None:
            continue
        records.append((ts, value))

    if not records:
        return pd.DataFrame(columns=["date", measure_ref])

    df = pd.DataFrame(records, columns=["date", measure_ref])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df[measure_ref] = pd.to_numeric(df[measure_ref], errors="coerce")
    return df.dropna(subset=[measure_ref]).sort_values("date")


def fetch_flood_monitoring_measure_readings(
    measure: str,
    start_utc: datetime | None = None,
    end_utc: datetime | None = None,
    latest: bool = False,
) -> pd.DataFrame:
    measure_ref = normalize_measure_ref(measure)
    if measure.lower().startswith("http") and "readings" in measure.lower():
        base_url = measure
    else:
        base_url = (
            "https://environment.data.gov.uk/flood-monitoring/id/measures/"
            f"{urllib.parse.quote(measure_ref, safe='')}/readings.json"
        )

    params: dict[str, str] = {}
    if latest:
        params["latest"] = ""
    else:
        if start_utc is not None:
            params["since"] = iso_utc(start_utc) or ""
        if end_utc is not None:
            # The Flood Monitoring API does not expose maxeq-date on this route.
            # We filter the upper bound locally after download.
            pass
        params["_sorted"] = ""
        params["_limit"] = os.getenv("FLOOD_MONITORING_READINGS_LIMIT", "10000")

    encoded = []
    for key, value in params.items():
        encoded.append(key if value == "" else f"{key}={urllib.parse.quote(value)}")
    separator = "&" if "?" in base_url else "?"
    url = base_url + (separator + "&".join(encoded) if encoded else "")
    rows = list_from_api_payload(http_get_json(url))

    records = []
    for row in rows:
        ts, value = extract_datetime_and_value(row)
        if ts is None or value is None:
            continue
        if end_utc is not None and ts.to_pydatetime() > end_utc:
            continue
        records.append((ts, value))

    if not records:
        return pd.DataFrame(columns=["date", measure_ref])

    df = pd.DataFrame(records, columns=["date", measure_ref])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df[measure_ref] = pd.to_numeric(df[measure_ref], errors="coerce")
    return df.dropna(subset=[measure_ref]).sort_values("date")


def fetch_ea_rain_history(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    measures = split_env_list("EA_RAIN_MEASURE_IDS")
    if not measures:
        if env_bool("REQUIRE_RAIN_API", False):
            raise RuntimeError("EA_RAIN_MEASURE_IDS is required for live API mode.")
        return pd.DataFrame(columns=["date", "catchment_rain_mm"])

    parts = []
    for measure in measures:
        df = fetch_ea_measure_readings(measure, start_utc, end_utc)
        if df.empty:
            continue
        value_col = [col for col in df.columns if col != "date"][0]
        parts.append(df.rename(columns={value_col: normalize_measure_ref(measure)}))

    if not parts:
        raise RuntimeError("EA rainfall API returned no usable readings.")

    merged = parts[0]
    for part in parts[1:]:
        merged = pd.merge_asof(
            merged.sort_values("date"),
            part.sort_values("date"),
            on="date",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=7),
        )

    value_cols = [col for col in merged.columns if col != "date"]
    merged["catchment_rain_mm"] = merged[value_cols].mean(axis=1)
    return merged[["date", "catchment_rain_mm"]].dropna()


def fetch_open_meteo_rain_history(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    lat = float(os.getenv("OPEN_METEO_LAT", "51.527"))
    lon = float(os.getenv("OPEN_METEO_LONG", "-0.007"))
    params = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "minutely_15": "precipitation",
            "hourly": "precipitation",
            "past_days": min(92, max(1, math.ceil((end_utc - start_utc).total_seconds() / 86400))),
            "forecast_days": 1,
            "timezone": "UTC",
            "precipitation_unit": "mm",
        }
    )
    url = f"https://api.open-meteo.com/v1/forecast?{params}"
    payload = http_get_json(url)

    if isinstance(payload.get("minutely_15"), dict) and "time" in payload["minutely_15"]:
        block = payload["minutely_15"]
    elif isinstance(payload.get("hourly"), dict) and "time" in payload["hourly"]:
        block = payload["hourly"]
    else:
        return pd.DataFrame(columns=["date", "catchment_rain_mm"])

    values = block.get("precipitation") or []
    times = block.get("time") or []
    df = pd.DataFrame({"date": times, "catchment_rain_mm": values})
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["catchment_rain_mm"] = pd.to_numeric(df["catchment_rain_mm"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"])
    mask = (df["date"] >= pd.Timestamp(start_utc)) & (df["date"] <= pd.Timestamp(end_utc))
    return df.loc[mask, ["date", "catchment_rain_mm"]].sort_values("date")


def fetch_rain_history(start_utc: datetime, end_utc: datetime) -> pd.DataFrame:
    source = os.getenv("RAIN_SOURCE", "auto").strip().lower()
    errors = []
    if source in {"auto", "ea", "ea_hydrology", "hydrology"} and split_env_list("EA_RAIN_MEASURE_IDS"):
        try:
            return fetch_ea_rain_history(start_utc, end_utc)
        except Exception as exc:
            errors.append(f"ea_hydrology:{exc}")
            if source != "auto":
                raise

    if source in {"auto", "open_meteo", "open-meteo", "openmeteo"}:
        try:
            return fetch_open_meteo_rain_history(start_utc, end_utc)
        except Exception as exc:
            errors.append(f"open_meteo:{exc}")
            if source != "auto":
                raise

    if env_bool("REQUIRE_RAIN_API", False):
        raise RuntimeError("No rainfall API returned data. " + " | ".join(errors))
    return pd.DataFrame(columns=["date", "catchment_rain_mm"])


def fetch_external_level_history(start_utc: datetime, end_utc: datetime) -> dict[str, Any]:
    hydrology_measures = split_env_list("EA_WATER_LEVEL_MEASURE_IDS")
    flood_measures = split_env_list("FLOOD_MONITORING_LEVEL_MEASURE_IDS")

    results: dict[str, Any] = {
        "status": "not_configured",
        "hydrology_measures": len(hydrology_measures),
        "flood_monitoring_measures": len(flood_measures),
        "series": [],
        "errors": [],
    }

    for measure in hydrology_measures:
        try:
            df = fetch_ea_measure_readings(measure, start_utc, end_utc)
            value_col = [col for col in df.columns if col != "date"][0] if not df.empty else normalize_measure_ref(measure)
            latest = df.tail(1).iloc[0].to_dict() if not df.empty else None
            results["series"].append(
                {
                    "provider": "ea_hydrology",
                    "measure": normalize_measure_ref(measure),
                    "rows": int(len(df)),
                    "latest_utc": iso_utc(latest["date"]) if latest else None,
                    "latest_value_m": round(float(latest[value_col]), 4) if latest else None,
                }
            )
        except Exception as exc:
            results["errors"].append({"provider": "ea_hydrology", "measure": measure, "message": str(exc)})

    for measure in flood_measures:
        try:
            df = fetch_flood_monitoring_measure_readings(measure, start_utc, end_utc)
            value_col = [col for col in df.columns if col != "date"][0] if not df.empty else normalize_measure_ref(measure)
            latest = df.tail(1).iloc[0].to_dict() if not df.empty else None
            results["series"].append(
                {
                    "provider": "flood_monitoring",
                    "measure": normalize_measure_ref(measure),
                    "rows": int(len(df)),
                    "latest_utc": iso_utc(latest["date"]) if latest else None,
                    "latest_value_m": round(float(latest[value_col]), 4) if latest else None,
                }
            )
        except Exception as exc:
            results["errors"].append({"provider": "flood_monitoring", "measure": measure, "message": str(exc)})

    if results["series"] and not results["errors"]:
        results["status"] = "ok"
    elif results["series"]:
        results["status"] = "partial"
    elif results["errors"]:
        results["status"] = "error"
    return results


def build_data_quality(history: pd.DataFrame) -> dict[str, Any]:
    quality: dict[str, Any] = {
        "status": "data_available",
        "messages": [],
        "external_levels": None,
    }
    if "date" in history.columns:
        dates = pd.to_datetime(history["date"], utc=True, errors="coerce").dropna()
        string_fields = [
            "water_source",
            "water_resample_method",
            "water_selection_mode",
            "water_selection_reason",
            "water_api_error",
            "sonar_storage",
            "sonar_latest_utc",
            "sonar_error",
        ]
        integer_fields = [
            "water_source_points",
            "water_raw_points",
            "sonar_total_points",
            "sonar_window_raw_points",
            "sonar_aligned_points",
        ]
        float_fields = [
            "water_raw_interval_minutes",
            "water_source_coverage_ratio",
            "water_observed_span_hours",
            "water_latest_age_minutes",
            "sonar_coverage_ratio",
            "sonar_observed_span_hours",
            "sonar_latest_age_minutes",
            "sonar_minimum_coverage",
            "sonar_minimum_span_hours",
            "sonar_maximum_age_minutes",
        ]
        for field in string_fields:
            if field in history.columns and not history[field].dropna().empty:
                quality[field] = str(history[field].dropna().iloc[-1])
        for field in integer_fields:
            if field in history.columns and not history[field].dropna().empty:
                quality[field] = int(float(history[field].dropna().iloc[-1]))
        for field in float_fields:
            if field in history.columns and not history[field].dropna().empty:
                quality[field] = round(float(history[field].dropna().iloc[-1]), 4)
        for field in ("sonar_available", "sonar_ready"):
            if field in history.columns and not history[field].dropna().empty:
                quality[field] = bool(history[field].dropna().iloc[-1])
        if "sonar_reasons" in history.columns and not history["sonar_reasons"].dropna().empty:
            raw_reasons = history["sonar_reasons"].dropna().iloc[-1]
            try:
                quality["sonar_reasons"] = json.loads(raw_reasons) if isinstance(raw_reasons, str) else list(raw_reasons)
            except (TypeError, ValueError, json.JSONDecodeError):
                quality["sonar_reasons"] = [str(raw_reasons)]

        selection_mode = quality.get("water_selection_mode")
        if selection_mode == "api_warmup":
            quality["messages"].append(
                "Sonar is still warming up; prediction is using the public water-level API."
            )
        elif selection_mode == "sonar_degraded":
            quality["status"] = "degraded"
            quality["messages"].append(
                "Public water-level APIs failed; prediction is using padded incomplete sonar history."
            )
        elif selection_mode == "history_fallback":
            quality["status"] = "degraded"
            quality["messages"].append(
                "Prediction is using static reference history because operational water sources failed."
            )

        if not dates.empty:
            span_hours = (dates.max() - dates.min()).total_seconds() / 3600.0
            latest_age_minutes = (now_utc() - dates.max().to_pydatetime()).total_seconds() / 60.0
            quality["history_span_hours"] = round(span_hours, 2)
            quality["history_latest_utc"] = iso_utc(dates.max())
            quality["history_latest_age_minutes"] = round(latest_age_minutes, 1)
            observed_span = quality.get("water_observed_span_hours", span_hours)
            if observed_span < 24 * 7 - 1:
                quality["messages"].append("Less than seven days of source water-level history; sequence was padded.")
            source_age = quality.get("water_latest_age_minutes", latest_age_minutes)
            if env_bool("USE_LIVE_API", False) and source_age > 60:
                quality["messages"].append("Latest source water-level timestamp is older than 60 minutes.")
        else:
            quality["status"] = "no_history"

    end_utc = floor_to_interval(env_datetime_utc("LIVE_END_UTC") or now_utc())
    start_utc = end_utc - timedelta(hours=24)
    external = fetch_external_level_history(start_utc, end_utc)
    if external["status"] != "not_configured":
        quality["external_levels"] = external
    return quality


def build_admiralty_url(station_id: str, start_utc: datetime, end_utc: datetime) -> str:
    template = os.getenv(
        "ADMIRALTY_TIDAL_EVENTS_ENDPOINT_TEMPLATE",
        "https://admiraltyapi.azure-api.net/uktidalapi/api/V1/Stations/{station_id}/TidalEvents",
    )
    url = template.format(
        station_id=urllib.parse.quote(station_id, safe=""),
        start=urllib.parse.quote(iso_utc(start_utc) or "", safe=""),
        end=urllib.parse.quote(iso_utc(end_utc) or "", safe=""),
        start_date=start_utc.date().isoformat(),
        end_date=end_utc.date().isoformat(),
    )
    query = os.getenv("ADMIRALTY_TIDAL_EVENTS_QUERY", "")
    if query:
        separator = "&" if "?" in url else "?"
        url = url + separator + query.format(
            start=urllib.parse.quote(iso_utc(start_utc) or "", safe=""),
            end=urllib.parse.quote(iso_utc(end_utc) or "", safe=""),
            start_date=start_utc.date().isoformat(),
            end_date=end_utc.date().isoformat(),
        ).lstrip("?")
    return url


def fetch_admiralty_tide_history(
    start_utc: datetime, end_utc: datetime, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    key = os.getenv("ADMIRALTY_API_KEY", "")
    station_id = os.getenv("ADMIRALTY_STATION_ID") or os.getenv("ADMIRALTY_STATION_NAME", "Sheerness")
    if not key:
        if env_bool("REQUIRE_TIDE_API", False):
            raise RuntimeError("ADMIRALTY_API_KEY is required for live tide API mode.")
        return pd.DataFrame(columns=["date", "sheerness_tidal_m"])

    url = build_admiralty_url(station_id, start_utc - timedelta(days=1), end_utc + timedelta(days=1))
    payload = http_get_json(
        url,
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Accept": "application/json",
        },
    )
    rows = list_from_api_payload(payload)

    records = []
    for row in rows:
        ts, value = extract_datetime_and_value(row)
        if ts is None or value is None:
            continue
        records.append((ts, value))

    if len(records) < 2:
        raise RuntimeError(
            "Admiralty tide API returned fewer than two usable tide points. "
            "Check ADMIRALTY_STATION_ID and ADMIRALTY_TIDAL_EVENTS_ENDPOINT_TEMPLATE."
        )

    events = pd.DataFrame(records, columns=["date", "height_m"]).dropna()
    events["date"] = pd.to_datetime(events["date"], utc=True)
    events = events.drop_duplicates("date").sort_values("date").set_index("date")

    grid = pd.DataFrame(index=target_index)
    combined = pd.concat([events, grid], axis=0).sort_index()
    combined["height_m"] = combined["height_m"].interpolate(method="time").ffill().bfill()
    tide = combined.loc[target_index, "height_m"].rename("sheerness_tidal_m").reset_index()
    tide = tide.rename(columns={"index": "date"})
    return tide


def fetch_flood_monitoring_tide_history(
    start_utc: datetime, end_utc: datetime, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    measure = os.getenv("TIDE_FLOOD_MONITORING_MEASURE_ID", "0001-level-tidal_level-i-15_min-mAOD")
    df = fetch_flood_monitoring_measure_readings(measure, start_utc, end_utc)
    if df.empty:
        return pd.DataFrame(columns=["date", "sheerness_tidal_m"])
    value_col = [col for col in df.columns if col != "date"][0]
    events = df[["date", value_col]].rename(columns={value_col: "height_m"})
    events = events.dropna().drop_duplicates("date").sort_values("date").set_index("date")
    grid = pd.DataFrame(index=target_index)
    combined = pd.concat([events, grid], axis=0).sort_index()
    combined["height_m"] = combined["height_m"].interpolate(method="time").ffill().bfill()
    tide = combined.loc[target_index, "height_m"].rename("sheerness_tidal_m").reset_index()
    return tide.rename(columns={"index": "date"})


def fetch_tide_history(
    start_utc: datetime, end_utc: datetime, target_index: pd.DatetimeIndex
) -> pd.DataFrame:
    source = os.getenv("TIDE_SOURCE", "auto").strip().lower()
    errors = []
    if source in {"auto", "admiralty", "ukho"} and os.getenv("ADMIRALTY_API_KEY", ""):
        try:
            return fetch_admiralty_tide_history(start_utc, end_utc, target_index)
        except Exception as exc:
            errors.append(f"admiralty:{exc}")
            if source != "auto":
                raise

    if source in {"auto", "flood_monitoring", "flood-monitoring", "flood"}:
        try:
            return fetch_flood_monitoring_tide_history(start_utc, end_utc, target_index)
        except Exception as exc:
            errors.append(f"flood_monitoring:{exc}")
            if source != "auto":
                raise

    if env_bool("REQUIRE_TIDE_API", False):
        raise RuntimeError("No tide API returned data. " + " | ".join(errors))
    return pd.DataFrame(columns=["date", "sheerness_tidal_m"])


def prepare_sonar_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", TARGET_COLUMN])
    if "date" not in df.columns:
        for candidate in ("_time", "time", "timestamp"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
    if "date" not in df.columns:
        raise ValueError("Sonar history CSV must contain date, _time, time, or timestamp.")

    distance_col = None
    for candidate in ("distance_mm", "sonar_dist_mm", "sonar_dist_mm_cleaned"):
        if candidate in df.columns:
            distance_col = candidate
            break

    recompute_from_distance = env_bool("WATER_RECOMPUTE_FROM_DISTANCE", True)
    if distance_col is not None and (TARGET_COLUMN not in df.columns or recompute_from_distance):
        sensor_height = float(os.getenv("SONAR_SENSOR_HEIGHT_M", "5.0"))
        max_threshold = float(os.getenv("SONAR_MAX_THRESHOLD_MM", "4800"))
        fallback_distance = float(os.getenv("SONAR_DEFAULT_FALLBACK_MM", "5000"))
        median_window = int(os.getenv("SONAR_MEDIAN_FILTER_SIZE", "5"))
        distance = pd.to_numeric(df[distance_col], errors="coerce")
        has_distance = distance.notna()
        distance = distance.mask(distance > max_threshold, fallback_distance)
        if env_bool("SONAR_APPLY_MEDIAN_FILTER", True):
            distance = distance.ffill().bfill().rolling(
                window=max(1, median_window), center=True, min_periods=1
            ).median()
        recomputed = (sensor_height - distance / 1000.0).clip(lower=0.0)
        existing = pd.to_numeric(df.get(TARGET_COLUMN), errors="coerce")
        df[TARGET_COLUMN] = recomputed.where(has_distance, existing)
    elif TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Sonar history CSV must contain {TARGET_COLUMN} or a distance_mm column."
        )

    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    return df[["date", TARGET_COLUMN]].dropna().sort_values("date")


def load_sonar_history_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Sonar history CSV not found: {path}")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return prepare_sonar_history_frame(pd.read_csv(path))
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"Unable to read sonar history CSV {path}: {last_error}")


def sonar_storage_mode() -> str:
    mode = os.getenv("SONAR_STORAGE", "sqlite").strip().lower()
    if mode not in {"sqlite", "csv"}:
        raise ValueError("SONAR_STORAGE must be sqlite or csv.")
    return mode


def sonar_database_path() -> Path:
    return env_path("SONAR_DATABASE", DATA_ROOT / "sonar_history.sqlite3")


def sonar_csv_path() -> Path:
    return env_path("SONAR_HISTORY_CSV", DATA_ROOT / "sonar_history.csv")


def prediction_database_path() -> Path:
    raw = os.getenv("PREDICTION_DATABASE", "").strip()
    if not raw:
        return DATA_ROOT / "prediction_history.sqlite3"
    configured = Path(raw).expanduser()
    return configured if configured.is_absolute() else APP_ROOT / configured


def connect_prediction_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id TEXT PRIMARY KEY,
            forecast_generated_utc TEXT NOT NULL,
            history_last_utc TEXT,
            valid_until_utc TEXT,
            site TEXT,
            source TEXT,
            risk_level INTEGER,
            risk_label TEXT,
            max_predicted_m REAL,
            eta_minutes INTEGER,
            next_flood_utc TEXT,
            data_quality_json TEXT,
            payload_json TEXT NOT NULL,
            stored_at_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_points (
            run_id TEXT NOT NULL,
            target_utc TEXT NOT NULL,
            lead_minutes INTEGER NOT NULL,
            predicted_water_m REAL NOT NULL,
            flood_probability REAL,
            PRIMARY KEY (run_id, target_utc),
            FOREIGN KEY (run_id) REFERENCES forecast_runs(run_id)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_points_target "
        "ON forecast_points(target_utc)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecast_runs_generated "
        "ON forecast_runs(forecast_generated_utc)"
    )
    connection.commit()
    return connection


def archive_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    if not env_bool("PREDICTION_ARCHIVE_ENABLED", True):
        return {"archived": False, "reason": "PREDICTION_ARCHIVE_ENABLED is disabled"}

    forecast = payload.get("forecast")
    if not isinstance(forecast, list) or not forecast:
        return {"archived": False, "reason": "Prediction payload has no forecast points"}

    generated_utc = payload.get("forecast_generated_utc")
    if not generated_utc:
        return {"archived": False, "reason": "Prediction payload has no generation time"}

    run_identity = "|".join(
        [
            str(payload.get("site", "")),
            str(generated_utc),
            str(payload.get("history_last_utc", "")),
        ]
    )
    run_id = hashlib.sha256(run_identity.encode("utf-8")).hexdigest()
    history_last = pd.to_datetime(payload.get("history_last_utc"), utc=True, errors="coerce")
    interval_minutes = int(
        payload.get("model", {}).get("interval_minutes", INTERVAL_MINUTES)
    )
    points: list[dict[str, Any]] = []
    for index, row in enumerate(forecast):
        target = pd.to_datetime(row.get("time_utc"), utc=True, errors="coerce")
        water_level = optional_finite_number(row.get("water_level_m"))
        if pd.isna(target) or water_level is None:
            continue
        if pd.isna(history_last):
            lead_minutes = (index + 1) * interval_minutes
        else:
            lead_minutes = max(
                0, int(round((target - history_last).total_seconds() / 60.0))
            )
        points.append(
            {
                "run_id": run_id,
                "target_utc": iso_utc(target),
                "lead_minutes": lead_minutes,
                "predicted_water_m": water_level,
                "flood_probability": optional_finite_number(
                    row.get("flood_probability")
                ),
            }
        )

    if not points:
        return {"archived": False, "reason": "Prediction contains no valid forecast points"}

    database = prediction_database_path()
    connection = connect_prediction_database(database)
    try:
        run_parameters = {
            "run_id": run_id,
            "forecast_generated_utc": str(generated_utc),
            "history_last_utc": payload.get("history_last_utc"),
            "valid_until_utc": payload.get("valid_until_utc"),
            "site": payload.get("site"),
            "source": payload.get("source"),
            "risk_level": optional_finite_number(
                payload.get("risk_level"), integer=True
            ),
            "risk_label": payload.get("risk_label"),
            "max_predicted_m": optional_finite_number(
                payload.get("max_predicted_m")
            ),
            "eta_minutes": optional_finite_number(
                payload.get("eta_minutes"), integer=True
            ),
            "next_flood_utc": payload.get("next_flood_utc"),
            "data_quality_json": json.dumps(
                json_safe(payload.get("data_quality", {})),
                sort_keys=True,
                separators=(",", ":"),
            ),
            "payload_json": json.dumps(
                json_safe(payload), sort_keys=True, separators=(",", ":")
            ),
            "stored_at_utc": iso_utc(now_utc()),
        }
        with connection:
            connection.execute(
                """
                INSERT INTO forecast_runs (
                    run_id, forecast_generated_utc, history_last_utc,
                    valid_until_utc, site, source, risk_level, risk_label,
                    max_predicted_m, eta_minutes, next_flood_utc,
                    data_quality_json, payload_json, stored_at_utc
                ) VALUES (
                    :run_id, :forecast_generated_utc, :history_last_utc,
                    :valid_until_utc, :site, :source, :risk_level, :risk_label,
                    :max_predicted_m, :eta_minutes, :next_flood_utc,
                    :data_quality_json, :payload_json, :stored_at_utc
                )
                ON CONFLICT(run_id) DO UPDATE SET
                    valid_until_utc=excluded.valid_until_utc,
                    risk_level=excluded.risk_level,
                    risk_label=excluded.risk_label,
                    max_predicted_m=excluded.max_predicted_m,
                    eta_minutes=excluded.eta_minutes,
                    next_flood_utc=excluded.next_flood_utc,
                    data_quality_json=excluded.data_quality_json,
                    payload_json=excluded.payload_json,
                    stored_at_utc=excluded.stored_at_utc
                """,
                run_parameters,
            )
            connection.execute(
                "DELETE FROM forecast_points WHERE run_id = ?", (run_id,)
            )
            connection.executemany(
                """
                INSERT INTO forecast_points (
                    run_id, target_utc, lead_minutes,
                    predicted_water_m, flood_probability
                ) VALUES (
                    :run_id, :target_utc, :lead_minutes,
                    :predicted_water_m, :flood_probability
                )
                """,
                points,
            )
    finally:
        connection.close()

    return {
        "archived": True,
        "run_id": run_id,
        "points": len(points),
        "database": str(database),
    }


def connect_sonar_database(path: Path, create: bool = False) -> sqlite3.Connection:
    if not create and not path.exists():
        raise FileNotFoundError(f"Sonar SQLite database not found: {path}")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30.0)
    connection.execute("PRAGMA busy_timeout=30000")
    if create:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sonar_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_key TEXT NOT NULL UNIQUE,
                date_utc TEXT NOT NULL,
                internal_water_m REAL NOT NULL,
                distance_mm REAL,
                device_id TEXT,
                dev_eui TEXT,
                f_cnt INTEGER,
                packet_count INTEGER,
                topic TEXT,
                payload_json TEXT NOT NULL,
                stored_at_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_sonar_readings_date ON sonar_readings(date_utc)"
        )
        connection.commit()
    return connection


def load_sonar_history_database(path: Path) -> pd.DataFrame:
    connection = connect_sonar_database(path, create=False)
    try:
        frame = pd.read_sql_query(
            """
            SELECT date_utc AS date, internal_water_m, distance_mm
            FROM sonar_readings
            ORDER BY date_utc, id
            """,
            connection,
        )
    finally:
        connection.close()
    return prepare_sonar_history_frame(frame)


def load_configured_sonar_history() -> tuple[pd.DataFrame, str]:
    mode = sonar_storage_mode()
    if mode == "csv":
        path = sonar_csv_path()
        return load_sonar_history_csv(path), f"mqtt_csv:{path}"

    database = sonar_database_path()
    if database.exists():
        frame = load_sonar_history_database(database)
        if not frame.empty:
            return frame, f"mqtt_sqlite:{database}"

    legacy_csv = sonar_csv_path()
    if env_bool("SONAR_IMPORT_LEGACY_CSV", True) and legacy_csv.exists():
        frame = load_sonar_history_csv(legacy_csv)
        if not frame.empty:
            return frame, f"mqtt_csv_legacy:{legacy_csv}"

    if database.exists():
        return pd.DataFrame(columns=["date", TARGET_COLUMN]), f"mqtt_sqlite:{database}"
    raise FileNotFoundError(
        f"No sonar storage found at {database} or legacy CSV {legacy_csv}."
    )


def fetch_primary_api_water_history(start_utc: datetime, end_utc: datetime) -> tuple[pd.DataFrame, str]:
    source = os.getenv("PRIMARY_WATER_LEVEL_SOURCE", "auto").strip().lower()
    flood_measure = (
        os.getenv("PRIMARY_FLOOD_MONITORING_LEVEL_MEASURE_ID")
        or (split_env_list("FLOOD_MONITORING_LEVEL_MEASURE_IDS")[:1] or [""])[0]
        or "5390TH-level-stage-i-15_min-mASD"
    )
    hydrology_measure = (
        os.getenv("PRIMARY_EA_WATER_LEVEL_MEASURE_ID")
        or (split_env_list("EA_WATER_LEVEL_MEASURE_IDS")[:1] or [""])[0]
        or "f463f458-8115-42d8-834a-c6872116737d-level-i-900-m-qualified"
    )

    errors = []
    if source in {"auto", "flood_monitoring", "flood-monitoring", "flood"}:
        try:
            df = fetch_flood_monitoring_measure_readings(flood_measure, start_utc, end_utc)
            value_col = [col for col in df.columns if col != "date"][0] if not df.empty else None
            if value_col:
                return df.rename(columns={value_col: TARGET_COLUMN})[["date", TARGET_COLUMN]], f"flood_monitoring:{normalize_measure_ref(flood_measure)}"
        except Exception as exc:
            errors.append(f"flood_monitoring:{exc}")
            if source != "auto":
                raise

    if source in {"auto", "ea_hydrology", "hydrology", "ea"}:
        try:
            df = fetch_ea_measure_readings(hydrology_measure, start_utc, end_utc)
            value_col = [col for col in df.columns if col != "date"][0] if not df.empty else None
            if value_col:
                return df.rename(columns={value_col: TARGET_COLUMN})[["date", TARGET_COLUMN]], f"ea_hydrology:{normalize_measure_ref(hydrology_measure)}"
        except Exception as exc:
            errors.append(f"ea_hydrology:{exc}")
            if source != "auto":
                raise

    raise RuntimeError("No API water-level source returned data. " + " | ".join(errors))


@dataclass
class WaterSourceSelection:
    data: pd.DataFrame
    source_label: str
    mode: str
    diagnostics: dict[str, Any]


def evaluate_sonar_readiness(
    water: pd.DataFrame, start_utc: datetime, end_utc: datetime, storage_label: str
) -> dict[str, Any]:
    minimum_coverage = float(os.getenv("SONAR_READY_MIN_COVERAGE", "0.90"))
    minimum_span_hours = float(os.getenv("SONAR_READY_MIN_SPAN_HOURS", "167.0"))
    maximum_age_minutes = float(os.getenv("SONAR_READY_MAX_AGE_MINUTES", "30"))
    maximum_future_minutes = float(
        os.getenv("SONAR_READY_MAX_FUTURE_MINUTES", str(INTERVAL_MINUTES))
    )
    report: dict[str, Any] = {
        "storage": storage_label,
        "available": False,
        "ready": False,
        "total_points": int(len(water)),
        "window_raw_points": 0,
        "aligned_points": 0,
        "coverage_ratio": 0.0,
        "observed_span_hours": 0.0,
        "latest_utc": None,
        "latest_age_minutes": None,
        "minimum_coverage": minimum_coverage,
        "minimum_span_hours": minimum_span_hours,
        "maximum_age_minutes": maximum_age_minutes,
        "reasons": [],
    }
    if water.empty:
        report["reasons"].append("no_valid_sonar_points")
        return report

    source = water[["date", TARGET_COLUMN]].copy()
    source["date"] = pd.to_datetime(source["date"], utc=True, errors="coerce")
    source[TARGET_COLUMN] = pd.to_numeric(source[TARGET_COLUMN], errors="coerce")
    source = source.dropna().sort_values("date")
    future_limit = pd.Timestamp(end_utc) + pd.Timedelta(minutes=maximum_future_minutes)
    window = source[
        (source["date"] >= pd.Timestamp(start_utc)) & (source["date"] <= future_limit)
    ]
    report["window_raw_points"] = int(len(window))
    if window.empty:
        report["reasons"].append("no_sonar_points_in_prediction_window")
        return report

    latest = window["date"].max()
    earliest = window["date"].min()
    latest_age_minutes = (pd.Timestamp(end_utc) - latest).total_seconds() / 60.0
    span_hours = (latest - earliest).total_seconds() / 3600.0
    indexed = window.set_index("date")[[TARGET_COLUMN]]
    indexed = indexed[~indexed.index.duplicated(keep="last")]
    grid = pd.date_range(
        start=start_utc, end=end_utc, freq=f"{INTERVAL_MINUTES}min", tz="UTC"
    )
    aligned = resample_water_to_model_interval(indexed).reindex(grid)
    aligned_points = int(aligned[TARGET_COLUMN].notna().sum())
    coverage = aligned_points / len(grid) if len(grid) else 0.0

    report.update(
        {
            "available": aligned_points > 0,
            "aligned_points": aligned_points,
            "coverage_ratio": round(float(coverage), 4),
            "observed_span_hours": round(float(span_hours), 2),
            "latest_utc": iso_utc(latest),
            "latest_age_minutes": round(float(latest_age_minutes), 1),
        }
    )
    if coverage < minimum_coverage:
        report["reasons"].append("coverage_below_threshold")
    if span_hours < minimum_span_hours:
        report["reasons"].append("span_below_seven_days")
    if latest_age_minutes > maximum_age_minutes:
        report["reasons"].append("latest_point_is_stale")
    if latest_age_minutes < -maximum_future_minutes:
        report["reasons"].append("latest_point_is_too_far_in_future")
    report["ready"] = not report["reasons"]
    return report


def selection_diagnostics(
    mode: str,
    reason: str,
    sonar_report: dict[str, Any] | None = None,
    api_error: str | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "water_selection_mode": mode,
        "water_selection_reason": reason,
    }
    if sonar_report:
        for key, value in sonar_report.items():
            diagnostics[f"sonar_{key}"] = value
    if api_error:
        diagnostics["water_api_error"] = api_error
    return diagnostics


def choose_water_level_history(start_utc: datetime, end_utc: datetime) -> WaterSourceSelection:
    source = os.getenv("PRIMARY_WATER_LEVEL_SOURCE", "auto").strip().lower()
    valid_sources = {
        "auto",
        "mqtt",
        "flood_monitoring",
        "flood-monitoring",
        "flood",
        "ea_hydrology",
        "hydrology",
        "ea",
    }
    if source not in valid_sources:
        raise ValueError(f"Unsupported PRIMARY_WATER_LEVEL_SOURCE: {source}")

    sonar = pd.DataFrame(columns=["date", TARGET_COLUMN])
    sonar_label = f"mqtt_{sonar_storage_mode()}"
    sonar_error: str | None = None
    sonar_report: dict[str, Any] | None = None
    if source in {"auto", "mqtt"}:
        try:
            sonar, sonar_label = load_configured_sonar_history()
            sonar_report = evaluate_sonar_readiness(
                sonar, start_utc, end_utc, sonar_label
            )
        except Exception as exc:
            sonar_error = str(exc)
            sonar_report = evaluate_sonar_readiness(
                sonar, start_utc, end_utc, sonar_label
            )
            sonar_report["error"] = sonar_error

    if source == "mqtt":
        if sonar_report and sonar_report["available"]:
            return WaterSourceSelection(
                sonar,
                sonar_label,
                "sonar_explicit",
                selection_diagnostics(
                    "sonar_explicit",
                    "PRIMARY_WATER_LEVEL_SOURCE explicitly requires MQTT sonar.",
                    sonar_report,
                ),
            )
        raise RuntimeError(f"MQTT sonar source has no usable data: {sonar_error or sonar_report}")

    if source == "auto" and sonar_report and sonar_report["ready"]:
        return WaterSourceSelection(
            sonar,
            sonar_label,
            "sonar_ready",
            selection_diagnostics(
                "sonar_ready",
                "Sonar passed coverage, span, and freshness gates.",
                sonar_report,
            ),
        )

    if source in valid_sources - {"mqtt"}:
        api_error: str | None = None
        try:
            api_data, api_label = fetch_primary_api_water_history(start_utc, end_utc)
            mode = "api_warmup" if source == "auto" and sonar_report and sonar_report["available"] else "api_primary"
            reason = (
                "Sonar is still warming up; public API remains the operational source."
                if mode == "api_warmup"
                else "Public API is the configured operational water-level source."
            )
            return WaterSourceSelection(
                api_data,
                api_label,
                mode,
                selection_diagnostics(mode, reason, sonar_report),
            )
        except Exception as exc:
            api_error = str(exc)
            if source != "auto":
                raise

        if (
            sonar_report
            and sonar_report["available"]
            and env_bool("SONAR_ALLOW_DEGRADED_FALLBACK", True)
        ):
            return WaterSourceSelection(
                sonar,
                sonar_label,
                "sonar_degraded",
                selection_diagnostics(
                    "sonar_degraded",
                    "Public API failed; using incomplete sonar history as an emergency fallback.",
                    sonar_report,
                    api_error,
                ),
            )

    if env_bool("ALLOW_HISTORY_CSV_FALLBACK", False):
        fallback = env_path("HISTORY_CSV", DEFAULT_HISTORY_CSV)
        return WaterSourceSelection(
            load_sonar_history_csv(fallback),
            f"history_csv:{fallback}",
            "history_fallback",
            selection_diagnostics(
                "history_fallback",
                "Operational water sources failed; using reference history fallback.",
                sonar_report,
            ),
        )

    raise RuntimeError(
        "No water-level source is available. Sonar is not ready and public APIs failed. "
        f"Sonar error: {sonar_error or 'none'}."
    )


def infer_median_interval_minutes(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    diffs = index.sort_values().to_series().diff().dropna()
    if diffs.empty:
        return None
    return round(float(diffs.median().total_seconds() / 60.0), 2)


def resample_water_to_model_interval(source: pd.DataFrame) -> pd.DataFrame:
    method = os.getenv("WATER_RESAMPLE_METHOD", "mean").strip().lower()
    rule = f"{INTERVAL_MINUTES}min"
    series = source[[TARGET_COLUMN]].sort_index()

    if method == "last":
        return series.resample(rule).last()
    if method == "first":
        return series.resample(rule).first()
    if method == "median":
        return series.resample(rule).median()
    if method == "max":
        return series.resample(rule).max()
    if method == "min":
        return series.resample(rule).min()
    return series.resample(rule).mean()


def align_water_history_to_grid(
    water: pd.DataFrame,
    grid: pd.DatetimeIndex,
    source_label: str,
    selection_metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    source = water.set_index("date").sort_index()
    source = source[~source.index.duplicated(keep="last")]
    window_end = grid[-1] + pd.Timedelta(minutes=INTERVAL_MINUTES)
    source = source[(source.index >= grid[0]) & (source.index < window_end)]
    raw_points = int(source[TARGET_COLUMN].notna().sum())
    raw_interval = infer_median_interval_minutes(source.index)
    observed_span_hours = 0.0
    latest_age_minutes: float | None = None
    if raw_points:
        observed_span_hours = (
            source.index.max() - source.index.min()
        ).total_seconds() / 3600.0
        latest_age_minutes = (
            grid[-1] - source.index.max()
        ).total_seconds() / 60.0
    resampled = resample_water_to_model_interval(source)
    aligned = resampled.reindex(grid)
    available = int(aligned[TARGET_COLUMN].notna().sum())
    if available == 0:
        raise RuntimeError(f"Water-level source {source_label} has no readings in the requested window.")

    selection_mode = (selection_metadata or {}).get("water_selection_mode")
    allow_degraded_pad = selection_mode == "sonar_degraded" and env_bool(
        "SONAR_ALLOW_DEGRADED_FALLBACK", True
    )
    if available < SEQ_LEN * 0.9 and not (
        env_bool("ALLOW_SHORT_HISTORY_PAD", True) or allow_degraded_pad
    ):
        raise RuntimeError(
            f"Water-level source {source_label} only has {available} aligned points. "
            "Set ALLOW_SHORT_HISTORY_PAD=1 to pad missing history."
        )

    aligned[TARGET_COLUMN] = aligned[TARGET_COLUMN].interpolate(limit_direction="both").ffill().bfill()
    out = aligned.reset_index().rename(columns={"index": "date"})
    out["water_source"] = source_label
    out["water_raw_points"] = raw_points
    out["water_raw_interval_minutes"] = raw_interval if raw_interval is not None else np.nan
    out["water_resample_method"] = os.getenv("WATER_RESAMPLE_METHOD", "mean").strip().lower()
    out["water_source_points"] = available
    out["water_source_coverage_ratio"] = round(available / len(grid), 4)
    out["water_observed_span_hours"] = round(observed_span_hours, 2)
    out["water_latest_age_minutes"] = (
        round(latest_age_minutes, 1) if latest_age_minutes is not None else np.nan
    )
    for key, value in (selection_metadata or {}).items():
        if isinstance(value, (list, dict)):
            value = json.dumps(json_safe(value), separators=(",", ":"))
        out[key] = value
    return out


def build_live_history() -> pd.DataFrame:
    end_utc = floor_to_interval(env_datetime_utc("LIVE_END_UTC") or now_utc())
    start_utc = end_utc - timedelta(minutes=(SEQ_LEN - 1) * INTERVAL_MINUTES)
    grid = pd.date_range(start=start_utc, end=end_utc, freq=f"{INTERVAL_MINUTES}min", tz="UTC")

    selection = choose_water_level_history(start_utc, end_utc)
    live = align_water_history_to_grid(
        selection.data,
        grid,
        selection.source_label,
        selection.diagnostics,
    )

    rain = fetch_rain_history(start_utc, end_utc)
    if not rain.empty:
        rain = rain.set_index("date").sort_index()
        rain_15 = rain.resample(f"{INTERVAL_MINUTES}min").sum().reindex(grid).fillna(0.0)
        live["catchment_rain_mm"] = rain_15["catchment_rain_mm"].values

    tide = fetch_tide_history(start_utc, end_utc, grid)
    if not tide.empty:
        live = live.merge(tide, on="date", how="left")

    reference = pd.read_csv(env_path("HISTORY_CSV", DEFAULT_HISTORY_CSV))
    reference_features = prepare_feature_frame(reference)
    for col in FEATURE_COLUMNS:
        if col not in live.columns:
            if col in reference_features.columns:
                live[col] = reference_features[col].tail(len(live)).values
            else:
                live[col] = 0.0

    metadata_cols = [
        col for col in live.columns if col != "date" and col not in FEATURE_COLUMNS
    ]
    return live[["date"] + metadata_cols + FEATURE_COLUMNS]


def max_consecutive_true(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for item in mask:
        if bool(item):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def format_eta(minutes: int | None) -> str:
    if minutes is None:
        return "#"
    minutes = max(0, int(minutes))
    hours = minutes // 60
    rem = minutes % 60
    if hours == 0:
        return f"{rem}m"
    return f"{hours}h{rem:02d}m"


def compute_risk(forecast: list[dict[str, Any]]) -> dict[str, Any]:
    values = np.array([row["water_level_m"] for row in forecast], dtype=float)
    times = [pd.Timestamp(row["time_utc"]) for row in forecast]

    max_level = float(np.nanmax(values))
    warning_mask = values >= WARNING_LEVEL_M
    watch_mask = values >= WATCH_LEVEL_M

    warning_idx = np.where(warning_mask)[0]
    watch_idx = np.where(watch_mask)[0]
    first_warning = int(warning_idx[0]) if len(warning_idx) else None
    first_watch = int(watch_idx[0]) if len(watch_idx) else None

    eta_minutes = None
    next_flood_utc = None
    if first_warning is not None:
        eta_minutes = (first_warning + 1) * INTERVAL_MINUTES
        next_flood_utc = times[first_warning]

    watch_eta_minutes = None
    next_watch_utc = None
    if first_watch is not None:
        watch_eta_minutes = (first_watch + 1) * INTERVAL_MINUTES
        next_watch_utc = times[first_watch]

    above_warning_minutes = max_consecutive_true(warning_mask) * INTERVAL_MINUTES

    if (
        max_level >= SEVERE_LEVEL_M
        or above_warning_minutes >= 120
        or (eta_minutes is not None and eta_minutes <= 120)
    ):
        risk_level = 3
        risk_label = "Severe"
    elif max_level >= WARNING_LEVEL_M or above_warning_minutes >= 60:
        risk_level = 2
        risk_label = "Warning"
    elif max_level >= WATCH_LEVEL_M:
        risk_level = 1
        risk_label = "Watch"
    else:
        risk_level = 0
        risk_label = "No risk"

    if risk_level == 0:
        reason = f"Forecast peak {max_level:.2f}m is below watch level {WATCH_LEVEL_M:.2f}m."
    elif risk_level == 1:
        reason = (
            f"Forecast peak {max_level:.2f}m approaches the flood line "
            f"but remains below {WARNING_LEVEL_M:.2f}m."
        )
    elif risk_level == 2:
        reason = (
            f"Forecast reaches {max_level:.2f}m or remains above "
            f"{WARNING_LEVEL_M:.2f}m for {above_warning_minutes} minutes."
        )
    else:
        reason = (
            f"Severe trigger: peak {max_level:.2f}m, warning ETA "
            f"{eta_minutes} minutes, or {above_warning_minutes} minutes above threshold."
        )

    return {
        "risk_level": risk_level,
        "risk_label": risk_label,
        "risk_reason": reason,
        "display": format_eta(eta_minutes) if eta_minutes is not None else "#",
        "eta_minutes": eta_minutes,
        "watch_eta_minutes": watch_eta_minutes,
        "next_flood_utc": iso_utc(next_flood_utc),
        "next_watch_utc": iso_utc(next_watch_utc),
        "max_predicted_m": round(max_level, 3),
        "watch_level_m": WATCH_LEVEL_M,
        "warning_level_m": WARNING_LEVEL_M,
        "severe_level_m": SEVERE_LEVEL_M,
        "duration_above_warning_minutes": above_warning_minutes,
    }


@dataclass
class PredictorConfig:
    history_csv: Path
    checkpoint: Path
    use_gpu: bool = False


class PatchTSTPredictor:
    def __init__(self, config: PredictorConfig):
        self.config = config
        self.available = False
        self.error: str | None = None
        self.scaler: StandardScaler | None = None
        self.model: Any = None
        self.device: Any = None
        self._load()

    def _args(self) -> SimpleNamespace:
        return SimpleNamespace(
            enc_in=len(FEATURE_COLUMNS),
            seq_len=SEQ_LEN,
            label_len=LABEL_LEN,
            pred_len=PRED_LEN,
            e_layers=2,
            n_heads=4,
            d_model=32,
            d_ff=128,
            dropout=0.2,
            fc_dropout=0.2,
            head_dropout=0.0,
            individual=0,
            patch_len=16,
            stride=8,
            padding_patch="end",
            revin=0,
            affine=0,
            subtract_last=0,
            decomposition=0,
            kernel_size=25,
        )

    def _load(self) -> None:
        try:
            if not self.config.history_csv.exists():
                raise FileNotFoundError(f"History CSV not found: {self.config.history_csv}")
            if not self.config.checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.config.checkpoint}")

            import torch
            import importlib

            torch_threads = max(1, int(os.getenv("TORCH_NUM_THREADS", "1")))
            torch.set_num_threads(torch_threads)
            try:
                torch.set_num_interop_threads(
                    max(1, int(os.getenv("TORCH_NUM_INTEROP_THREADS", "1")))
                )
            except RuntimeError:
                pass

            sys.path.insert(0, str(RUNTIME_ROOT))
            patchtst_module = importlib.import_module("models.PatchTST")

            reference = pd.read_csv(self.config.history_csv)
            reference_features = prepare_feature_frame(reference)
            train_end = int(len(reference_features) * 0.7)
            self.scaler = StandardScaler()
            self.scaler.fit(reference_features.iloc[:train_end].values)

            self.device = torch.device(
                "cuda:0" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
            )
            self.model = patchtst_module.Model(self._args()).float().to(self.device)
            state = torch.load(self.config.checkpoint, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
            self.available = True
            self.error = None
        except Exception as exc:  # keep HTTP server alive even if model is missing
            self.available = False
            self.error = str(exc)

    def predict(self, history: pd.DataFrame) -> dict[str, Any]:
        if not self.available or self.model is None or self.scaler is None:
            raise RuntimeError(f"Predictor unavailable: {self.error}")
        frame = prepare_feature_frame(history)
        if len(frame) < SEQ_LEN:
            raise RuntimeError(f"Need at least {SEQ_LEN} rows; received {len(frame)}.")

        import torch

        tail = frame.tail(SEQ_LEN)
        scaled = self.scaler.transform(tail.values)
        batch_x = torch.tensor(scaled, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            outputs, clf_outputs = self.model(batch_x)
            target_scaled = outputs[:, -PRED_LEN:, -1].detach().cpu().numpy()[0]
            clf_prob = torch.sigmoid(clf_outputs[:, -PRED_LEN:, 0]).detach().cpu().numpy()[0]

        mean = self.scaler.mean_[-1]
        scale = self.scaler.scale_[-1]
        water_levels = target_scaled * scale + mean
        if not np.isfinite(water_levels).all():
            raise RuntimeError("Model output contains NaN or infinite values; check live feature inputs.")

        last_ts = tail.index[-1]
        future_times = pd.date_range(
            start=last_ts + pd.Timedelta(minutes=INTERVAL_MINUTES),
            periods=PRED_LEN,
            freq=f"{INTERVAL_MINUTES}min",
            tz="UTC",
        )

        forecast = []
        for ts, level, prob in zip(future_times, water_levels, clf_prob):
            forecast.append(
                {
                    "time_utc": iso_utc(ts),
                    "water_level_m": round(float(level), 4),
                    "flood_probability": round(float(prob), 4),
                }
            )

        alert = compute_risk(forecast)
        generated = now_utc()
        return {
            "site": os.getenv("SITE_ID", "house_mill"),
            "status": "ok",
            "source": "patchtst_15min",
            "forecast_generated_utc": iso_utc(generated),
            "valid_until_utc": iso_utc(generated + timedelta(minutes=30)),
            "history_last_utc": iso_utc(last_ts),
            "model": {
                "checkpoint": str(self.config.checkpoint),
                "history_csv": str(self.config.history_csv),
                "seq_len": SEQ_LEN,
                "pred_len": PRED_LEN,
                "interval_minutes": INTERVAL_MINUTES,
            },
            **alert,
            "forecast": forecast,
        }


class RuntimeService:
    def __init__(self) -> None:
        self.history_csv = env_path("HISTORY_CSV", DEFAULT_HISTORY_CSV)
        self.checkpoint = env_path("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT)
        self.predictor = PatchTSTPredictor(
            PredictorConfig(
                history_csv=self.history_csv,
                checkpoint=self.checkpoint,
                use_gpu=os.getenv("USE_GPU", "0") == "1",
            )
        )
        self.state_lock = threading.Lock()
        self.last_payload: dict[str, Any] | None = None
        self.last_error: str | None = None
        self.next_recheck_utc: datetime | None = None
        self.background_started = False

    def load_history(self) -> pd.DataFrame:
        if env_bool("USE_LIVE_API", False):
            return build_live_history()
        return pd.read_csv(self.history_csv)

    def run_prediction(self) -> dict[str, Any]:
        history = self.load_history()
        payload = self.predictor.predict(history)
        payload["data_quality"] = build_data_quality(history)
        try:
            archive_status = archive_prediction(payload)
        except Exception as exc:
            archive_status = {"archived": False, "reason": str(exc)}
        payload["archive_status"] = archive_status
        publish_status = publish_alert(payload)
        payload["publish_status"] = publish_status
        with self.state_lock:
            self.last_payload = payload
            self.last_error = None
            self.next_recheck_utc = now_utc() + timedelta(minutes=30)
        return payload

    def update_countdown(self) -> dict[str, Any] | None:
        with self.state_lock:
            if not self.last_payload:
                return None
            payload = dict(self.last_payload)

        next_flood = payload.get("next_flood_utc")
        if next_flood:
            flood_dt = pd.Timestamp(next_flood).to_pydatetime()
            if flood_dt.tzinfo is None:
                flood_dt = flood_dt.replace(tzinfo=timezone.utc)
            eta = math.ceil((flood_dt - now_utc()).total_seconds() / 60)
            payload["eta_minutes"] = max(0, eta)
            payload["display"] = format_eta(payload["eta_minutes"])
            payload["countdown_updated_utc"] = iso_utc(now_utc())

        publish_status = publish_alert(payload)
        payload["publish_status"] = publish_status
        with self.state_lock:
            self.last_payload = payload
        return payload

    def background_loop(self) -> None:
        while True:
            try:
                with self.state_lock:
                    should_recheck = (
                        self.next_recheck_utc is None or now_utc() >= self.next_recheck_utc
                    )
                    has_risk = bool(
                        self.last_payload and self.last_payload.get("risk_level", 0) > 0
                    )
                if should_recheck:
                    payload = self.run_prediction()
                    publish_status = payload.get("publish_status", {})
                    archive_status = payload.get("archive_status", {})
                    print(
                        "[worker] prediction "
                        f"risk={payload.get('risk_level')} "
                        f"display={payload.get('display')} "
                        f"water_source={payload.get('data_quality', {}).get('water_source')} "
                        f"archived={archive_status.get('archived')} "
                        f"archive_reason={archive_status.get('reason')} "
                        f"published={publish_status.get('published')} "
                        f"publish_reason={publish_status.get('reason')}",
                        flush=True,
                    )
                elif has_risk:
                    payload = self.update_countdown()
                    if payload:
                        publish_status = payload.get("publish_status", {})
                        print(
                            "[worker] countdown "
                            f"eta_minutes={payload.get('eta_minutes')} "
                            f"published={publish_status.get('published')} "
                            f"publish_reason={publish_status.get('reason')}",
                            flush=True,
                        )
            except Exception as exc:
                with self.state_lock:
                    self.last_error = str(exc)
                print(f"[worker] error: {exc}", file=sys.stderr, flush=True)
            time.sleep(60)

    def start_background(self) -> None:
        if self.background_started or os.getenv("BACKGROUND_LOOP", "1") != "1":
            return
        thread = threading.Thread(target=self.background_loop, daemon=True)
        thread.start()
        self.background_started = True

    def status(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "predictor_available": self.predictor.available,
                "predictor_error": self.predictor.error,
                "history_csv": str(self.history_csv),
                "checkpoint": str(self.checkpoint),
                "last_error": self.last_error,
                "next_recheck_utc": iso_utc(self.next_recheck_utc),
                "last_payload": self.last_payload,
            }


_RUNTIME_SERVICE: RuntimeService | None = None
_RUNTIME_SERVICE_LOCK = threading.Lock()


def get_runtime_service() -> RuntimeService:
    global _RUNTIME_SERVICE
    if _RUNTIME_SERVICE is None:
        with _RUNTIME_SERVICE_LOCK:
            if _RUNTIME_SERVICE is None:
                _RUNTIME_SERVICE = RuntimeService()
    return _RUNTIME_SERVICE


def publish_alert(payload: dict[str, Any]) -> dict[str, Any]:
    if os.getenv("PUBLISH_ENABLED", "0") != "1":
        return {"published": False, "reason": "PUBLISH_ENABLED is not 1"}

    try:
        import paho.mqtt.client as mqtt
    except Exception as exc:
        return {"published": False, "reason": f"paho-mqtt unavailable: {exc}"}

    host = os.getenv("ALERT_MQTT_HOST") or os.getenv("PROF_MQTT_HOST", "")
    topic = os.getenv("ALERT_MQTT_TOPIC") or os.getenv("PROF_MQTT_TOPIC", "housemill/flood/forecast")
    port = int(os.getenv("ALERT_MQTT_PORT") or os.getenv("PROF_MQTT_PORT", "8883"))
    username = os.getenv("ALERT_MQTT_USERNAME") or os.getenv("PROF_MQTT_USERNAME", "")
    password = os.getenv("ALERT_MQTT_PASSWORD") or os.getenv("PROF_MQTT_PASSWORD", "")
    if not host:
        return {"published": False, "reason": "ALERT_MQTT_HOST is not configured"}

    compact = dict(payload)
    compact.pop("forecast", None)

    client: Any | None = None
    loop_started = False
    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except (AttributeError, TypeError):
            client = mqtt.Client()
        if username:
            client.username_pw_set(username, password=password or None)
        if (os.getenv("ALERT_MQTT_TLS") or os.getenv("PROF_MQTT_TLS", "1")) == "1":
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        client.connect(host, port, keepalive=30)
        client.loop_start()
        loop_started = True
        result = client.publish(topic, json.dumps(json_safe(compact)), qos=0, retain=True)
        result.wait_for_publish(timeout=10.0)
        published = result.rc == mqtt.MQTT_ERR_SUCCESS and result.is_published()
        return {
            "published": published,
            "topic": topic,
            "mqtt_rc": result.rc,
            "reason": None if published else "MQTT message was not acknowledged as sent.",
        }
    except Exception as exc:
        return {
            "published": False,
            "topic": topic,
            "reason": f"MQTT publish failed: {exc}",
        }
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
            if loop_started:
                try:
                    client.loop_stop()
                except Exception:
                    pass


def nested_get(data: dict[str, Any], path: list[str]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def convert_distance_to_mm(value: float, unit: str) -> float:
    unit = unit.strip().lower()
    if unit in {"m", "meter", "metre", "meters", "metres"}:
        return value * 1000.0
    if unit in {"cm", "centimeter", "centimetre", "centimeters", "centimetres"}:
        return value * 10.0
    return value


def extract_water_from_mqtt(topic: str, payload_bytes: bytes) -> dict[str, Any] | None:
    try:
        payload_text = payload_bytes.decode("utf-8")
        payload = json.loads(payload_text)
    except Exception:
        return None

    decoded = payload
    if isinstance(payload, dict):
        decoded = (
            payload.get("uplink_message", {}).get("decoded_payload")
            or payload.get("decoded_payload")
            or payload
        )
    if not isinstance(decoded, dict):
        return None

    uplink = payload.get("uplink_message", {}) if isinstance(payload, dict) else {}
    end_device_ids = payload.get("end_device_ids", {}) if isinstance(payload, dict) else {}
    application_ids = end_device_ids.get("application_ids", {}) if isinstance(end_device_ids, dict) else {}

    ts = (
        decoded.get("timestamp")
        or decoded.get("time")
        or payload.get("received_at")
        or uplink.get("received_at")
        or nested_get(payload, ["uplink_message", "settings", "time"])
        or iso_utc(now_utc())
    )
    record: dict[str, Any] = {
        "date": ts,
        "topic": topic,
        "device_id": end_device_ids.get("device_id"),
        "dev_eui": end_device_ids.get("dev_eui"),
        "application_id": application_ids.get("application_id"),
        "f_port": uplink.get("f_port"),
        "f_cnt": uplink.get("f_cnt"),
        "packet_count": decoded.get("packet_count"),
        "battery_percentage": decoded.get("battery_percentage"),
        "battery_voltage": decoded.get("battery_voltage"),
        "battery_status": decoded.get("battery_status"),
        "solar_status": decoded.get("solar_status"),
    }

    for key in ("internal_water_m", "water_level_m", "level_m"):
        if key in decoded:
            record[TARGET_COLUMN] = float(decoded[key])
            return record

    configured_distance_keys = split_env_list("WATER_DISTANCE_FIELDS")
    distance_keys = configured_distance_keys or [
        "distance_mm",
        "sonar_dist_mm",
        "sonar_distance_mm",
        "distance",
    ]
    distance_unit = os.getenv("WATER_DISTANCE_UNIT", "mm")
    for key in distance_keys:
        if key in decoded:
            distance_raw = float(decoded[key])
            distance = convert_distance_to_mm(distance_raw, distance_unit)
            sensor_height = float(os.getenv("SONAR_SENSOR_HEIGHT_M", "5.0"))
            record["distance_raw"] = distance_raw
            record["distance_raw_field"] = key
            record["distance_raw_unit"] = distance_unit
            record["distance_mm"] = distance
            record[TARGET_COLUMN] = max(0.0, sensor_height - distance / 1000.0)
            return record

    return None


def optional_finite_number(value: Any, integer: bool = False) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if integer else number


def sonar_database_parameters(record: dict[str, Any]) -> dict[str, Any]:
    timestamp = pd.to_datetime(record.get("date"), utc=True, errors="coerce")
    water_level = optional_finite_number(record.get(TARGET_COLUMN))
    if pd.isna(timestamp) or water_level is None:
        raise ValueError("Sonar record requires a valid date and internal_water_m value.")
    timestamp_utc = iso_utc(timestamp)
    identity = {
        "date": timestamp_utc,
        "device": record.get("dev_eui") or record.get("device_id"),
        "f_cnt": record.get("f_cnt"),
        "packet_count": record.get("packet_count"),
        "distance_mm": optional_finite_number(record.get("distance_mm")),
        TARGET_COLUMN: water_level,
    }
    event_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
    return {
        "event_key": event_key,
        "date_utc": timestamp_utc,
        "internal_water_m": water_level,
        "distance_mm": optional_finite_number(record.get("distance_mm")),
        "device_id": record.get("device_id"),
        "dev_eui": record.get("dev_eui"),
        "f_cnt": optional_finite_number(record.get("f_cnt"), integer=True),
        "packet_count": optional_finite_number(record.get("packet_count"), integer=True),
        "topic": record.get("topic"),
        "payload_json": json.dumps(
            json_safe(record), sort_keys=True, separators=(",", ":"), default=str
        ),
        "stored_at_utc": iso_utc(now_utc()),
    }


def insert_sonar_database_records(path: Path, records: list[dict[str, Any]]) -> int:
    if not records:
        return 0
    parameters = [sonar_database_parameters(record) for record in records]
    connection = connect_sonar_database(path, create=True)
    try:
        before = connection.total_changes
        with connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO sonar_readings (
                    event_key, date_utc, internal_water_m, distance_mm,
                    device_id, dev_eui, f_cnt, packet_count, topic,
                    payload_json, stored_at_utc
                ) VALUES (
                    :event_key, :date_utc, :internal_water_m, :distance_mm,
                    :device_id, :dev_eui, :f_cnt, :packet_count, :topic,
                    :payload_json, :stored_at_utc
                )
                """,
                parameters,
            )
        return connection.total_changes - before
    finally:
        connection.close()


def migrate_legacy_sonar_csv(path: Path) -> int:
    if not env_bool("SONAR_IMPORT_LEGACY_CSV", True):
        return 0
    legacy_path = sonar_csv_path()
    if not legacy_path.exists():
        return 0

    connection = connect_sonar_database(path, create=True)
    try:
        existing = int(connection.execute("SELECT COUNT(*) FROM sonar_readings").fetchone()[0])
    finally:
        connection.close()
    if existing:
        return 0

    legacy = load_sonar_history_csv(legacy_path)
    records = [
        {"date": row["date"], TARGET_COLUMN: row[TARGET_COLUMN], "topic": "legacy_csv_import"}
        for _, row in legacy.iterrows()
    ]
    return insert_sonar_database_records(path, records)


def append_sonar_record_csv(record: dict[str, Any]) -> None:
    path = sonar_csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    df = pd.DataFrame([record])
    if not exists:
        df.to_csv(path, mode="w", index=False, header=True)
        return

    existing_columns = list(pd.read_csv(path, nrows=0).columns)
    new_columns = [col for col in df.columns if col not in existing_columns]
    if new_columns:
        old = pd.read_csv(path)
        for col in new_columns:
            old[col] = np.nan
        for col in old.columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[list(old.columns)]
        temporary = path.with_suffix(path.suffix + ".tmp")
        pd.concat([old, df], ignore_index=True).to_csv(temporary, index=False)
        os.replace(temporary, path)
        return

    for col in existing_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = df[existing_columns]
    df.to_csv(path, mode="a", index=False, header=False)


def append_sonar_record(record: dict[str, Any]) -> None:
    ensure_data_root()
    if sonar_storage_mode() == "csv":
        append_sonar_record_csv(record)
        return

    database = sonar_database_path()
    migrate_legacy_sonar_csv(database)
    insert_sonar_database_records(database, [record])
    if env_bool("SONAR_CSV_MIRROR", False):
        append_sonar_record_csv(record)


def collect_water_mqtt_forever() -> None:
    try:
        import paho.mqtt.client as mqtt
    except Exception as exc:
        raise RuntimeError(f"paho-mqtt is required for MQTT collection: {exc}") from exc

    host = os.getenv("WATER_MQTT_HOST") or os.getenv("TTN_MQTT_HOST", "")
    port = int(os.getenv("WATER_MQTT_PORT") or os.getenv("TTN_MQTT_PORT", "8883"))
    username = os.getenv("WATER_MQTT_USERNAME") or os.getenv("TTN_MQTT_USERNAME", "")
    password = os.getenv("WATER_MQTT_PASSWORD") or os.getenv("TTN_MQTT_PASSWORD", "")
    topic = os.getenv("WATER_MQTT_TOPIC") or os.getenv("TTN_UPLINK_TOPIC", "#")
    if not host:
        raise RuntimeError("WATER_MQTT_HOST is not configured.")

    def on_connect(
        client: Any,
        _userdata: Any,
        _flags: Any,
        reason_code: Any,
        _properties: Any = None,
    ) -> None:
        rc = getattr(reason_code, "value", reason_code)
        if rc != 0:
            print(f"[water-mqtt] connect failed rc={rc}")
            return
        print(f"[water-mqtt] connected to {host}:{port}, subscribing {topic}")
        client.subscribe(topic, qos=0)

    def on_message(_client: Any, _userdata: Any, msg: Any) -> None:
        record = extract_water_from_mqtt(msg.topic, msg.payload)
        if not record:
            print(f"[water-mqtt] ignored payload on {msg.topic}")
            return
        append_sonar_record(record)
        print(
            f"[water-mqtt] {record['date']} {TARGET_COLUMN}="
            f"{record[TARGET_COLUMN]:.3f}m"
        )

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except (AttributeError, TypeError):
        client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    if username:
        client.username_pw_set(username, password=password or None)
    if (os.getenv("WATER_MQTT_TLS") or os.getenv("TTN_MQTT_TLS", "1")) == "1":
        client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
    client.connect(host, port, keepalive=60)
    client.loop_forever()


def check_admiralty_station(station_name: str) -> dict[str, Any]:
    key = os.getenv("ADMIRALTY_API_KEY", "")
    endpoint = os.getenv(
        "ADMIRALTY_STATIONS_ENDPOINT",
        "https://admiraltyapi.azure-api.net/uktidalapi/api/V1/Stations",
    )
    if not key:
        return {
            "status": "needs_key",
            "message": "Set ADMIRALTY_API_KEY and call this endpoint again.",
            "endpoint": endpoint,
        }

    data = http_get_json(
        endpoint,
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Accept": "application/json",
        },
    )

    if isinstance(data, dict):
        rows = data.get("items") or data.get("value") or data.get("stations") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    needle = station_name.lower()
    matches = []
    for row in rows:
        text = json.dumps(row, ensure_ascii=False).lower()
        if needle in text:
            matches.append(row)

    return {
        "status": "ok",
        "station_name": station_name,
        "match_count": len(matches),
        "matches": matches[:10],
    }


def compact_station_item(item: dict[str, Any], provider: str) -> dict[str, Any]:
    measures = []
    for measure in item.get("measures", []) or []:
        if not isinstance(measure, dict):
            continue
        mid = measure.get("@id") or measure.get("id") or ""
        parameter = measure.get("parameter") or measure.get("parameterName")
        period = measure.get("period")
        qualifier = measure.get("qualifier")
        unit = measure.get("unitName")
        if parameter and "level" not in str(parameter).lower():
            continue
        measures.append(
            {
                "id": normalize_measure_ref(str(mid)),
                "url": mid,
                "parameter": parameter,
                "period_seconds": period,
                "qualifier": qualifier,
                "unit": unit,
            }
        )

    return {
        "provider": provider,
        "label": item.get("label"),
        "station_reference": item.get("stationReference") or item.get("notation"),
        "rloi_id": item.get("RLOIid"),
        "river_name": item.get("riverName"),
        "lat": item.get("lat"),
        "long": item.get("long"),
        "status": item.get("status"),
        "station_url": item.get("@id"),
        "measures": measures,
    }


def discover_hydrology_water_level_stations(
    lat: float, lon: float, dist_km: float, limit: int
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "lat": lat,
            "long": lon,
            "dist": dist_km,
            "observedProperty": "waterLevel",
            "_limit": limit,
        }
    )
    url = f"https://environment.data.gov.uk/hydrology/id/stations.json?{params}"
    payload = http_get_json(url)
    return [compact_station_item(item, "ea_hydrology") for item in list_from_api_payload(payload)]


def discover_flood_monitoring_water_level_stations(
    lat: float, lon: float, dist_km: float, limit: int
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "lat": lat,
            "long": lon,
            "dist": dist_km,
            "parameter": "level",
            "_view": "full",
            "_limit": limit,
        }
    )
    url = f"https://environment.data.gov.uk/flood-monitoring/id/stations.json?{params}"
    payload = http_get_json(url)
    return [compact_station_item(item, "flood_monitoring") for item in list_from_api_payload(payload)]


def discover_riverlevels_alternatives(
    lat: float = 51.527,
    lon: float = -0.007,
    dist_km: float = 5.0,
    limit: int = 20,
) -> dict[str, Any]:
    hydrology: list[dict[str, Any]] = []
    flood_monitoring: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    try:
        hydrology = discover_hydrology_water_level_stations(lat, lon, dist_km, limit)
    except Exception as exc:
        errors.append({"provider": "ea_hydrology", "message": str(exc)})
    try:
        flood_monitoring = discover_flood_monitoring_water_level_stations(lat, lon, dist_km, limit)
    except Exception as exc:
        errors.append({"provider": "flood_monitoring", "message": str(exc)})

    recommended = [
        {
            "provider": "ea_hydrology",
            "name": "Lee Bridge",
            "measure_id": "f463f458-8115-42d8-834a-c6872116737d-level-i-900-m-qualified",
            "role": "nearby River Lee 15min level, good RiverLevels replacement.",
        },
        {
            "provider": "flood_monitoring",
            "name": "Lea Bridge",
            "measure_id": "5390TH-level-stage-i-15_min-mASD",
            "role": "near-real-time Check for Flooding / EA level feed.",
        },
        {
            "provider": "flood_monitoring",
            "name": "Tower Pier",
            "measure_id": "0007-level-tidal_level-i-15_min-mAOD",
            "role": "nearby Thames Tideway tidal level for diagnostics.",
        },
        {
            "provider": "flood_monitoring",
            "name": "Silvertown",
            "measure_id": "0001-level-tidal_level-i-15_min-mAOD",
            "role": "nearby Thames Tideway tidal level for diagnostics.",
        },
    ]

    return {
        "status": "partial" if errors else "ok",
        "query": {"lat": lat, "long": lon, "dist_km": dist_km, "limit": limit},
        "recommendation": (
            "Use EA Hydrology for historical 15min water levels; use Flood Monitoring "
            "for recent/latest level checks. Keep RiverLevels only as daily fallback."
        ),
        "recommended_measures": recommended,
        "ea_hydrology": hydrology,
        "flood_monitoring": flood_monitoring,
        "errors": errors,
    }


def check_riverlevels(location: str, days: int) -> dict[str, Any]:
    url = f"https://riverlevels.uk/{location}/data/json/{days}"
    data = http_get_json(url)
    return {
        "status": "ok",
        "source": "riverlevels.uk",
        "url": url,
        "note": "RiverLevels downloads are daily min/avg/max data, not 15-minute data.",
        "data": data,
    }


def search_ea_stations(search: str, observed_property: str) -> dict[str, Any]:
    params = urllib.parse.urlencode(
        {
            "search": search,
            "observedProperty": observed_property,
            "_limit": 10,
        }
    )
    url = f"https://environment.data.gov.uk/hydrology/id/stations.json?{params}"
    return {
        "status": "ok",
        "url": url,
        "data": http_get_json(url),
    }


def check_all_apis() -> dict[str, Any]:
    report: dict[str, Any] = {
        "time_utc": iso_utc(now_utc()),
        "admiralty": {"status": "not_checked"},
        "riverlevels": {
            "status": "configured" if os.getenv("RIVERLEVELS_LOCATION") else "optional_not_configured",
            "note": "RiverLevels is daily min/avg/max only; it is not the operational 15-minute source.",
        },
        "ea_rain": {
            "source": os.getenv("RAIN_SOURCE", "auto"),
            "measure_count": len(split_env_list("EA_RAIN_MEASURE_IDS")),
            "status": "not_configured",
        },
        "tide": {
            "source": os.getenv("TIDE_SOURCE", "auto"),
            "status": "not_checked",
        },
        "riverlevels_alternatives": {
            "ea_water_level_measure_count": len(split_env_list("EA_WATER_LEVEL_MEASURE_IDS")),
            "flood_monitoring_measure_count": len(split_env_list("FLOOD_MONITORING_LEVEL_MEASURE_IDS")),
            "status": "not_configured",
        },
        "primary_water_level": {
            "source": os.getenv("PRIMARY_WATER_LEVEL_SOURCE", "auto"),
            "status": "not_checked",
        },
        "mqtt": {
            "water_source_configured": bool(os.getenv("WATER_MQTT_HOST") or os.getenv("TTN_MQTT_HOST")),
            "alert_publish_configured": bool(os.getenv("ALERT_MQTT_HOST") or os.getenv("PROF_MQTT_HOST")),
        },
        "sonar": {
            "storage_mode": sonar_storage_mode(),
            "status": "not_checked",
        },
    }

    try:
        report["admiralty"] = check_admiralty_station(
            os.getenv("ADMIRALTY_STATION_NAME", "Sheerness")
        )
    except Exception as exc:
        report["admiralty"] = {"status": "error", "message": str(exc)}

    sonar_end = floor_to_interval(now_utc())
    sonar_start = sonar_end - timedelta(minutes=(SEQ_LEN - 1) * INTERVAL_MINUTES)
    try:
        sonar_frame, sonar_label = load_configured_sonar_history()
        sonar_report = evaluate_sonar_readiness(
            sonar_frame, sonar_start, sonar_end, sonar_label
        )
        sonar_report["status"] = "ready" if sonar_report["ready"] else (
            "warming_up" if sonar_report["available"] else "empty"
        )
        report["sonar"] = sonar_report
    except FileNotFoundError as exc:
        report["sonar"].update({"status": "not_collected", "message": str(exc)})
    except Exception as exc:
        report["sonar"].update({"status": "error", "message": str(exc)})

    measures = split_env_list("EA_RAIN_MEASURE_IDS")
    if measures:
        end_utc = now_utc()
        start_utc = end_utc - timedelta(hours=24)
        try:
            sample = fetch_ea_measure_readings(measures[0], start_utc, end_utc)
            report["ea_rain"] = {
                "measure_count": len(measures),
                "status": "ok" if not sample.empty else "empty",
                "sample_measure": normalize_measure_ref(measures[0]),
                "sample_rows": int(len(sample)),
            }
        except Exception as exc:
            report["ea_rain"] = {
                "measure_count": len(measures),
                "status": "error",
                "message": str(exc),
            }

    try:
        rain_sample = fetch_rain_history(now_utc() - timedelta(hours=24), now_utc())
        report["rain"] = {
            "source": os.getenv("RAIN_SOURCE", "auto"),
            "status": "ok" if not rain_sample.empty else "empty",
            "rows": int(len(rain_sample)),
            "latest_utc": iso_utc(rain_sample["date"].max()) if not rain_sample.empty else None,
        }
    except Exception as exc:
        report["rain"] = {
            "source": os.getenv("RAIN_SOURCE", "auto"),
            "status": "error",
            "message": str(exc),
        }

    try:
        end_utc = now_utc()
        start_utc = end_utc - timedelta(hours=24)
        tide_grid = pd.date_range(start=floor_to_interval(start_utc), end=floor_to_interval(end_utc), freq=f"{INTERVAL_MINUTES}min", tz="UTC")
        tide_sample = fetch_tide_history(start_utc, end_utc, tide_grid)
        report["tide"] = {
            "source": os.getenv("TIDE_SOURCE", "auto"),
            "status": "ok" if not tide_sample.empty else "empty",
            "rows": int(len(tide_sample)),
            "latest_utc": iso_utc(tide_sample["date"].max()) if not tide_sample.empty else None,
        }
    except Exception as exc:
        report["tide"] = {
            "source": os.getenv("TIDE_SOURCE", "auto"),
            "status": "error",
            "message": str(exc),
        }

    try:
        end_utc = now_utc()
        start_utc = end_utc - timedelta(hours=24)
        primary_df, primary_source = fetch_primary_api_water_history(start_utc, end_utc)
        report["primary_water_level"] = {
            "source": primary_source,
            "status": "ok" if not primary_df.empty else "empty",
            "rows": int(len(primary_df)),
            "latest_utc": iso_utc(primary_df["date"].max()) if not primary_df.empty else None,
            "latest_value_m": round(float(primary_df[TARGET_COLUMN].iloc[-1]), 4) if not primary_df.empty else None,
        }
    except Exception as exc:
        report["primary_water_level"] = {
            "source": os.getenv("PRIMARY_WATER_LEVEL_SOURCE", "auto"),
            "status": "error",
            "message": str(exc),
        }

    if os.getenv("RIVERLEVELS_LOCATION"):
        try:
            probe = check_riverlevels(os.getenv("RIVERLEVELS_LOCATION", ""), 10)
            levels = probe.get("data", {}).get("levels", [])
            report["riverlevels"].update(
                {
                    "status": "ok",
                    "sample_rows": len(levels) if isinstance(levels, list) else 0,
                    "url": probe.get("url"),
                }
            )
        except Exception as exc:
            report["riverlevels"].update({"status": "error", "message": str(exc)})

    level_measures = split_env_list("EA_WATER_LEVEL_MEASURE_IDS") + split_env_list("FLOOD_MONITORING_LEVEL_MEASURE_IDS")
    if level_measures:
        try:
            report["riverlevels_alternatives"] = fetch_external_level_history(
                now_utc() - timedelta(hours=24), now_utc()
            )
        except Exception as exc:
            report["riverlevels_alternatives"] = {
                "status": "error",
                "message": str(exc),
                "ea_water_level_measure_count": len(split_env_list("EA_WATER_LEVEL_MEASURE_IDS")),
                "flood_monitoring_measure_count": len(split_env_list("FLOOD_MONITORING_LEVEL_MEASURE_IDS")),
            }

    return report


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "HouseMillFloodServer/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{stamp}] {self.address_string()} {fmt % args}")

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(json_safe(payload), indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length).decode("utf-8")
        return json.loads(data)

    def parsed(self) -> tuple[str, dict[str, list[str]]]:
        parsed_url = urllib.parse.urlparse(self.path)
        return parsed_url.path, urllib.parse.parse_qs(parsed_url.query)

    def do_GET(self) -> None:
        path, query = self.parsed()
        try:
            if path == "/":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "service": "House Mill flood prediction server",
                        "endpoints": [
                            "/health",
                            "/status",
                            "/predict",
                            "/api/check/all",
                            "/api/discover/riverlevels-alternatives?lat=51.527&long=-0.007&dist=5",
                            "/api/check/admiralty?station=Sheerness",
                            "/api/check/riverlevels?location=river-avon-evesham&days=10",
                            "/api/check/ea?search=Deptford&observedProperty=rainfall",
                        ],
                    },
                )
            elif path == "/health":
                service = get_runtime_service()
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "time_utc": iso_utc(now_utc()),
                        "predictor_available": service.predictor.available,
                        "predictor_error": service.predictor.error,
                    },
                )
            elif path == "/status":
                self.send_json(HTTPStatus.OK, get_runtime_service().status())
            elif path == "/predict":
                self.send_json(HTTPStatus.OK, get_runtime_service().run_prediction())
            elif path == "/api/check/admiralty":
                station = query.get("station", ["Sheerness"])[0]
                self.send_json(HTTPStatus.OK, check_admiralty_station(station))
            elif path == "/api/check/all":
                self.send_json(HTTPStatus.OK, check_all_apis())
            elif path == "/api/discover/riverlevels-alternatives":
                lat = float(query.get("lat", ["51.527"])[0])
                lon = float(query.get("long", query.get("lon", ["-0.007"]))[0])
                dist = float(query.get("dist", ["5"])[0])
                limit = int(query.get("limit", ["20"])[0])
                self.send_json(
                    HTTPStatus.OK,
                    discover_riverlevels_alternatives(lat=lat, lon=lon, dist_km=dist, limit=limit),
                )
            elif path == "/api/check/riverlevels":
                location = query.get("location", [os.getenv("RIVERLEVELS_LOCATION", "")])[0]
                days = int(query.get("days", ["10"])[0])
                if not location:
                    self.send_json(
                        HTTPStatus.BAD_REQUEST,
                        {"status": "error", "message": "location query parameter is required"},
                    )
                else:
                    self.send_json(HTTPStatus.OK, check_riverlevels(location, days))
            elif path == "/api/check/ea":
                search = query.get("search", ["Deptford"])[0]
                observed = query.get("observedProperty", ["rainfall"])[0]
                self.send_json(HTTPStatus.OK, search_ea_stations(search, observed))
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"status": "not_found", "path": path})
        except urllib.error.HTTPError as exc:
            self.send_json(
                exc.code,
                {
                    "status": "upstream_http_error",
                    "code": exc.code,
                    "reason": exc.reason,
                    "body": exc.read().decode("utf-8", errors="replace")[:1000],
                },
            )
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "error", "message": str(exc)})

    def do_POST(self) -> None:
        path, _query = self.parsed()
        try:
            body = self.read_json()
            if path == "/predict":
                self.send_json(HTTPStatus.OK, get_runtime_service().run_prediction())
            elif path == "/countdown/update":
                payload = get_runtime_service().update_countdown()
                self.send_json(
                    HTTPStatus.OK,
                    payload or {"status": "empty", "message": "No prediction payload yet."},
                )
            elif path == "/ingest/sonar":
                distance_mm = float(body["distance_mm"])
                sensor_height_m = float(os.getenv("SONAR_SENSOR_HEIGHT_M", "5.0"))
                water_level_m = max(0.0, sensor_height_m - distance_mm / 1000.0)
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "accepted",
                        "timestamp": body.get("timestamp") or iso_utc(now_utc()),
                        "distance_mm": distance_mm,
                        "internal_water_m": round(water_level_m, 4),
                        "note": "This endpoint validates conversion only. Production storage should write to the feature table.",
                    },
                )
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"status": "not_found", "path": path})
        except Exception as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"status": "error", "message": str(exc)})


def run_once() -> None:
    try:
        payload = get_runtime_service().run_prediction()
        print(json.dumps(json_safe(payload), indent=2))
    except Exception as exc:
        result = {
            "status": "not_ready",
            "message": str(exc),
            "time_utc": iso_utc(now_utc()),
        }
        print(json.dumps(result, indent=2))
        raise SystemExit(2)


def run_server() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    service = get_runtime_service()
    service.start_background()
    print(f"House Mill flood server listening on http://{host}:{port}")
    print(f"Predictor available: {service.predictor.available}")
    if service.predictor.error:
        print(f"Predictor error: {service.predictor.error}")
    httpd = ThreadingHTTPServer((host, port), RequestHandler)
    httpd.serve_forever()


def run_worker() -> None:
    print("House Mill flood worker started. Prediction is rechecked every 30 minutes.")
    get_runtime_service().background_loop()


def main() -> None:
    parser = argparse.ArgumentParser(description="House Mill cloud flood prediction runner")
    parser.add_argument(
        "--mode",
        choices=[
            "once",
            "server",
            "worker",
            "collect-water-mqtt",
            "check-apis",
            "check-river-alternatives",
        ],
        default=os.getenv("RUN_MODE", "once"),
        help="once is the local-safe default. Use server/worker on DigitalOcean.",
    )
    args = parser.parse_args()

    if args.mode == "once":
        run_once()
    elif args.mode == "server":
        run_server()
    elif args.mode == "worker":
        run_worker()
    elif args.mode == "collect-water-mqtt":
        collect_water_mqtt_forever()
    elif args.mode == "check-apis":
        print(json.dumps(json_safe(check_all_apis()), indent=2))
    elif args.mode == "check-river-alternatives":
        print(json.dumps(json_safe(discover_riverlevels_alternatives()), indent=2))


if __name__ == "__main__":
    main()
