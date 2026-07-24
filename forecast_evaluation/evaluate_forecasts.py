from __future__ import annotations

import argparse
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = APP_ROOT.parent
DEFAULT_PREDICTION_DB = (
    DEFAULT_PROJECT_ROOT / "cloud_flood_server" / "data" / "prediction_history.sqlite3"
)
DEFAULT_SONAR_DB = (
    DEFAULT_PROJECT_ROOT / "cloud_flood_server" / "data" / "sonar_history.sqlite3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare archived House Mill forecasts with later sonar observations."
    )
    parser.add_argument(
        "--prediction-db",
        type=Path,
        default=DEFAULT_PREDICTION_DB,
        help="SQLite database containing forecast_runs and forecast_points.",
    )
    parser.add_argument(
        "--sonar-db",
        type=Path,
        default=DEFAULT_SONAR_DB,
        help="SQLite database containing sonar_readings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "output",
        help="Directory for CSV, JSON, and PNG reports.",
    )
    parser.add_argument(
        "--match-tolerance-minutes",
        type=float,
        default=8.0,
        help="Maximum time difference in nearest mode.",
    )
    parser.add_argument(
        "--actual-resample-method",
        choices=["mean", "last", "first", "median", "max", "min", "nearest"],
        default="mean",
        help="How 10-minute sonar readings are aligned to the model's 15-minute targets.",
    )
    parser.add_argument(
        "--minimum-run-coverage",
        type=float,
        default=0.90,
        help="Minimum matched-point ratio needed to score one complete 24-hour run.",
    )
    parser.add_argument("--watch-level-m", type=float, default=4.20)
    parser.add_argument("--warning-level-m", type=float, default=4.43)
    parser.add_argument("--severe-level-m", type=float, default=4.70)
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def read_forecasts(path: Path) -> pd.DataFrame:
    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        required = {"forecast_runs", "forecast_points"}
        if not required.issubset(tables):
            missing = ", ".join(sorted(required - tables))
            raise RuntimeError(
                f"Prediction database is missing {missing}. Deploy the forecast archive update first."
            )
        frame = pd.read_sql_query(
            """
            SELECT
                p.run_id,
                r.forecast_generated_utc,
                r.history_last_utc,
                r.site,
                r.source,
                r.risk_level AS predicted_risk_level,
                r.risk_label AS predicted_risk_label,
                r.max_predicted_m,
                r.eta_minutes AS predicted_eta_minutes,
                r.data_quality_json,
                p.target_utc,
                p.lead_minutes,
                p.predicted_water_m,
                p.flood_probability
            FROM forecast_points AS p
            JOIN forecast_runs AS r ON r.run_id = p.run_id
            ORDER BY r.forecast_generated_utc, p.target_utc
            """,
            connection,
        )
    finally:
        connection.close()

    for column in ("forecast_generated_utc", "history_last_utc", "target_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    frame["predicted_water_m"] = pd.to_numeric(
        frame["predicted_water_m"], errors="coerce"
    )
    frame["lead_minutes"] = pd.to_numeric(frame["lead_minutes"], errors="coerce")
    quality = frame["data_quality_json"].map(parse_data_quality)
    frame["water_source"] = quality.map(lambda item: item.get("water_source"))
    frame["water_selection_mode"] = quality.map(
        lambda item: item.get("water_selection_mode")
    )
    return frame.dropna(
        subset=["run_id", "forecast_generated_utc", "target_utc", "predicted_water_m"]
    )


def parse_data_quality(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_sonar(path: Path) -> pd.DataFrame:
    connection = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if "sonar_readings" not in tables:
            raise RuntimeError("Sonar database is missing the sonar_readings table.")
        frame = pd.read_sql_query(
            """
            SELECT date_utc AS actual_utc, internal_water_m AS actual_water_m
            FROM sonar_readings
            ORDER BY date_utc, id
            """,
            connection,
        )
    finally:
        connection.close()

    frame["actual_utc"] = pd.to_datetime(frame["actual_utc"], utc=True, errors="coerce")
    frame["actual_water_m"] = pd.to_numeric(frame["actual_water_m"], errors="coerce")
    frame = frame.dropna(subset=["actual_utc", "actual_water_m"])
    return (
        frame.groupby("actual_utc", as_index=False)["actual_water_m"]
        .mean()
        .sort_values("actual_utc")
    )


def match_observations(
    forecasts: pd.DataFrame,
    sonar: pd.DataFrame,
    tolerance_minutes: float,
    method: str = "mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_actual = sonar["actual_utc"].max()
    if method == "nearest":
        complete_through = latest_actual
    else:
        complete_through = latest_actual.floor("15min") - pd.Timedelta(minutes=15)
    matured = forecasts[forecasts["target_utc"] <= complete_through].copy()
    pending = forecasts[forecasts["target_utc"] > complete_through].copy()
    if matured.empty:
        matured["actual_utc"] = pd.NaT
        matured["actual_water_m"] = np.nan
        return matured, pending

    if method == "nearest":
        actual = sonar.sort_values("actual_utc").copy()
        actual["actual_samples_in_bin"] = 1
        matched = pd.merge_asof(
            matured.sort_values("target_utc"),
            actual,
            left_on="target_utc",
            right_on="actual_utc",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )
    else:
        series = sonar.set_index("actual_utc")["actual_water_m"].sort_index()
        resampler = series.resample("15min")
        aggregate = getattr(resampler, method)()
        actual = pd.DataFrame(
            {
                "actual_utc": aggregate.index,
                "actual_water_m": aggregate.to_numpy(),
                "actual_samples_in_bin": resampler.count().to_numpy(),
            }
        )
        matched = matured.merge(
            actual, left_on="target_utc", right_on="actual_utc", how="left"
        )
    matched["match_delta_minutes"] = (
        matched["actual_utc"] - matched["target_utc"]
    ).dt.total_seconds().abs() / 60.0
    matched["error_m"] = matched["predicted_water_m"] - matched["actual_water_m"]
    matched["absolute_error_m"] = matched["error_m"].abs()
    matched["squared_error_m2"] = matched["error_m"] ** 2
    return matched, pending


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def point_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    valid = frame.dropna(subset=["predicted_water_m", "actual_water_m"])
    total = int(len(frame))
    matched = int(len(valid))
    result: dict[str, Any] = {
        "matured_points": total,
        "matched_points": matched,
        "coverage": round(matched / total, 6) if total else None,
        "bias_m": None,
        "mae_m": None,
        "mean_absolute_error_m": None,
        "median_absolute_error_m": None,
        "maximum_absolute_error_m": None,
        "median_signed_error_m": None,
        "rmse_m": None,
        "pearson_r": None,
        "r_squared": None,
    }
    if not matched:
        return result

    errors = valid["predicted_water_m"] - valid["actual_water_m"]
    absolute_errors = errors.abs()
    result["bias_m"] = round(float(errors.mean()), 6)
    result["mae_m"] = round(float(absolute_errors.mean()), 6)
    result["mean_absolute_error_m"] = result["mae_m"]
    result["median_absolute_error_m"] = round(float(absolute_errors.median()), 6)
    result["maximum_absolute_error_m"] = round(float(absolute_errors.max()), 6)
    result["median_signed_error_m"] = round(float(errors.median()), 6)
    result["rmse_m"] = round(float(np.sqrt(np.mean(errors**2))), 6)
    if matched >= 2:
        correlation = valid["predicted_water_m"].corr(valid["actual_water_m"])
        result["pearson_r"] = (
            round(float(correlation), 6) if pd.notna(correlation) else None
        )
        denominator = float(
            ((valid["actual_water_m"] - valid["actual_water_m"].mean()) ** 2).sum()
        )
        if denominator > 0:
            r_squared = 1.0 - float((errors**2).sum()) / denominator
            result["r_squared"] = round(r_squared, 6)
    return result


def lead_time_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    if "lead_minutes" not in frame.columns:
        values = pd.Series(dtype=float)
    else:
        values = pd.to_numeric(frame["lead_minutes"], errors="coerce").dropna()
    if values.empty:
        return {
            "points": 0,
            "minimum_minutes": None,
            "maximum_minutes": None,
            "mean_minutes": None,
            "median_minutes": None,
            "minimum_hours": None,
            "maximum_hours": None,
            "mean_hours": None,
            "median_hours": None,
        }

    minimum = float(values.min())
    maximum = float(values.max())
    mean = float(values.mean())
    median = float(values.median())
    return {
        "points": int(len(values)),
        "minimum_minutes": round(minimum, 3),
        "maximum_minutes": round(maximum, 3),
        "mean_minutes": round(mean, 3),
        "median_minutes": round(median, 3),
        "minimum_hours": round(minimum / 60.0, 4),
        "maximum_hours": round(maximum / 60.0, 4),
        "mean_hours": round(mean / 60.0, 4),
        "median_hours": round(median / 60.0, 4),
    }


def maximum_error_point(frame: pd.DataFrame) -> dict[str, Any] | None:
    valid = frame.dropna(
        subset=["predicted_water_m", "actual_water_m", "absolute_error_m"]
    )
    if valid.empty:
        return None
    row = valid.loc[valid["absolute_error_m"].idxmax()]
    return {
        "run_id": row.get("run_id"),
        "forecast_generated_utc": row.get("forecast_generated_utc"),
        "target_utc": row.get("target_utc"),
        "lead_minutes": finite_float(row.get("lead_minutes")),
        "predicted_water_m": finite_float(row.get("predicted_water_m")),
        "actual_water_m": finite_float(row.get("actual_water_m")),
        "signed_error_m": finite_float(row.get("error_m")),
        "absolute_error_m": finite_float(row.get("absolute_error_m")),
    }


def lead_bucket(value: float) -> str:
    if value <= 60:
        return "00-01h"
    if value <= 180:
        return "01-03h"
    if value <= 360:
        return "03-06h"
    if value <= 720:
        return "06-12h"
    return "12-24h"


def build_lead_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    valid = matched.dropna(subset=["actual_water_m"]).copy()
    columns = [
        "lead_bucket",
        "matched_points",
        "bias_m",
        "mae_m",
        "median_absolute_error_m",
        "maximum_absolute_error_m",
        "rmse_m",
        "pearson_r",
        "r_squared",
    ]
    if valid.empty:
        return pd.DataFrame(columns=columns)
    valid["lead_bucket"] = valid["lead_minutes"].map(lead_bucket)
    rows = []
    for bucket, group in valid.groupby("lead_bucket", sort=True):
        metrics = point_metrics(group)
        rows.append({"lead_bucket": bucket, **{k: metrics[k] for k in columns[1:]}})
    return pd.DataFrame(rows, columns=columns)


def build_source_metrics(matched: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "water_selection_mode",
        "water_source",
        "forecast_runs",
        "matched_points",
        "bias_m",
        "mae_m",
        "median_absolute_error_m",
        "maximum_absolute_error_m",
        "rmse_m",
        "pearson_r",
        "r_squared",
    ]
    valid = matched.dropna(subset=["actual_water_m"]).copy()
    if valid.empty:
        return pd.DataFrame(columns=columns)
    valid["water_selection_mode"] = valid["water_selection_mode"].fillna("unknown")
    valid["water_source"] = valid["water_source"].fillna("unknown")
    rows = []
    for (mode, source), group in valid.groupby(
        ["water_selection_mode", "water_source"], sort=True
    ):
        metrics = point_metrics(group)
        rows.append(
            {
                "water_selection_mode": mode,
                "water_source": source,
                "forecast_runs": int(group["run_id"].nunique()),
                **{key: metrics[key] for key in columns[3:]},
            }
        )
    return pd.DataFrame(rows, columns=columns)


def max_consecutive(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def actual_risk(
    group: pd.DataFrame, watch: float, warning: float, severe: float
) -> dict[str, Any]:
    ordered = group.dropna(subset=["actual_water_m"]).sort_values("target_utc")
    values = ordered["actual_water_m"].to_numpy(dtype=float)
    leads = ordered["lead_minutes"].to_numpy(dtype=float)
    max_level = float(np.max(values))
    warning_mask = values >= warning
    warning_indices = np.where(warning_mask)[0]
    eta = int(round(leads[warning_indices[0]])) if len(warning_indices) else None
    duration = max_consecutive(warning_mask) * 15
    if max_level >= severe or duration >= 120 or (eta is not None and eta <= 120):
        level = 3
    elif max_level >= warning or duration >= 60:
        level = 2
    elif max_level >= watch:
        level = 1
    else:
        level = 0
    return {
        "actual_risk_level": level,
        "actual_max_m": max_level,
        "actual_eta_minutes": eta,
        "actual_duration_above_warning_minutes": duration,
    }


def build_run_metrics(
    forecasts: pd.DataFrame,
    matched: pd.DataFrame,
    latest_actual: pd.Timestamp,
    minimum_coverage: float,
    watch: float,
    warning: float,
    severe: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, all_points in forecasts.groupby("run_id", sort=False):
        target_end = all_points["target_utc"].max()
        if target_end > latest_actual:
            continue
        scored = matched[matched["run_id"] == run_id]
        expected = len(all_points)
        available = int(scored["actual_water_m"].notna().sum())
        coverage = available / expected if expected else 0.0
        if available == 0 or coverage < minimum_coverage:
            continue
        valid = scored.dropna(subset=["actual_water_m"]).sort_values("target_utc")
        metrics = point_metrics(valid)
        observed = actual_risk(valid, watch, warning, severe)
        first = all_points.iloc[0]
        predicted_risk = int(first["predicted_risk_level"])
        actual_level = int(observed["actual_risk_level"])
        predicted_eta = finite_float(first["predicted_eta_minutes"])
        actual_eta = finite_float(observed["actual_eta_minutes"])
        rows.append(
            {
                "run_id": run_id,
                "forecast_generated_utc": first["forecast_generated_utc"],
                "target_start_utc": all_points["target_utc"].min(),
                "target_end_utc": target_end,
                "water_selection_mode": first.get("water_selection_mode"),
                "water_source": first.get("water_source"),
                "expected_points": expected,
                "matched_points": available,
                "coverage": round(coverage, 6),
                "predicted_risk_level": predicted_risk,
                "actual_risk_level": actual_level,
                "risk_exact_match": predicted_risk == actual_level,
                "risk_within_one_level": abs(predicted_risk - actual_level) <= 1,
                "predicted_max_m": finite_float(first["max_predicted_m"]),
                "actual_max_m": round(float(observed["actual_max_m"]), 6),
                "peak_error_m": round(
                    float(first["max_predicted_m"]) - float(observed["actual_max_m"]),
                    6,
                ),
                "predicted_eta_minutes": predicted_eta,
                "actual_eta_minutes": actual_eta,
                "eta_error_minutes": (
                    round(predicted_eta - actual_eta, 3)
                    if predicted_eta is not None and actual_eta is not None
                    else None
                ),
                "mae_m": metrics["mae_m"],
                "rmse_m": metrics["rmse_m"],
            }
        )
    return pd.DataFrame(rows)


def risk_summary(run_metrics: pd.DataFrame) -> dict[str, Any]:
    if run_metrics.empty:
        return {
            "completed_scored_runs": 0,
            "exact_accuracy": None,
            "within_one_level_accuracy": None,
            "warning_false_alarms": 0,
            "warning_misses": 0,
        }
    predicted_warning = run_metrics["predicted_risk_level"] >= 2
    actual_warning = run_metrics["actual_risk_level"] >= 2
    return {
        "completed_scored_runs": int(len(run_metrics)),
        "exact_accuracy": round(float(run_metrics["risk_exact_match"].mean()), 6),
        "within_one_level_accuracy": round(
            float(run_metrics["risk_within_one_level"].mean()), 6
        ),
        "warning_false_alarms": int((predicted_warning & ~actual_warning).sum()),
        "warning_misses": int((~predicted_warning & actual_warning).sum()),
    }


def confusion_matrix(run_metrics: pd.DataFrame) -> pd.DataFrame:
    labels = [0, 1, 2, 3]
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    matrix.index.name = "actual_risk_level"
    matrix.columns.name = "predicted_risk_level"
    for row in run_metrics.itertuples():
        matrix.loc[int(row.actual_risk_level), int(row.predicted_risk_level)] += 1
    return matrix


def write_charts(
    output_dir: Path,
    matched: pd.DataFrame,
    lead_metrics: pd.DataFrame,
    matrix: pd.DataFrame,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed; CSV and JSON reports were still created."

    valid = matched.dropna(subset=["actual_water_m"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(
            valid["actual_water_m"],
            valid["predicted_water_m"],
            s=12,
            alpha=0.35,
        )
        lower = min(valid["actual_water_m"].min(), valid["predicted_water_m"].min())
        upper = max(valid["actual_water_m"].max(), valid["predicted_water_m"].max())
        ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
        ax.set_xlabel("Actual sonar water level (m)")
        ax.set_ylabel("Predicted water level (m)")
        ax.set_title("Forecast vs actual")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "predicted_vs_actual.png", dpi=180)
        plt.close(fig)

    if not lead_metrics.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.bar(lead_metrics["lead_bucket"], lead_metrics["mae_m"], label="MAE")
        ax.plot(
            lead_metrics["lead_bucket"],
            lead_metrics["rmse_m"],
            color="crimson",
            marker="o",
            label="RMSE",
        )
        ax.set_xlabel("Forecast lead time")
        ax.set_ylabel("Error (m)")
        ax.set_title("Error by lead time")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "error_by_lead_time.png", dpi=180)
        plt.close(fig)

    if int(matrix.to_numpy().sum()) > 0:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        for row in range(4):
            for column in range(4):
                ax.text(column, row, str(matrix.iloc[row, column]), ha="center", va="center")
        ax.set_xticks(range(4), ["0", "1", "2", "3"])
        ax.set_yticks(range(4), ["0", "1", "2", "3"])
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Actual risk")
        ax.set_title("Risk confusion matrix")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / "risk_confusion_matrix.png", dpi=180)
        plt.close(fig)
    return None


def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    args = parse_args()
    prediction_db = require_file(args.prediction_db, "Prediction database")
    sonar_db = require_file(args.sonar_db, "Sonar database")
    if not 0 < args.minimum_run_coverage <= 1:
        raise ValueError("--minimum-run-coverage must be in (0, 1].")
    if args.match_tolerance_minutes <= 0:
        raise ValueError("--match-tolerance-minutes must be positive.")

    forecasts = read_forecasts(prediction_db)
    sonar = read_sonar(sonar_db)
    if forecasts.empty:
        raise RuntimeError("Prediction archive is empty; wait for the worker to run.")
    if sonar.empty:
        raise RuntimeError("Sonar database is empty; no actual values can be scored.")

    matched, pending = match_observations(
        forecasts,
        sonar,
        args.match_tolerance_minutes,
        args.actual_resample_method,
    )
    overall = point_metrics(matched)
    evaluated_points = matched.dropna(
        subset=["predicted_water_m", "actual_water_m"]
    )
    evaluated_lead_times = lead_time_statistics(evaluated_points)
    archived_lead_times = lead_time_statistics(forecasts)
    lead_metrics = build_lead_metrics(matched)
    source_metrics = build_source_metrics(matched)
    latest_actual = sonar["actual_utc"].max()
    actual_complete_through = (
        latest_actual
        if args.actual_resample_method == "nearest"
        else latest_actual.floor("15min") - pd.Timedelta(minutes=15)
    )
    run_metrics = build_run_metrics(
        forecasts,
        matched,
        actual_complete_through,
        args.minimum_run_coverage,
        args.watch_level_m,
        args.warning_level_m,
        args.severe_level_m,
    )
    matrix = confusion_matrix(run_metrics)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output_dir / "matched_forecast_points.csv", index=False)
    pending.to_csv(output_dir / "pending_forecast_points.csv", index=False)
    lead_metrics.to_csv(output_dir / "lead_time_metrics.csv", index=False)
    source_metrics.to_csv(output_dir / "water_source_metrics.csv", index=False)
    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    matrix.to_csv(output_dir / "risk_confusion_matrix.csv")
    chart_warning = write_charts(output_dir, matched, lead_metrics, matrix)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction_database": str(prediction_db),
        "sonar_database": str(sonar_db),
        "forecast_runs": int(forecasts["run_id"].nunique()),
        "forecast_points": int(len(forecasts)),
        "sonar_points": int(len(sonar)),
        "latest_actual_utc": latest_actual,
        "actual_complete_through_utc": actual_complete_through,
        "pending_future_points": int(len(pending)),
        "actual_resample_method": args.actual_resample_method,
        "match_tolerance_minutes": args.match_tolerance_minutes,
        "minimum_run_coverage": args.minimum_run_coverage,
        "thresholds_m": {
            "watch": args.watch_level_m,
            "warning": args.warning_level_m,
            "severe": args.severe_level_m,
        },
        "point_metrics": overall,
        "maximum_error_point": maximum_error_point(matched),
        "evaluated_lead_time_statistics": evaluated_lead_times,
        "archived_lead_time_statistics": archived_lead_times,
        "water_source_metrics": source_metrics.to_dict(orient="records"),
        "risk_metrics": risk_summary(run_metrics),
        "chart_warning": chart_warning,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, default=json_default))
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
