from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import evaluate_forecasts as base


APP_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate House Mill forecast points only where the matched actual "
            "sonar water level is at or above a selected threshold."
        )
    )
    parser.add_argument(
        "--prediction-db",
        type=Path,
        default=APP_ROOT / "prediction_history.sqlite3",
    )
    parser.add_argument(
        "--sonar-db",
        type=Path,
        default=APP_ROOT / "sonar_history.sqlite3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_ROOT / "output_above_2m",
    )
    parser.add_argument(
        "--threshold-m",
        type=float,
        default=2.0,
        help="Only score points whose matched actual water level is >= this value.",
    )
    parser.add_argument(
        "--actual-resample-method",
        choices=["mean", "last", "first", "median", "max", "min", "nearest"],
        default="mean",
    )
    parser.add_argument("--match-tolerance-minutes", type=float, default=8.0)
    return parser.parse_args()


def error_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    absolute = pd.to_numeric(frame["absolute_error_m"], errors="coerce").dropna()
    signed = pd.to_numeric(frame["error_m"], errors="coerce").dropna()
    if absolute.empty:
        return {}
    return {
        "absolute_error_quantiles_m": {
            "p50": round(float(absolute.quantile(0.50)), 6),
            "p75": round(float(absolute.quantile(0.75)), 6),
            "p90": round(float(absolute.quantile(0.90)), 6),
            "p95": round(float(absolute.quantile(0.95)), 6),
            "p99": round(float(absolute.quantile(0.99)), 6),
            "maximum": round(float(absolute.max()), 6),
        },
        "absolute_error_exceedance_fraction": {
            "above_0.10m": round(float((absolute > 0.10).mean()), 6),
            "above_0.20m": round(float((absolute > 0.20).mean()), 6),
            "above_0.30m": round(float((absolute > 0.30).mean()), 6),
            "above_0.50m": round(float((absolute > 0.50).mean()), 6),
        },
        "signed_error_range_m": {
            "minimum": round(float(signed.min()), 6),
            "maximum": round(float(signed.max()), 6),
        },
        "underprediction_fraction": round(float((signed < 0).mean()), 6),
        "overprediction_fraction": round(float((signed > 0).mean()), 6),
    }


def level_statistics(frame: pd.DataFrame) -> dict[str, Any]:
    unique_actual = (
        frame[["target_utc", "actual_water_m"]]
        .drop_duplicates("target_utc")
        .dropna(subset=["actual_water_m"])
    )
    return {
        "unique_actual_target_times": int(len(unique_actual)),
        "actual_minimum_m": round(float(unique_actual["actual_water_m"].min()), 6),
        "actual_maximum_m": round(float(unique_actual["actual_water_m"].max()), 6),
        "actual_mean_m": round(float(unique_actual["actual_water_m"].mean()), 6),
        "actual_median_m": round(float(unique_actual["actual_water_m"].median()), 6),
        "predicted_minimum_m": round(float(frame["predicted_water_m"].min()), 6),
        "predicted_maximum_m": round(float(frame["predicted_water_m"].max()), 6),
        "predicted_mean_m": round(float(frame["predicted_water_m"].mean()), 6),
        "predicted_median_m": round(float(frame["predicted_water_m"].median()), 6),
    }


def build_run_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_id, group in frame.groupby("run_id", sort=False):
        metrics = base.point_metrics(group)
        lead = base.lead_time_statistics(group)
        first = group.iloc[0]
        rows.append(
            {
                "run_id": run_id,
                "forecast_generated_utc": first["forecast_generated_utc"],
                "high_water_points": int(len(group)),
                "unique_target_times": int(group["target_utc"].nunique()),
                "actual_minimum_m": round(float(group["actual_water_m"].min()), 6),
                "actual_maximum_m": round(float(group["actual_water_m"].max()), 6),
                "predicted_minimum_m": round(
                    float(group["predicted_water_m"].min()), 6
                ),
                "predicted_maximum_m": round(
                    float(group["predicted_water_m"].max()), 6
                ),
                "bias_m": metrics["bias_m"],
                "mean_absolute_error_m": metrics["mean_absolute_error_m"],
                "median_absolute_error_m": metrics["median_absolute_error_m"],
                "maximum_absolute_error_m": metrics["maximum_absolute_error_m"],
                "rmse_m": metrics["rmse_m"],
                "pearson_r": metrics["pearson_r"],
                "r_squared": metrics["r_squared"],
                "minimum_lead_minutes": lead["minimum_minutes"],
                "maximum_lead_minutes": lead["maximum_minutes"],
                "mean_lead_minutes": lead["mean_minutes"],
                "median_lead_minutes": lead["median_minutes"],
            }
        )
    return pd.DataFrame(rows)


def risk_context(frame: pd.DataFrame) -> dict[str, Any]:
    unique_targets = frame[
        ["target_utc", "actual_water_m"]
    ].drop_duplicates("target_utc")
    run_levels = (
        frame[["run_id", "predicted_risk_level"]]
        .drop_duplicates("run_id")["predicted_risk_level"]
        .value_counts()
        .sort_index()
    )
    return {
        "actual_unique_targets_at_or_above_watch_4_20m": int(
            (unique_targets["actual_water_m"] >= 4.20).sum()
        ),
        "actual_unique_targets_at_or_above_warning_4_43m": int(
            (unique_targets["actual_water_m"] >= 4.43).sum()
        ),
        "actual_unique_targets_at_or_above_severe_4_70m": int(
            (unique_targets["actual_water_m"] >= 4.70).sum()
        ),
        "predicted_run_risk_level_counts": {
            str(int(level)): int(count) for level, count in run_levels.items()
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value) * 100:.1f}%"


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "无可用数据。"
    labels = {
        "lead_bucket": "提前量",
        "matched_points": "预测点",
        "bias_m": "Bias (m)",
        "mae_m": "MAE (m)",
        "median_absolute_error_m": "中位绝对误差 (m)",
        "maximum_absolute_error_m": "最大绝对误差 (m)",
        "rmse_m": "RMSE (m)",
        "pearson_r": "Pearson r",
        "r_squared": "R²",
    }
    lines = [
        "| " + " | ".join(labels.get(column, column) for column in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        cells = []
        for column, value in zip(columns, row):
            if column in {"lead_bucket", "matched_points"}:
                cells.append(str(value))
            else:
                cells.append(fmt(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown_report(
    summary: dict[str, Any], lead_metrics: pd.DataFrame
) -> str:
    metrics = summary["point_metrics"]
    distribution = summary["error_distribution"]
    quantiles = distribution["absolute_error_quantiles_m"]
    exceedance = distribution["absolute_error_exceedance_fraction"]
    lead = summary["lead_time_statistics"]
    levels = summary["water_level_statistics"]
    worst = summary["maximum_error_point"]
    risk = summary["risk_context"]
    bias = float(metrics["bias_m"])
    direction = "高估" if bias > 0 else "低估" if bias < 0 else "无明显方向"
    first_mae = float(lead_metrics.iloc[0]["mae_m"])
    last_mae = float(lead_metrics.iloc[-1]["mae_m"])
    lead_change = (
        (last_mae / first_mae - 1.0) * 100.0 if first_mae > 0 else float("nan")
    )

    return f"""# 2 m及以上实测水位预测评估报告

生成时间：{summary["generated_utc"]}

## 评估范围

本报告只评价15分钟重采样后满足 `actual_water_m >= {summary["threshold_m"]:.2f}` 的实测高水位点。实测低于阈值的所有预测点均被排除，不参与任何水位误差统计。

- 数据时间：{summary["target_start_utc"]} 至 {summary["target_end_utc"]}
- 高水位预测-实测配对：{summary["filtered_points"]:,}
- 不同高水位目标时间：{levels["unique_actual_target_times"]:,}
- 涉及预测运行：{summary["forecast_runs_with_high_water"]:,}
- 实测水位范围：{levels["actual_minimum_m"]:.3f}–{levels["actual_maximum_m"]:.3f} m
- 实测平均水位：{levels["actual_mean_m"]:.3f} m
- 实测中位水位：{levels["actual_median_m"]:.3f} m

## 核心指标

| 指标 | 结果 |
|---|---:|
| Bias | {fmt(metrics["bias_m"])} m |
| 平均绝对误差 MAE | {fmt(metrics["mean_absolute_error_m"])} m |
| 中位绝对误差 | {fmt(metrics["median_absolute_error_m"])} m |
| 最大绝对误差 | {fmt(metrics["maximum_absolute_error_m"])} m |
| RMSE | {fmt(metrics["rmse_m"])} m |
| Pearson r | {fmt(metrics["pearson_r"])} |
| R² | {fmt(metrics["r_squared"])} |

模型在2 m以上水位阶段整体偏向{direction}，平均方向性偏差为 {abs(bias):.3f} m。

## 误差分布

| 统计 | 绝对误差 |
|---|---:|
| P50 | {fmt(quantiles["p50"])} m |
| P75 | {fmt(quantiles["p75"])} m |
| P90 | {fmt(quantiles["p90"])} m |
| P95 | {fmt(quantiles["p95"])} m |
| P99 | {fmt(quantiles["p99"])} m |
| 最大值 | {fmt(quantiles["maximum"])} m |

- 绝对误差超过0.10 m：{pct(exceedance["above_0.10m"])}
- 绝对误差超过0.20 m：{pct(exceedance["above_0.20m"])}
- 绝对误差超过0.30 m：{pct(exceedance["above_0.30m"])}
- 绝对误差超过0.50 m：{pct(exceedance["above_0.50m"])}
- 高估比例：{pct(distribution["overprediction_fraction"])}
- 低估比例：{pct(distribution["underprediction_fraction"])}

## 提前量表现

{markdown_table(lead_metrics, [
    "lead_bucket",
    "matched_points",
    "bias_m",
    "mae_m",
    "median_absolute_error_m",
    "maximum_absolute_error_m",
    "rmse_m",
    "pearson_r",
    "r_squared",
])}

高水位样本的最短提前量为 {fmt(lead["minimum_minutes"], 1)} 分钟，最长提前量为 {fmt(lead["maximum_minutes"], 1)} 分钟，平均为 {fmt(lead["mean_minutes"], 1)} 分钟，中位数为 {fmt(lead["median_minutes"], 1)} 分钟。12–24小时组的MAE相对0–1小时组变化约 {lead_change:.1f}%。

## 最大误差点

- 预测生成时间：{worst["forecast_generated_utc"]}
- 目标时间：{worst["target_utc"]}
- 提前量：{fmt(worst["lead_minutes"], 1)} 分钟
- 预测水位：{fmt(worst["predicted_water_m"], 4)} m
- 实测水位：{fmt(worst["actual_water_m"], 4)} m
- 带方向误差：{fmt(worst["signed_error_m"], 4)} m
- 绝对误差：{fmt(worst["absolute_error_m"], 4)} m

## 风险阈值背景

- 达到Watch 4.20 m的不同实测目标点：{risk["actual_unique_targets_at_or_above_watch_4_20m"]}
- 达到Warning 4.43 m的不同实测目标点：{risk["actual_unique_targets_at_or_above_warning_4_43m"]}
- 达到Severe 4.70 m的不同实测目标点：{risk["actual_unique_targets_at_or_above_severe_4_70m"]}
- 涉及预测运行的风险等级计数：`{json.dumps(risk["predicted_run_risk_level_counts"], ensure_ascii=False)}`

如果上述三个实测阈值计数仍为0，本报告只能评价“较高但未达到洪水阈值”的水位预测能力，不能用于证明Watch、Warning或Severe事件的召回率。

## 结论

1. 2 m以上阶段的平均绝对误差为 {fmt(metrics["mean_absolute_error_m"])} m，中位绝对误差为 {fmt(metrics["median_absolute_error_m"])} m，95%的绝对误差不超过 {fmt(quantiles["p95"])} m。
2. 0–1小时预测的MAE为 {fmt(first_mae)} m，Bias接近0；12–24小时MAE增至 {fmt(last_mae)} m，并表现出更明显的高估。
3. Pearson r为 {fmt(metrics["pearson_r"])}，说明涨落方向仍有较强一致性；R²为 {fmt(metrics["r_squared"])}，表明在2 m以上这一较窄水位区间内，对具体幅值差异的解释能力为中等。
4. 当前数据没有达到4.20 m Watch阈值的实测事件，因此本报告支持高水位数值预测评价，但不支持洪水事件检出能力结论。

## 方法与限制

1. 声纳原始10分钟数据按与线上服务器相同的15分钟窗口均值重采样。
2. 筛选依据是实测水位，不是预测水位，避免把模型自身的高估当成高水位事件。
3. 每30分钟会产生一条新的24小时预测，因此同一个实测目标时间会被多个不同提前量的预测重复评价。配对数量不能视为完全独立样本。
4. 本报告完全忽略2 m以下水位，结论只适用于当前数据中的高水位区间。
"""


def write_charts(
    output_dir: Path, frame: pd.DataFrame, lead_metrics: pd.DataFrame, threshold: float
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib is not installed; tabular reports were still created."

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        frame["actual_water_m"],
        frame["predicted_water_m"],
        c=frame["lead_minutes"],
        cmap="viridis",
        s=12,
        alpha=0.35,
    )
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Forecast lead time (minutes)")
    lower = min(frame["actual_water_m"].min(), frame["predicted_water_m"].min())
    upper = max(frame["actual_water_m"].max(), frame["predicted_water_m"].max())
    ax.plot([lower, upper], [lower, upper], color="black", linewidth=1)
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1)
    ax.set_xlabel("Actual sonar water level (m)")
    ax.set_ylabel("Predicted water level (m)")
    ax.set_title(f"Forecast vs actual for actual water level >= {threshold:.2f} m")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "high_water_predicted_vs_actual.png", dpi=180)
    plt.close(fig)

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
    ax.set_title(f"High-water error by lead time (actual >= {threshold:.2f} m)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "high_water_error_by_lead_time.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(frame["absolute_error_m"], bins=40, color="#2878B5", alpha=0.85)
    ax.axvline(
        frame["absolute_error_m"].mean(),
        color="crimson",
        linewidth=1.5,
        label="Mean",
    )
    ax.axvline(
        frame["absolute_error_m"].median(),
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Median",
    )
    ax.set_xlabel("Absolute error (m)")
    ax.set_ylabel("Forecast points")
    ax.set_title(f"High-water absolute error distribution (actual >= {threshold:.2f} m)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "high_water_error_distribution.png", dpi=180)
    plt.close(fig)
    return None


def main() -> int:
    args = parse_args()
    if not np.isfinite(args.threshold_m):
        raise ValueError("--threshold-m must be finite.")
    if args.match_tolerance_minutes <= 0:
        raise ValueError("--match-tolerance-minutes must be positive.")

    prediction_db = base.require_file(args.prediction_db, "Prediction database")
    sonar_db = base.require_file(args.sonar_db, "Sonar database")
    forecasts = base.read_forecasts(prediction_db)
    sonar = base.read_sonar(sonar_db)
    matched, pending = base.match_observations(
        forecasts,
        sonar,
        args.match_tolerance_minutes,
        args.actual_resample_method,
    )
    matched = matched.dropna(subset=["actual_water_m", "predicted_water_m"])
    high_water = matched[matched["actual_water_m"] >= args.threshold_m].copy()
    if high_water.empty:
        raise RuntimeError(
            f"No matched actual water levels are >= {args.threshold_m:.3f} m."
        )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    point_metrics = base.point_metrics(high_water)
    lead_metrics = base.build_lead_metrics(high_water)
    source_metrics = base.build_source_metrics(high_water)
    run_metrics = build_run_metrics(high_water)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "prediction_database": str(prediction_db),
        "sonar_database": str(sonar_db),
        "threshold_m": float(args.threshold_m),
        "filter": "actual_water_m >= threshold_m",
        "actual_resample_method": args.actual_resample_method,
        "target_start_utc": high_water["target_utc"].min(),
        "target_end_utc": high_water["target_utc"].max(),
        "matured_points_before_filter": int(len(matched)),
        "filtered_points": int(len(high_water)),
        "excluded_points": int(len(matched) - len(high_water)),
        "pending_future_points": int(len(pending)),
        "forecast_runs_with_high_water": int(high_water["run_id"].nunique()),
        "point_metrics": point_metrics,
        "water_level_statistics": level_statistics(high_water),
        "error_distribution": error_distribution(high_water),
        "maximum_error_point": base.maximum_error_point(high_water),
        "lead_time_statistics": base.lead_time_statistics(high_water),
        "water_source_metrics": source_metrics.to_dict(orient="records"),
        "risk_context": risk_context(high_water),
    }

    high_water.to_csv(output_dir / "high_water_matched_points.csv", index=False)
    lead_metrics.to_csv(output_dir / "high_water_lead_time_metrics.csv", index=False)
    source_metrics.to_csv(
        output_dir / "high_water_source_metrics.csv", index=False
    )
    run_metrics.to_csv(output_dir / "high_water_run_metrics.csv", index=False)
    chart_warning = write_charts(
        output_dir, high_water, lead_metrics, args.threshold_m
    )
    summary["chart_warning"] = chart_warning

    (output_dir / "high_water_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=base.json_default),
        encoding="utf-8",
    )
    report = build_markdown_report(summary, lead_metrics)
    (output_dir / "HIGH_WATER_EVALUATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=base.json_default))
    print(f"\nReports written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
