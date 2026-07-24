from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import evaluate_forecasts as evaluator


class ForecastEvaluationTests(unittest.TestCase):
    def test_default_mean_resampling_matches_server_interval_logic(self) -> None:
        forecasts = pd.DataFrame(
            {
                "run_id": ["run-mean"],
                "target_utc": pd.to_datetime(["2026-07-23T00:15:00Z"], utc=True),
                "predicted_water_m": [4.15],
            }
        )
        sonar = pd.DataFrame(
            {
                "actual_utc": pd.to_datetime(
                    [
                        "2026-07-23T00:16:00Z",
                        "2026-07-23T00:26:00Z",
                        "2026-07-23T00:45:00Z",
                    ],
                    utc=True,
                ),
                "actual_water_m": [4.0, 4.2, 4.3],
            }
        )

        matched, pending = evaluator.match_observations(forecasts, sonar, 8)

        self.assertTrue(pending.empty)
        self.assertEqual(int(matched.iloc[0]["actual_samples_in_bin"]), 2)
        self.assertAlmostEqual(float(matched.iloc[0]["actual_water_m"]), 4.1)

    def test_matches_ten_minute_sonar_and_scores_completed_run(self) -> None:
        forecasts = pd.DataFrame(
            {
                "run_id": ["run-1"] * 4,
                "forecast_generated_utc": pd.to_datetime(
                    ["2026-07-23T00:00:00Z"] * 4, utc=True
                ),
                "history_last_utc": pd.to_datetime(
                    ["2026-07-23T00:00:00Z"] * 4, utc=True
                ),
                "site": ["house_mill"] * 4,
                "source": ["patchtst_15min"] * 4,
                "predicted_risk_level": [2] * 4,
                "predicted_risk_label": ["Warning"] * 4,
                "max_predicted_m": [4.5] * 4,
                "predicted_eta_minutes": [60] * 4,
                "target_utc": pd.to_datetime(
                    [
                        "2026-07-23T00:15:00Z",
                        "2026-07-23T00:30:00Z",
                        "2026-07-23T00:45:00Z",
                        "2026-07-23T01:00:00Z",
                    ],
                    utc=True,
                ),
                "lead_minutes": [15, 30, 45, 60],
                "predicted_water_m": [4.1, 4.2, 4.4, 4.5],
                "flood_probability": [0.1, 0.2, 0.7, 0.9],
            }
        )
        sonar = pd.DataFrame(
            {
                "actual_utc": pd.to_datetime(
                    [
                        "2026-07-23T00:10:00Z",
                        "2026-07-23T00:30:00Z",
                        "2026-07-23T00:40:00Z",
                        "2026-07-23T01:00:00Z",
                    ],
                    utc=True,
                ),
                "actual_water_m": [4.0, 4.25, 4.35, 4.45],
            }
        )

        matched, pending = evaluator.match_observations(
            forecasts, sonar, 8, method="nearest"
        )
        runs = evaluator.build_run_metrics(
            forecasts,
            matched,
            sonar["actual_utc"].max(),
            minimum_coverage=0.90,
            watch=4.20,
            warning=4.43,
            severe=4.70,
        )

        self.assertTrue(pending.empty)
        self.assertEqual(matched["actual_water_m"].notna().sum(), 4)
        self.assertEqual(len(runs), 1)
        self.assertEqual(int(runs.iloc[0]["actual_risk_level"]), 3)
        metrics = evaluator.point_metrics(matched)
        self.assertAlmostEqual(metrics["mae_m"], 0.0625, places=6)
        self.assertAlmostEqual(metrics["median_absolute_error_m"], 0.05, places=6)
        self.assertAlmostEqual(metrics["maximum_absolute_error_m"], 0.1, places=6)
        lead_times = evaluator.lead_time_statistics(matched)
        self.assertEqual(lead_times["minimum_minutes"], 15)
        self.assertEqual(lead_times["maximum_minutes"], 60)
        self.assertEqual(lead_times["mean_minutes"], 37.5)
        self.assertEqual(lead_times["median_minutes"], 37.5)
        worst = evaluator.maximum_error_point(matched)
        self.assertIsNotNone(worst)
        self.assertAlmostEqual(worst["absolute_error_m"], 0.1, places=6)
        self.assertEqual(worst["lead_minutes"], 15)

    def test_database_readers_reject_legacy_sonar_only_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "legacy.sqlite3"
            connection = sqlite3.connect(database)
            try:
                connection.execute(
                    "CREATE TABLE sonar_readings (id INTEGER, date_utc TEXT, internal_water_m REAL)"
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(RuntimeError, "forecast_runs"):
                evaluator.read_forecasts(database)


if __name__ == "__main__":
    unittest.main()
