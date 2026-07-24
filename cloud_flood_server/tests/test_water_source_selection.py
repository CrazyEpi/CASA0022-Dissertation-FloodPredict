from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd


APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import server  # noqa: E402


class LazyRuntimeInitializationTests(unittest.TestCase):
    def test_import_does_not_initialize_model_runtime(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, server; "
                    "print(server._RUNTIME_SERVICE is None); "
                    "print('torch' in sys.modules)"
                ),
            ],
            cwd=APP_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip().splitlines(), ["True", "False"])


class PredictionArchiveTests(unittest.TestCase):
    def test_archives_complete_forecast_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "predictions.sqlite3"
            payload = {
                "site": "house_mill",
                "source": "patchtst_15min",
                "forecast_generated_utc": "2026-07-23T12:02:00Z",
                "history_last_utc": "2026-07-23T12:00:00Z",
                "valid_until_utc": "2026-07-23T12:32:00Z",
                "risk_level": 1,
                "risk_label": "Watch",
                "max_predicted_m": 4.25,
                "eta_minutes": None,
                "data_quality": {"water_source": "mqtt_sqlite:test"},
                "model": {"interval_minutes": 15},
                "forecast": [
                    {
                        "time_utc": "2026-07-23T12:15:00Z",
                        "water_level_m": 4.1,
                        "flood_probability": 0.2,
                    },
                    {
                        "time_utc": "2026-07-23T12:30:00Z",
                        "water_level_m": 4.25,
                        "flood_probability": 0.4,
                    },
                ],
            }
            env = {
                "PREDICTION_ARCHIVE_ENABLED": "1",
                "PREDICTION_DATABASE": str(database),
            }
            with patch.dict(os.environ, env, clear=False):
                first = server.archive_prediction(payload)
                second = server.archive_prediction(payload)

            self.assertTrue(first["archived"])
            self.assertEqual(first["run_id"], second["run_id"])
            connection = server.connect_prediction_database(database)
            try:
                run_count = connection.execute(
                    "SELECT COUNT(*) FROM forecast_runs"
                ).fetchone()[0]
                rows = connection.execute(
                    """
                    SELECT lead_minutes, predicted_water_m
                    FROM forecast_points
                    ORDER BY target_utc
                    """
                ).fetchall()
            finally:
                connection.close()

            self.assertEqual(run_count, 1)
            self.assertEqual(rows, [(15, 4.1), (30, 4.25)])


class WaterSourceSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.end = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)
        self.start = self.end - timedelta(
            minutes=(server.SEQ_LEN - 1) * server.INTERVAL_MINUTES
        )
        self.base_env = {
            "PRIMARY_WATER_LEVEL_SOURCE": "auto",
            "SONAR_READY_MIN_COVERAGE": "0.90",
            "SONAR_READY_MIN_SPAN_HOURS": "167.0",
            "SONAR_READY_MAX_AGE_MINUTES": "30",
            "SONAR_READY_MAX_FUTURE_MINUTES": "15",
            "SONAR_ALLOW_DEGRADED_FALLBACK": "1",
        }

    @staticmethod
    def water_frame(start: datetime, end: datetime, frequency: str) -> pd.DataFrame:
        dates = pd.date_range(start=start, end=end, freq=frequency, tz="UTC")
        return pd.DataFrame(
            {
                "date": dates,
                server.TARGET_COLUMN: [4.0 + index / 100000 for index in range(len(dates))],
            }
        )

    def api_frame(self) -> pd.DataFrame:
        return self.water_frame(self.start, self.end, "15min")

    def test_short_sonar_uses_api_during_warmup(self) -> None:
        sonar = self.water_frame(self.end - timedelta(hours=24), self.end, "10min")
        with (
            patch.dict(os.environ, self.base_env, clear=False),
            patch.object(
                server,
                "load_configured_sonar_history",
                return_value=(sonar, "mqtt_test"),
            ),
            patch.object(
                server,
                "fetch_primary_api_water_history",
                return_value=(self.api_frame(), "api_test"),
            ),
        ):
            selection = server.choose_water_level_history(self.start, self.end)

        self.assertEqual(selection.mode, "api_warmup")
        self.assertEqual(selection.source_label, "api_test")
        self.assertFalse(selection.diagnostics["sonar_ready"])
        self.assertIn(
            "span_below_seven_days", selection.diagnostics["sonar_reasons"]
        )

    def test_complete_fresh_sonar_becomes_primary(self) -> None:
        sonar = self.water_frame(self.start, self.end, "10min")
        with (
            patch.dict(os.environ, self.base_env, clear=False),
            patch.object(
                server,
                "load_configured_sonar_history",
                return_value=(sonar, "mqtt_test"),
            ),
            patch.object(
                server,
                "fetch_primary_api_water_history",
                side_effect=AssertionError("API should not be called for ready sonar"),
            ),
        ):
            selection = server.choose_water_level_history(self.start, self.end)

        self.assertEqual(selection.mode, "sonar_ready")
        self.assertEqual(selection.source_label, "mqtt_test")
        self.assertTrue(selection.diagnostics["sonar_ready"])
        self.assertGreaterEqual(selection.diagnostics["sonar_coverage_ratio"], 0.90)

    def test_stale_sonar_stays_on_api(self) -> None:
        stale_end = self.end - timedelta(hours=2)
        sonar = self.water_frame(self.start - timedelta(hours=2), stale_end, "10min")
        with (
            patch.dict(os.environ, self.base_env, clear=False),
            patch.object(
                server,
                "load_configured_sonar_history",
                return_value=(sonar, "mqtt_test"),
            ),
            patch.object(
                server,
                "fetch_primary_api_water_history",
                return_value=(self.api_frame(), "api_test"),
            ),
        ):
            selection = server.choose_water_level_history(self.start, self.end)

        self.assertEqual(selection.mode, "api_warmup")
        self.assertIn(
            "latest_point_is_stale", selection.diagnostics["sonar_reasons"]
        )

    def test_api_failure_uses_incomplete_sonar_as_degraded_fallback(self) -> None:
        sonar = self.water_frame(self.end - timedelta(hours=24), self.end, "10min")
        with (
            patch.dict(os.environ, self.base_env, clear=False),
            patch.object(
                server,
                "load_configured_sonar_history",
                return_value=(sonar, "mqtt_test"),
            ),
            patch.object(
                server,
                "fetch_primary_api_water_history",
                side_effect=RuntimeError("simulated API outage"),
            ),
        ):
            selection = server.choose_water_level_history(self.start, self.end)

        self.assertEqual(selection.mode, "sonar_degraded")
        self.assertIn("simulated API outage", selection.diagnostics["water_api_error"])

    def test_sqlite_storage_deduplicates_repeated_uplink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            database = Path(temporary_directory) / "sonar.sqlite3"
            csv_path = Path(temporary_directory) / "legacy.csv"
            env = {
                "SONAR_STORAGE": "sqlite",
                "SONAR_DATABASE": str(database),
                "SONAR_HISTORY_CSV": str(csv_path),
                "SONAR_IMPORT_LEGACY_CSV": "0",
                "SONAR_CSV_MIRROR": "0",
                "WATER_RECOMPUTE_FROM_DISTANCE": "1",
                "SONAR_APPLY_MEDIAN_FILTER": "1",
            }
            record = {
                "date": "2026-07-13T12:00:00Z",
                "topic": "v3/test/up",
                "device_id": "test-device",
                "dev_eui": "A8610A3233458C03",
                "f_cnt": 658,
                "packet_count": 658,
                "distance_mm": 4309.0,
                server.TARGET_COLUMN: 0.691,
            }
            with patch.dict(os.environ, env, clear=False):
                server.append_sonar_record(record)
                server.append_sonar_record(record)
                frame, label = server.load_configured_sonar_history()

            self.assertEqual(len(frame), 1)
            self.assertTrue(label.startswith("mqtt_sqlite:"))
            self.assertAlmostEqual(float(frame.iloc[0][server.TARGET_COLUMN]), 0.691)


if __name__ == "__main__":
    unittest.main()
