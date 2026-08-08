from __future__ import annotations

import unittest

import pandas as pd

import evaluate_high_water as high


class HighWaterEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            {
                "run_id": ["a", "a", "b"],
                "forecast_generated_utc": pd.to_datetime(
                    [
                        "2026-07-24T00:00:00Z",
                        "2026-07-24T00:00:00Z",
                        "2026-07-24T00:30:00Z",
                    ],
                    utc=True,
                ),
                "target_utc": pd.to_datetime(
                    [
                        "2026-07-24T01:00:00Z",
                        "2026-07-24T02:00:00Z",
                        "2026-07-24T02:00:00Z",
                    ],
                    utc=True,
                ),
                "lead_minutes": [60, 120, 90],
                "predicted_water_m": [2.2, 2.4, 2.0],
                "actual_water_m": [1.9, 2.1, 2.1],
                "error_m": [0.3, 0.3, -0.1],
                "absolute_error_m": [0.3, 0.3, 0.1],
                "predicted_risk_level": [0, 0, 0],
            }
        )

    def test_filter_uses_actual_not_predicted_water_level(self) -> None:
        filtered = self.frame[self.frame["actual_water_m"] >= 2.0]
        self.assertEqual(len(filtered), 2)
        self.assertNotIn(1.9, filtered["actual_water_m"].tolist())
        self.assertIn(2.0, filtered["predicted_water_m"].tolist())

    def test_high_water_statistics(self) -> None:
        filtered = self.frame[self.frame["actual_water_m"] >= 2.0]
        distribution = high.error_distribution(filtered)
        levels = high.level_statistics(filtered)
        self.assertAlmostEqual(
            distribution["absolute_error_quantiles_m"]["maximum"], 0.3
        )
        self.assertAlmostEqual(distribution["overprediction_fraction"], 0.5)
        self.assertEqual(levels["unique_actual_target_times"], 1)
        self.assertAlmostEqual(levels["actual_minimum_m"], 2.1)


if __name__ == "__main__":
    unittest.main()
