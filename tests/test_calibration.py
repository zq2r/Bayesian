import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "eval" / "calibration.py"
SPEC = importlib.util.spec_from_file_location("calibration", MODULE_PATH)
calibration = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(calibration)


class CalibrationTest(unittest.TestCase):
    def test_metrics_match_probability_errors(self):
        metrics, diagnostics = calibration.compute_metrics(
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 0.0],
            num_bins=2,
        )
        self.assertAlmostEqual(metrics["brier"], (0.0 + 0.25 + 1.0) / 3)
        self.assertAlmostEqual(metrics["positive_brier"], 1.0 / 3)
        self.assertAlmostEqual(metrics["ece"], 1.0 / 6)
        self.assertAlmostEqual(metrics["average_ce"], 0.125)
        self.assertEqual(sum(item["count"] for item in diagnostics["fixed_width_bins"]), 3)

    def test_grouped_predictions_align_with_mc_counts(self):
        predictions = [{
            "prefix_ids": ["p0", "p1"],
            "prm_mu_rel": [[0.2, 0.3], [0.7]],
            "prm_scores": [[0.4, 0.5], [0.8]],
        }]
        labels = [
            {"prefix_id": "p0", "mc_correct": 1, "mc_total": 2},
            {"prefix_id": "p1", "success_prob": 1.0},
        ]
        result = calibration.evaluate_calibration(predictions, labels, num_bins=2)
        self.assertEqual(result["models"]["final"]["num_prefixes"], 2)
        self.assertIn("positive_brier", result["models"]["reliability"])

    def test_rollout_labels_can_form_monte_carlo_target(self):
        targets = calibration.load_mc_targets([
            {"prefix_id": "p", "rollout_labels": [1, 0, 1, 1]},
        ])
        self.assertEqual(targets["p"], 0.75)

    def test_mc_probability_must_match_counts(self):
        with self.assertRaises(ValueError):
            calibration.load_mc_targets([{
                "prefix_id": "p",
                "success_prob": 0.0,
                "mc_correct": 1,
                "mc_total": 2,
            }])


if __name__ == "__main__":
    unittest.main()
