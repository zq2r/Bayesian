import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "eval" / "test_time_scaling.py"
SPEC = importlib.util.spec_from_file_location("test_time_scaling", MODULE_PATH)
tts = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tts)


class TestTimeScalingTest(unittest.TestCase):
    def _records(self):
        return [{
            "solutions_splits": [[f"step {i}"] for i in range(4)],
            "prm_scores": [[i / 4] for i in range(4)],
            "labels": [0, 0, 0, 1],
        }]

    def test_default_protocol_keeps_twenty_repeats(self):
        result = tts.evaluate_test_time_scaling(self._records(), n_grid=(1, 4))
        self.assertEqual(result["repeats"], 20)
        self.assertEqual(len(result["results"]["4"]["repeat_results"]), 20)
        self.assertEqual(result["results"]["4"]["accuracy_mean"], 1.0)
        self.assertEqual(result["results"]["4"]["oracle_accuracy_mean"], 1.0)

    def test_bayesian_head_scores_are_supported(self):
        record = self._records()[0]
        record.pop("prm_scores")
        record["prm_mu_heads"] = [[[0.1, 0.2]] for _ in range(4)]
        record["prm_rel_weights"] = [[[0.5, 0.5]] for _ in range(4)]
        result = tts.evaluate_test_time_scaling([record], n_grid=(2,), repeats=1)
        self.assertEqual(len(result["results"]["2"]["repeat_results"]), 1)


if __name__ == "__main__":
    unittest.main()
