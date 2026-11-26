import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from vn2_from_scratch.progress import Progress

class ProgressTests(unittest.TestCase):
    def test_snapshot_and_terminal_state(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            with Progress(path) as p:
                p.update("training", 0, 100, model="CatBoost")
                self.assertIsNone(json.loads(path.read_text())["stage_eta_seconds"])
                p.update("training", 50, 100, model="CatBoost")
                data = json.loads(path.read_text())
                self.assertEqual(data["completed"], 50)
                self.assertEqual(data["total"], 100)
                self.assertEqual(data["model"], "CatBoost")
                self.assertGreaterEqual(data["stage_eta_seconds"], 0)
                p.update("simulation", 1, 1, cost=3899., fill_rate=.8679)
                self.assertEqual(json.loads(path.read_text())["cost"], 3899.)
            self.assertEqual(json.loads(path.read_text())["status"], "completed")
            self.assertEqual(list(Path(tmp).iterdir()), [path])

    def test_failure_is_visible(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "progress.json"
            with self.assertRaises(ValueError):
                with Progress(path) as p:
                    p.update("training", 1, 2)
                    raise ValueError("sensitive message")
            data = json.loads(path.read_text())
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["error_type"], "ValueError")
            self.assertNotIn("sensitive message", path.read_text())

if __name__ == "__main__":
    unittest.main()
