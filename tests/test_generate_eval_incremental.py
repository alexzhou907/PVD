import importlib.util
import json
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_eval_incremental.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_eval_incremental", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ScalarLike:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class GenerateEvalIncrementalTests(unittest.TestCase):
    def test_make_batch_artifact_paths_uses_zero_padded_names(self):
        module = load_module()
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = module.make_batch_artifact_paths(Path(tmpdir), 7)

        self.assertEqual(paths["samples"].name, "batch_00007_samples.pth")
        self.assertEqual(paths["refs"].name, "batch_00007_refs.pth")
        self.assertEqual(paths["metrics_log"].name, "batch_metrics.jsonl")
        self.assertEqual(paths["samples_so_far"].name, "samples_so_far.pth")
        self.assertEqual(paths["refs_so_far"].name, "refs_so_far.pth")

    def test_persist_batch_artifacts_writes_batch_files_and_jsonl(self):
        import tempfile

        module = load_module()
        saved = {}

        def fake_save(obj, path):
            path = Path(path)
            saved[path.name] = obj
            path.write_text(json.dumps(obj))

        batch_metrics = {
            "MMD-CD": ScalarLike(0.125),
            "JSD": 0.25,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            record = module.persist_batch_artifacts(
                output_dir=tmp_path,
                batch_idx=3,
                generated_batch=[[1.0, 2.0, 3.0]],
                reference_batch=[[4.0, 5.0, 6.0]],
                metrics=batch_metrics,
                generated_so_far=[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]],
                reference_so_far=[[4.0, 5.0, 6.0], [10.0, 11.0, 12.0]],
                save_fn=fake_save,
            )

            self.assertIn("batch_00003_samples.pth", saved)
            self.assertIn("batch_00003_refs.pth", saved)
            self.assertIn("samples_so_far.pth", saved)
            self.assertIn("refs_so_far.pth", saved)

            metrics_lines = (tmp_path / "batch_metrics.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(metrics_lines), 1)

            payload = json.loads(metrics_lines[0])
            self.assertEqual(payload["batch_idx"], 3)
            self.assertEqual(payload["num_generated"], 1)
            self.assertEqual(payload["num_reference"], 1)
            self.assertEqual(payload["metrics"]["MMD-CD"], 0.125)
            self.assertEqual(payload["metrics"]["JSD"], 0.25)

            self.assertTrue(Path(record["samples"]).exists())
            self.assertTrue(Path(record["refs"]).exists())
            self.assertTrue(Path(record["metrics_log"]).exists())

    def test_export_point_cloud_batch_writes_xyz_files_per_sample(self):
        import tempfile

        module = load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            sample_batch = [
                [[1.0, 2.0, 3.0], [4.5, 5.5, 6.5]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]
            exported = module.export_point_cloud_batch(
                output_dir=out_dir,
                batch_prefix="batch_00003_samples",
                point_cloud_batch=sample_batch,
                export_xyz=True,
                export_ply=False,
            )

            self.assertEqual(len(exported["xyz"]), 2)
            self.assertEqual(exported["xyz"][0].name, "batch_00003_samples_sample_000.xyz")
            self.assertEqual(exported["xyz"][1].name, "batch_00003_samples_sample_001.xyz")
            self.assertEqual(exported["ply"], [])

            xyz_lines = exported["xyz"][0].read_text().strip().splitlines()
            self.assertEqual(xyz_lines[0], "1.000000 2.000000 3.000000")
            self.assertEqual(xyz_lines[1], "4.500000 5.500000 6.500000")


if __name__ == "__main__":
    unittest.main()
