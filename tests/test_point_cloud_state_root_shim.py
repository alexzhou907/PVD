from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PointCloudStateRootShimTests(unittest.TestCase):
    def test_root_shim_can_reach_geometry_bridge(self):
        script = textwrap.dedent(
            f"""
            import sys
            import torch
            sys.path.insert(0, {str(ROOT)!r})
            import point_cloud_state

            state = point_cloud_state.dense_to_point_cloud_state(
                torch.randn(2, 3, 4),
                features=torch.randn(2, 5, 4),
                grid_size=0.25,
            )
            metadata = point_cloud_state.state_to_metadata(state)
            assert metadata.grid_size == 0.25
            assert tuple(metadata.points.shape) == (8, 3)
            print("ok")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
