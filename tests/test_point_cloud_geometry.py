import importlib.util
import pathlib
import tempfile
import sys
import unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "modules"))
sys.path.insert(0, str(ROOT))

from point_cloud_state import build_state_triplets, build_state_with_stride, dense_to_point_cloud_state, state_to_metadata


GEOMETRY_PATH = ROOT / "modules" / "point_cloud_geometry.py"


def load_point_cloud_geometry_module():
    spec = importlib.util.spec_from_file_location("pvd_point_cloud_geometry_test", GEOMETRY_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PointCloudGeometryTests(unittest.TestCase):
    def test_resolve_pointelligence_root_accepts_monorepo_layout(self):
        geometry = load_point_cloud_geometry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir) / "pointelligence"
            module_file = root / "PVD.remote.sync" / "modules" / "point_cloud_geometry.py"
            (root / "layers").mkdir(parents=True)
            (root / "layers" / "metadata.py").write_text("# stub\n")
            module_file.parent.mkdir(parents=True)
            module_file.write_text("# stub\n")

            resolved = geometry._resolve_pointelligence_root(module_file)

            self.assertEqual(resolved, root)

    def test_resolve_pointelligence_root_accepts_sibling_layout(self):
        geometry = load_point_cloud_geometry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = pathlib.Path(tmpdir)
            pointelligence_root = workspace / "pointelligence"
            module_file = workspace / "PVD.codex" / "modules" / "point_cloud_geometry.py"
            (pointelligence_root / "layers").mkdir(parents=True)
            (pointelligence_root / "layers" / "metadata.py").write_text("# stub\n")
            module_file.parent.mkdir(parents=True)
            module_file.write_text("# stub\n")

            resolved = geometry._resolve_pointelligence_root(module_file)

            self.assertEqual(resolved, pointelligence_root)

    def test_sparse_engine_shim_prefers_real_package(self):
        geometry = load_point_cloud_geometry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            sparse_dir = tmp_path / "sparse_engines"
            sparse_dir.mkdir()
            (sparse_dir / "__init__.py").write_text("", encoding="utf-8")
            (sparse_dir / "indexed_distance_triton.py").write_text("VALUE = 'real_distance'\n", encoding="utf-8")

            saved_modules = {
                name: sys.modules.pop(name, None)
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]
            }
            sys.path.insert(0, tmpdir)
            try:
                geometry._install_sparse_engine_shim()
                imported = importlib.import_module("sparse_engines.indexed_distance_triton")
                self.assertEqual(imported.VALUE, "real_distance")
            finally:
                sys.path.remove(tmpdir)
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]:
                    sys.modules.pop(name, None)
                for name, value in saved_modules.items():
                    if value is not None:
                        sys.modules[name] = value

    def test_sparse_engine_shim_wraps_real_package_with_cpu_fallback(self):
        geometry = load_point_cloud_geometry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = pathlib.Path(tmpdir)
            sparse_dir = tmp_path / "sparse_engines"
            sparse_dir.mkdir()
            (sparse_dir / "__init__.py").write_text("", encoding="utf-8")
            (sparse_dir / "indexed_distance_triton.py").write_text(
                "\n".join(
                    [
                        "def indexed_distance(queries, q_inds, points, p_inds, distance_type='ball'):",
                        "    raise RuntimeError('gpu_only')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            sparse_cuda_dir = tmp_path / "sparse_engines_cuda"
            sparse_cuda_dir.mkdir()
            (sparse_cuda_dir / "__init__.py").write_text("", encoding="utf-8")
            (sparse_cuda_dir / "ops.py").write_text(
                "\n".join(
                    [
                        "def bucket_arrange(bucket_indices, num_buckets):",
                        "    raise RuntimeError('gpu_only')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            saved_modules = {
                name: sys.modules.pop(name, None)
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]
            }
            sys.path.insert(0, tmpdir)
            try:
                geometry._install_sparse_engine_shim()

                indexed_distance_mod = importlib.import_module("sparse_engines.indexed_distance_triton")
                distances = indexed_distance_mod.indexed_distance(
                    queries=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                    q_inds=torch.tensor([0], dtype=torch.int32),
                    points=torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
                    p_inds=torch.tensor([0], dtype=torch.int32),
                )
                self.assertTrue(torch.equal(distances, torch.tensor([1.0], dtype=torch.float32)))

                sparse_cuda_ops = importlib.import_module("sparse_engines_cuda.ops")
                bucket_sizes, bucket_slots = sparse_cuda_ops.bucket_arrange(
                    torch.tensor([1, 0, 1, 1], dtype=torch.int32),
                    3,
                )
                self.assertTrue(torch.equal(bucket_sizes, torch.tensor([1, 3, 0], dtype=torch.int32)))
                self.assertTrue(torch.equal(bucket_slots, torch.tensor([0, 0, 1, 2], dtype=torch.int32)))
            finally:
                sys.path.remove(tmpdir)
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]:
                    sys.modules.pop(name, None)
                for name, value in saved_modules.items():
                    if value is not None:
                        sys.modules[name] = value

    def test_load_geometry_api_finds_sparse_cuda_under_extensions(self):
        geometry = load_point_cloud_geometry_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir) / "pointelligence"
            layers_dir = root / "layers"
            sparse_dir = root / "sparse_engines"
            sparse_cuda_dir = root / "extensions" / "sparse_engines_cuda"
            layers_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            sparse_cuda_dir.mkdir(parents=True)

            (layers_dir / "metadata.py").write_text(
                "class MetaData:\n"
                "    def __init__(self, **kwargs):\n"
                "        self.__dict__.update(kwargs)\n",
                encoding="utf-8",
            )
            (layers_dir / "downsample.py").write_text("", encoding="utf-8")
            (layers_dir / "triplets.py").write_text(
                "\n".join(
                    [
                        "def build_triplets(*args, **kwargs):",
                        "    return None, None, None, None",
                        "def radius_scaler_for_kernel_size(*args, **kwargs):",
                        "    return 1.0",
                        "def handle_stride_and_build_triplets(*args, **kwargs):",
                        "    return None",
                        "def voxelize_3d(*args, **kwargs):",
                        "    return None",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (sparse_dir / "__init__.py").write_text("", encoding="utf-8")
            (sparse_dir / "indexed_distance_triton.py").write_text(
                "def indexed_distance(*args, **kwargs):\n"
                "    return 'real_distance'\n",
                encoding="utf-8",
            )
            (sparse_cuda_dir / "__init__.py").write_text("from . import _C, ops\n", encoding="utf-8")
            (sparse_cuda_dir / "_C.py").write_text("", encoding="utf-8")
            (sparse_cuda_dir / "ops.py").write_text(
                "VALUE = 'real_bucket_ops'\n"
                "def bucket_arrange(bucket_indices, num_buckets):\n"
                "    return bucket_indices.new_zeros((num_buckets,)), bucket_indices.new_zeros(bucket_indices.shape)\n",
                encoding="utf-8",
            )

            saved_modules = {
                name: sys.modules.pop(name, None)
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda._C",
                    "sparse_engines_cuda.ops",
                    "_pvd_pointelligence_layers",
                    "_pvd_pointelligence_layers.metadata",
                    "_pvd_pointelligence_layers.downsample",
                    "_pvd_pointelligence_layers.triplets",
                ]
            }
            old_cache = geometry._POINTELLIGENCE_CACHE
            original_resolve = geometry._resolve_pointelligence_root
            geometry._POINTELLIGENCE_CACHE = None
            geometry._resolve_pointelligence_root = lambda module_file: root
            original_sys_path = list(sys.path)
            try:
                geometry._load_pointelligence_geometry_api()
                sparse_cuda_ops = importlib.import_module("sparse_engines_cuda.ops")
                self.assertEqual(sparse_cuda_ops.VALUE, "real_bucket_ops")
            finally:
                geometry._POINTELLIGENCE_CACHE = old_cache
                geometry._resolve_pointelligence_root = original_resolve
                sys.path[:] = original_sys_path
                for name in [
                    "sparse_engines",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda._C",
                    "sparse_engines_cuda.ops",
                    "_pvd_pointelligence_layers",
                    "_pvd_pointelligence_layers.metadata",
                    "_pvd_pointelligence_layers.downsample",
                    "_pvd_pointelligence_layers.triplets",
                ]:
                    sys.modules.pop(name, None)
                for name, value in saved_modules.items():
                    if value is not None:
                        sys.modules[name] = value

    def make_state(self):
        coords = torch.tensor(
            [
                [
                    [0.00, 0.25, 0.50, 0.75],
                    [0.00, 0.00, 0.00, 0.00],
                    [0.00, 0.00, 0.00, 0.00],
                ],
                [
                    [0.00, 0.25, 0.50, 0.75],
                    [1.00, 1.00, 1.00, 1.00],
                    [0.00, 0.00, 0.00, 0.00],
                ],
            ],
            dtype=torch.float32,
        )
        features = torch.randn(2, 6, 4)
        return dense_to_point_cloud_state(coords, features=features, grid_size=0.25)

    def test_build_state_triplets_matches_pointelligence_metadata(self):
        geometry = load_point_cloud_geometry_module()
        _, build_triplets, radius_scaler_for_kernel_size, handle_stride_and_build_triplets, voxelize_3d = geometry._load_pointelligence_geometry_api()
        state = self.make_state()

        metadata = state_to_metadata(state)
        radius_scaler = radius_scaler_for_kernel_size(3, 1.0, "ball")
        metadata.i, metadata.j, metadata.k, metadata.num_neighbors = build_triplets(
            points=metadata.points,
            sample_inds=metadata.sample_inds,
            sample_sizes=metadata.sample_sizes,
            neighbor_radius=metadata.grid_size * radius_scaler,
            kernel_indexer=lambda points, grid_size: voxelize_3d(kernel_size=3, points=points, grid_size=grid_size),
            query_points=metadata.query_points,
            query_sample_inds=metadata.query_sample_inds,
            query_sample_sizes=metadata.query_sample_sizes,
            sort_by="k",
            return_num_neighbors=True,
            radius_scaler=radius_scaler,
        )
        state_out = build_state_triplets(state, kernel_size=3, return_num_neighbors=True)

        self.assertTrue(torch.equal(state_out.i, metadata.i))
        self.assertTrue(torch.equal(state_out.j, metadata.j))
        self.assertTrue(torch.equal(state_out.k, metadata.k))
        self.assertTrue(torch.equal(state_out.num_neighbors, metadata.num_neighbors))
        self.assertTrue(torch.equal(state_out.features, state.features))
        self.assertEqual(handle_stride_and_build_triplets.__name__, "handle_stride_and_build_triplets")

    def test_build_state_with_stride_matches_pointelligence_metadata(self):
        geometry = load_point_cloud_geometry_module()
        _, _, _, handle_stride_and_build_triplets, _ = geometry._load_pointelligence_geometry_api()
        state = self.make_state()

        metadata = state_to_metadata(state)
        metadata = handle_stride_and_build_triplets(metadata, stride=2, kernel_size=3)
        state_out = build_state_with_stride(state, stride=2, kernel_size=3)

        self.assertTrue(torch.equal(state_out.coords, metadata.points))
        self.assertTrue(torch.equal(state_out.sample_inds, metadata.sample_inds))
        self.assertTrue(torch.equal(state_out.sample_sizes, metadata.sample_sizes))
        self.assertTrue(torch.equal(state_out.i, metadata.i))
        self.assertTrue(torch.equal(state_out.j, metadata.j))
        self.assertTrue(torch.equal(state_out.k, metadata.k))
        self.assertTrue(torch.equal(state_out.downsample_indices, metadata.downsample_indices))
        self.assertIsNotNone(state_out.parent)
        self.assertTrue(torch.equal(state_out.parent.i_upsample, metadata.parent.i_upsample))
        self.assertTrue(torch.equal(state_out.parent.j_upsample, metadata.parent.j_upsample))
        self.assertTrue(torch.equal(state_out.parent.k_upsample, metadata.parent.k_upsample))
        self.assertIsNone(state_out.features)


if __name__ == "__main__":
    unittest.main()
