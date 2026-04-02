import importlib.util
from pathlib import Path
import tempfile
from types import ModuleType
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from point_cloud_state import (
    dense_to_point_cloud_state,
    point_cloud_state_to_dense,
    PointCloudState,
)


POINTELLIGENCE_PVCONV_PATH = Path(__file__).resolve().parents[1] / "modules" / "pvconv_pointelligence.py"
SHARED_MLP_PATH = Path(__file__).resolve().parents[1] / "modules" / "shared_mlp.py"


def load_shared_mlp_module():
    spec = importlib.util.spec_from_file_location("pvd_shared_mlp_for_pointelligence_test", SHARED_MLP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pointelligence_pvconv_module():
    shared_mlp_module = load_shared_mlp_module()

    for name in [
        "modules",
        "modules.shared_mlp",
        "pvd_pointelligence_pvconv_test",
    ]:
        sys.modules.pop(name, None)

    modules_pkg = ModuleType("modules")
    modules_pkg.__path__ = []
    shared_mlp_mod = ModuleType("modules.shared_mlp")
    shared_mlp_mod.SharedMLP = shared_mlp_module.SharedMLP

    sys.modules["modules"] = modules_pkg
    sys.modules["modules.shared_mlp"] = shared_mlp_mod

    spec = importlib.util.spec_from_file_location("pvd_pointelligence_pvconv_test", POINTELLIGENCE_PVCONV_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeMetaData:
    def __init__(
        self,
        *,
        points,
        sample_inds,
        sample_sizes,
        grid_size,
        i=None,
        j=None,
        k=None,
        downsample_indices=None,
        i_upsample=None,
        j_upsample=None,
        k_upsample=None,
        parent=None,
    ):
        self.points = points
        self.sample_inds = sample_inds
        self.sample_sizes = sample_sizes
        self.grid_size = grid_size
        self.i = i
        self.j = j
        self.k = k
        self.downsample_indices = downsample_indices
        self.i_upsample = i_upsample
        self.j_upsample = j_upsample
        self.k_upsample = k_upsample
        self.parent = parent

    def num_points(self):
        return int(self.points.shape[0])


class FakePointConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size_3 = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size_3 = tuple(kernel_size)
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x, i, j, k, n):
        del k
        i = i.to(dtype=torch.long, device=x.device)
        j = j.to(dtype=torch.long, device=x.device)
        gathered = x.index_select(0, j)
        reduced = x.new_zeros((n, x.shape[1]))
        reduced.index_add_(0, i, gathered)
        counts = x.new_zeros((n, 1))
        counts.index_add_(0, i, torch.ones((i.numel(), 1), device=x.device, dtype=x.dtype))
        return self.proj(reduced / counts.clamp_min(1.0))


class TrackingPointConv3d(FakePointConv3d):
    calls = []

    def forward(self, x, i, j, k, n):
        TrackingPointConv3d.calls.append(
            {
                "i_ptr": i.data_ptr(),
                "j_ptr": j.data_ptr(),
                "k_ptr": k.data_ptr(),
                "n": n,
            }
        )
        return super().forward(x, i, j, k, n)


def fake_conv_with_stride(conv_op, x, metadata, stride, receptive_field_scaler=1.0):
    del receptive_field_scaler
    state = fake_metadata_to_state(metadata)
    state = fake_build_state_with_stride(
        state,
        stride=stride,
        kernel_size=conv_op.kernel_size_3,
    )
    metadata = fake_state_to_metadata(state)
    return conv_op(x, metadata.i, metadata.j, metadata.k, metadata.num_points()), metadata


class FakeUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=False, receptive_field_scaler=1.0):
        super().__init__()
        del kernel_size, receptive_field_scaler
        self.proj = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x_low, m_low):
        parent = m_low.parent
        i_high = parent.i_upsample.to(dtype=torch.long, device=x_low.device)
        j_low = parent.j_upsample.to(dtype=torch.long, device=x_low.device)
        gathered = x_low.index_select(0, j_low)
        high = x_low.new_zeros((parent.points.shape[0], x_low.shape[1]))
        high.index_add_(0, i_high, gathered)
        counts = x_low.new_zeros((parent.points.shape[0], 1))
        counts.index_add_(0, i_high, torch.ones((i_high.numel(), 1), device=x_low.device, dtype=x_low.dtype))
        high = self.proj(high / counts.clamp_min(1.0))
        restored = FakeMetaData(
            points=parent.points,
            sample_inds=parent.sample_inds,
            sample_sizes=parent.sample_sizes,
            grid_size=parent.grid_size,
            parent=parent.parent,
        )
        return high, restored


class FailUpsample(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise AssertionError("same-scale pointelligence backend should not call Upsample")


def fake_state_to_metadata(state):
    parent = fake_state_to_metadata(state.parent) if state.parent is not None else None
    return FakeMetaData(
        points=state.coords,
        sample_inds=state.sample_inds,
        sample_sizes=state.sample_sizes,
        grid_size=state.grid_size,
        i=state.i,
        j=state.j,
        k=state.k,
        downsample_indices=state.downsample_indices,
        i_upsample=state.i_upsample,
        j_upsample=state.j_upsample,
        k_upsample=state.k_upsample,
        parent=parent,
    )


def fake_metadata_to_state(metadata, features=None):
    parent = fake_metadata_to_state(metadata.parent) if metadata.parent is not None else None
    return PointCloudState(
        coords=metadata.points,
        features=features,
        sample_inds=metadata.sample_inds,
        sample_sizes=metadata.sample_sizes,
        grid_size=metadata.grid_size,
        i=metadata.i,
        j=metadata.j,
        k=metadata.k,
        downsample_indices=metadata.downsample_indices,
        i_upsample=metadata.i_upsample,
        j_upsample=metadata.j_upsample,
        k_upsample=metadata.k_upsample,
        parent=parent,
    )


def fake_build_state_triplets(state, kernel_size, receptive_field_scaler=1.0, **kwargs):
    del kwargs
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    point_count = state.coords.shape[0]
    i = torch.arange(point_count, dtype=torch.long, device=state.coords.device)
    j = i.clone()
    k = torch.zeros(point_count, dtype=torch.long, device=state.coords.device)
    return state.replace_geometry(
        i=i,
        j=j,
        k=k,
        kernel_size=kernel_size,
        receptive_field_scaler=receptive_field_scaler,
        sort_by="k",
        return_num_neighbors=False,
        distance_type="ball",
    )


def fake_build_state_with_stride(state, stride, kernel_size, receptive_field_scaler=1.0, **kwargs):
    del stride, kernel_size, receptive_field_scaler, kwargs
    sample_sizes = state.sample_sizes.tolist()
    keep_indices = []
    upsample_i = []
    upsample_j = []
    start = 0
    down_start = 0
    down_sizes = []
    for count in sample_sizes:
        local = list(range(start, start + count))
        local_keep = local[::2] or [local[0]]
        keep_indices.extend(local_keep)
        down_sizes.append(len(local_keep))
        for high_idx in local:
            nearest = min(range(len(local_keep)), key=lambda idx: abs(local_keep[idx] - high_idx))
            upsample_i.append(high_idx)
            upsample_j.append(down_start + nearest)
        start += count
        down_start += len(local_keep)

    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=state.coords.device)
    down_sizes = torch.tensor(down_sizes, dtype=torch.long, device=state.coords.device)
    down_sample_inds = torch.arange(down_sizes.numel(), device=state.coords.device, dtype=torch.long).repeat_interleave(down_sizes)
    down_coords = state.coords.index_select(0, keep_indices)
    i = torch.arange(down_coords.shape[0], dtype=torch.long, device=state.coords.device)
    j = keep_indices.clone()
    k = torch.zeros(down_coords.shape[0], dtype=torch.long, device=state.coords.device)
    parent = state.geometry_only().replace_geometry(
        i_upsample=torch.tensor(upsample_i, dtype=torch.long, device=state.coords.device),
        j_upsample=torch.tensor(upsample_j, dtype=torch.long, device=state.coords.device),
        k_upsample=torch.zeros(len(upsample_i), dtype=torch.long, device=state.coords.device),
    )
    return PointCloudState(
        coords=down_coords,
        features=None,
        sample_inds=down_sample_inds,
        sample_sizes=down_sizes,
        grid_size=state.grid_size,
        parent=parent,
        i=i,
        j=j,
        k=k,
        downsample_indices=keep_indices,
    )


class PointelligencePVConvTests(unittest.TestCase):
    def test_loader_prefers_real_sparse_engines_package(self):
        module = load_pointelligence_pvconv_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sparse_dir = tmp_path / "sparse_engines"
            sparse_dir.mkdir()
            (sparse_dir / "__init__.py").write_text("", encoding="utf-8")
            (sparse_dir / "ops.py").write_text("VALUE = 'real_ops'\n", encoding="utf-8")

            saved_modules = {
                name: sys.modules.pop(name, None)
                for name in [
                    "sparse_engines",
                    "sparse_engines.ops",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]
            }
            sys.path.insert(0, tmpdir)
            try:
                module._install_sparse_engine_shim()
                imported = importlib.import_module("sparse_engines.ops")
                self.assertEqual(imported.VALUE, "real_ops")
            finally:
                sys.path.remove(tmpdir)
                for name in [
                    "sparse_engines",
                    "sparse_engines.ops",
                    "sparse_engines.indexed_distance_triton",
                    "sparse_engines_cuda",
                    "sparse_engines_cuda.ops",
                ]:
                    sys.modules.pop(name, None)
                for name, value in saved_modules.items():
                    if value is not None:
                        sys.modules[name] = value

    def test_load_pvconv_api_finds_sparse_cuda_under_extensions(self):
        module = load_pointelligence_pvconv_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "pointelligence"
            layers_dir = root / "layers"
            sparse_dir = root / "sparse_engines"
            sparse_cuda_dir = root / "extensions" / "sparse_engines_cuda"
            layers_dir.mkdir(parents=True)
            sparse_dir.mkdir(parents=True)
            sparse_cuda_dir.mkdir(parents=True)

            (layers_dir / "metadata.py").write_text("", encoding="utf-8")
            (layers_dir / "downsample.py").write_text("", encoding="utf-8")
            (layers_dir / "triplets.py").write_text("", encoding="utf-8")
            (layers_dir / "conv.py").write_text(
                "\n".join(
                    [
                        "class PointConv3d:",
                        "    def __init__(self, *args, **kwargs):",
                        "        self.kernel_size_3 = (3, 3, 3)",
                        "def conv_with_stride(*args, **kwargs):",
                        "    return None",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (layers_dir / "upsample.py").write_text(
                "class Upsample:\n"
                "    def __init__(self, *args, **kwargs):\n"
                "        pass\n",
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
                    "_pvd_pointelligence_layers.conv",
                    "_pvd_pointelligence_layers.upsample",
                ]
            }
            old_cache = module._POINTELLIGENCE_NN_CACHE
            original_resolve = module._resolve_pointelligence_root
            module._POINTELLIGENCE_NN_CACHE = None
            module._resolve_pointelligence_root = lambda module_file: root
            original_sys_path = list(sys.path)
            try:
                module._load_pointelligence_pvconv_api()
                sparse_cuda_ops = importlib.import_module("sparse_engines_cuda.ops")
                self.assertEqual(sparse_cuda_ops.VALUE, "real_bucket_ops")
            finally:
                module._POINTELLIGENCE_NN_CACHE = old_cache
                module._resolve_pointelligence_root = original_resolve
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
                    "_pvd_pointelligence_layers.conv",
                    "_pvd_pointelligence_layers.upsample",
                ]:
                    sys.modules.pop(name, None)
                for name, value in saved_modules.items():
                    if value is not None:
                        sys.modules[name] = value

    def test_pointelligence_pvconv_state_matches_dense_contract(self):
        module = load_pointelligence_pvconv_module()
        module._load_pointelligence_pvconv_api = lambda: (FakePointConv3d, fake_conv_with_stride, FakeUpsample)
        module.state_to_metadata = fake_state_to_metadata
        module.metadata_to_state = fake_metadata_to_state
        module.build_state_triplets = fake_build_state_triplets
        module.build_state_with_stride = fake_build_state_with_stride

        block = module.PointelligencePVConv(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            resolution=8,
            attention=False,
            dropout=None,
        )
        block.eval()

        dense_features = torch.randn(2, 8, 6)
        dense_coords = torch.randn(2, 3, 6)
        temb_sample = torch.randn(2, 4)
        dense_temb = temb_sample[:, :, None].expand(-1, -1, dense_features.shape[-1])

        dense_out, dense_coords_out, dense_temb_out = block((dense_features, dense_coords, dense_temb))

        parent = dense_to_point_cloud_state(dense_coords, grid_size=0.25).replace_features(None)
        state = dense_to_point_cloud_state(
            dense_coords,
            features=dense_features,
            grid_size=0.25,
        ).replace_query_context(
            query_coords=torch.zeros(4, 3),
            query_sample_inds=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 2], dtype=torch.long),
        ).replace_geometry(
            parent=parent,
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
        )

        state_out, temb_out = block((state, temb_sample))
        state_features_dense = point_cloud_state_to_dense(state_out.features, state_out.sample_sizes)

        self.assertTrue(torch.allclose(state_features_dense, dense_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(state_out.coords, state.coords))
        self.assertEqual(state_out.grid_size, 0.25)
        self.assertIs(state_out.parent, parent)
        self.assertTrue(torch.equal(state_out.query_points, state.query_points))
        self.assertTrue(torch.equal(state_out.i, state.i))
        self.assertTrue(torch.allclose(dense_coords_out, dense_coords))
        self.assertTrue(torch.allclose(temb_out, temb_sample))
        self.assertTrue(torch.allclose(dense_temb_out, dense_temb))

    def test_pointelligence_pvconv_same_scale_backend_reuses_one_triplet_build(self):
        module = load_pointelligence_pvconv_module()
        TrackingPointConv3d.calls = []

        def fail_conv_with_stride(*args, **kwargs):
            raise AssertionError("same-scale pointelligence backend should not call conv_with_stride")

        triplet_calls = {"count": 0}

        def counting_build_state_triplets(state, kernel_size, receptive_field_scaler=1.0, **kwargs):
            triplet_calls["count"] += 1
            return fake_build_state_triplets(
                state,
                kernel_size=kernel_size,
                receptive_field_scaler=receptive_field_scaler,
                **kwargs,
            )

        module._load_pointelligence_pvconv_api = lambda: (TrackingPointConv3d, fail_conv_with_stride, FailUpsample)
        module.state_to_metadata = fake_state_to_metadata
        module.metadata_to_state = fake_metadata_to_state
        module.build_state_triplets = counting_build_state_triplets

        block = module.PointelligencePVConv(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            resolution=8,
            attention=False,
            dropout=None,
        )
        block.eval()

        state = dense_to_point_cloud_state(
            torch.randn(2, 3, 6),
            features=torch.randn(2, 8, 6),
            grid_size=0.25,
        )

        state_out, _ = block((state, torch.randn(2, 4)))

        self.assertEqual(triplet_calls["count"], 1)
        self.assertEqual(len(TrackingPointConv3d.calls), 2)
        self.assertEqual(TrackingPointConv3d.calls[0]["i_ptr"], TrackingPointConv3d.calls[1]["i_ptr"])
        self.assertEqual(TrackingPointConv3d.calls[0]["j_ptr"], TrackingPointConv3d.calls[1]["j_ptr"])
        self.assertEqual(TrackingPointConv3d.calls[0]["k_ptr"], TrackingPointConv3d.calls[1]["k_ptr"])
        self.assertEqual(state_out.features.shape, torch.Size([12, 8]))

    def test_pointelligence_pvconv_reuses_cached_triplets_across_blocks(self):
        module = load_pointelligence_pvconv_module()

        triplet_calls = {"count": 0}

        def counting_build_state_triplets(state, kernel_size, receptive_field_scaler=1.0, **kwargs):
            triplet_calls["count"] += 1
            return fake_build_state_triplets(
                state,
                kernel_size=kernel_size,
                receptive_field_scaler=receptive_field_scaler,
                **kwargs,
            )

        module._load_pointelligence_pvconv_api = lambda: (FakePointConv3d, fake_conv_with_stride, FailUpsample)
        module.state_to_metadata = fake_state_to_metadata
        module.metadata_to_state = fake_metadata_to_state
        module.build_state_triplets = counting_build_state_triplets

        block = module.PointelligencePVConv(
            in_channels=8,
            out_channels=8,
            kernel_size=3,
            resolution=8,
            attention=False,
            dropout=None,
        )
        block.eval()

        state = dense_to_point_cloud_state(
            torch.randn(2, 3, 6),
            features=torch.randn(2, 8, 6),
            grid_size=0.25,
        )
        temb = torch.randn(2, 4)

        state_out, _ = block((state, temb))
        state_out, _ = block((state_out, temb))

        self.assertEqual(triplet_calls["count"], 1)


if __name__ == "__main__":
    unittest.main()
