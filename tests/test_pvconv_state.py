import importlib.util
from pathlib import Path
from types import ModuleType
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from point_cloud_state import dense_to_point_cloud_state, point_cloud_state_from_points_list, point_cloud_state_to_dense


PVCONV_PATH = Path(__file__).resolve().parents[1] / "modules" / "pvconv.py"
SHARED_MLP_PATH = Path(__file__).resolve().parents[1] / "modules" / "shared_mlp.py"
SHIMS_PATH = Path(__file__).resolve().parents[1] / "modules" / "ragged_dense_shims.py"


def load_shared_mlp_module():
    spec = importlib.util.spec_from_file_location("pvd_shared_mlp_for_pvconv_test", SHARED_MLP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pvconv_module():
    shared_mlp_module = load_shared_mlp_module()
    shims_spec = importlib.util.spec_from_file_location("modules.ragged_dense_shims", SHIMS_PATH)
    shims_module = importlib.util.module_from_spec(shims_spec)
    assert shims_spec.loader is not None
    shims_spec.loader.exec_module(shims_module)

    for name in [
        "modules",
        "modules.functional",
        "modules.ragged_dense_shims",
        "modules.voxelization",
        "modules.shared_mlp",
        "modules.se",
        "pvd_pvconv_test",
    ]:
        sys.modules.pop(name, None)

    modules_pkg = ModuleType("modules")
    modules_pkg.__path__ = []
    functional_mod = ModuleType("modules.functional")
    voxelization_mod = ModuleType("modules.voxelization")
    shared_mlp_mod = ModuleType("modules.shared_mlp")
    se_mod = ModuleType("modules.se")

    def trilinear_devoxelize(voxel_features, voxel_coords, resolution, training):
        return voxel_features.squeeze(-1).squeeze(-1)

    class FakeVoxelization(nn.Module):
        def __init__(self, resolution, normalize=True, eps=0):
            super().__init__()
            self.resolution = resolution

        def forward(self, features, coords):
            coord_bias = coords.mean(dim=-1, keepdim=True).sum(dim=1, keepdim=True)
            return (features + coord_bias).unsqueeze(-1).unsqueeze(-1), coords

    class IdentitySE3d(nn.Module):
        def __init__(self, channels, use_relu=False):
            super().__init__()

        def forward(self, x):
            return x

    functional_mod.trilinear_devoxelize = trilinear_devoxelize
    voxelization_mod.Voxelization = FakeVoxelization
    shared_mlp_mod.SharedMLP = shared_mlp_module.SharedMLP
    se_mod.SE3d = IdentitySE3d

    sys.modules["modules"] = modules_pkg
    sys.modules["modules.functional"] = functional_mod
    sys.modules["modules.ragged_dense_shims"] = shims_module
    sys.modules["modules.voxelization"] = voxelization_mod
    sys.modules["modules.shared_mlp"] = shared_mlp_mod
    sys.modules["modules.se"] = se_mod

    spec = importlib.util.spec_from_file_location("pvd_pvconv_test", PVCONV_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PVConvStateTests(unittest.TestCase):
    def test_pvconv_state_matches_per_sample_dense_on_variable_lengths(self):
        module = load_pvconv_module()
        block = module.PVConv(
            in_channels=4,
            out_channels=8,
            kernel_size=1,
            resolution=4,
            attention=False,
            dropout=None,
        )
        block.eval()

        points_list = [
            torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
            torch.tensor([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
        ]
        features_list = [torch.randn(2, 4), torch.randn(3, 4)]
        state = point_cloud_state_from_points_list(points_list, features_list=features_list, grid_size=0.25)
        temb_sample = torch.randn(2, 4)

        expected_features = []
        for points, features, temb in zip(points_list, features_list, temb_sample):
            dense_out, _, _ = block((
                features.transpose(0, 1).unsqueeze(0),
                points.transpose(0, 1).unsqueeze(0),
                temb.view(1, -1, 1).expand(1, -1, points.shape[0]),
            ))
            expected_features.append(dense_out.squeeze(0).transpose(0, 1))

        state_out, temb_out = block((state, temb_sample))

        self.assertEqual(state_out.sample_sizes.tolist(), state.sample_sizes.tolist())
        self.assertTrue(torch.allclose(state_out.features, torch.cat(expected_features, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(state_out.coords, state.coords))
        self.assertTrue(torch.allclose(temb_out, temb_sample))

    def test_pvconv_state_matches_dense_on_fixed_n(self):
        module = load_pvconv_module()
        block = module.PVConv(
            in_channels=8,
            out_channels=8,
            kernel_size=1,
            resolution=4,
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


if __name__ == "__main__":
    unittest.main()
