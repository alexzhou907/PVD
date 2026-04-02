import importlib.util
from pathlib import Path
from types import ModuleType
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from point_cloud_state import dense_to_point_cloud_state, point_cloud_state_from_points_list, point_cloud_state_to_dense


POINTNET_PATH = Path(__file__).resolve().parents[1] / "modules" / "pointnet.py"
SHARED_MLP_PATH = Path(__file__).resolve().parents[1] / "modules" / "shared_mlp.py"
SHIMS_PATH = Path(__file__).resolve().parents[1] / "modules" / "ragged_dense_shims.py"


def load_shared_mlp_module():
    spec = importlib.util.spec_from_file_location("pvd_shared_mlp_for_pointnet_test", SHARED_MLP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_pointnet_module():
    shared_mlp_module = load_shared_mlp_module()
    shims_spec = importlib.util.spec_from_file_location("modules.ragged_dense_shims", SHIMS_PATH)
    shims_module = importlib.util.module_from_spec(shims_spec)
    assert shims_spec.loader is not None
    shims_spec.loader.exec_module(shims_module)

    for name in [
        "modules",
        "modules.functional",
        "modules.ball_query",
        "modules.ragged_dense_shims",
        "modules.shared_mlp",
        "pvd_pointnet_test",
    ]:
        sys.modules.pop(name, None)

    modules_pkg = ModuleType("modules")
    modules_pkg.__path__ = []
    functional_mod = ModuleType("modules.functional")
    ball_query_mod = ModuleType("modules.ball_query")
    shared_mlp_mod = ModuleType("modules.shared_mlp")

    def furthest_point_sample(coords, num_centers):
        return coords[:, :, :num_centers]

    def nearest_neighbor_interpolate(points_coords, centers_coords, centers_features):
        distances = (points_coords[:, :, :, None] - centers_coords[:, :, None, :]).pow(2).sum(dim=1)
        neighbor_idx = distances.argmin(dim=-1)
        gather_idx = neighbor_idx[:, None, :].expand(-1, centers_features.shape[1], -1)
        return centers_features.gather(2, gather_idx)

    class FakeBallQuery(nn.Module):
        def __init__(self, radius, num_neighbors, include_coordinates=True):
            super().__init__()
            self.num_neighbors = num_neighbors
            self.include_coordinates = include_coordinates

        def forward(self, points_coords, centers_coords, temb, points_features=None):
            num_centers = centers_coords.shape[-1]
            neighbor_count = self.num_neighbors
            neighbor_coordinates = centers_coords.unsqueeze(-1).expand(-1, -1, num_centers, neighbor_count)
            grouped_temb = temb[:, :, :num_centers].unsqueeze(-1).expand(-1, -1, num_centers, neighbor_count)
            if points_features is None:
                neighbor_features = neighbor_coordinates
            else:
                grouped_features = points_features[:, :, :num_centers].unsqueeze(-1).expand(-1, -1, num_centers, neighbor_count)
                neighbor_features = torch.cat([neighbor_coordinates, grouped_features], dim=1) if self.include_coordinates else grouped_features
            return neighbor_features, grouped_temb

    functional_mod.furthest_point_sample = furthest_point_sample
    functional_mod.nearest_neighbor_interpolate = nearest_neighbor_interpolate
    ball_query_mod.BallQuery = FakeBallQuery
    shared_mlp_mod.SharedMLP = shared_mlp_module.SharedMLP

    sys.modules["modules"] = modules_pkg
    sys.modules["modules.functional"] = functional_mod
    sys.modules["modules.ball_query"] = ball_query_mod
    sys.modules["modules.ragged_dense_shims"] = shims_module
    sys.modules["modules.shared_mlp"] = shared_mlp_mod

    spec = importlib.util.spec_from_file_location("pvd_pointnet_test", POINTNET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PointNetStateTests(unittest.TestCase):
    def test_pointnet_sa_state_matches_per_sample_dense_on_variable_lengths(self):
        module = load_pointnet_module()
        sa = module.PointNetSAModule(
            num_centers=3,
            radius=0.1,
            num_neighbors=2,
            in_channels=4,
            out_channels=[8],
            include_coordinates=True,
        )
        sa.eval()

        points_list = [
            torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
            torch.tensor([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
        ]
        features_list = [
            torch.randn(2, 4),
            torch.randn(3, 4),
        ]
        state = point_cloud_state_from_points_list(points_list, features_list=features_list, grid_size=0.25)
        temb_sample = torch.randn(2, 4)

        expected_features, expected_coords, expected_sizes = [], [], []
        for points, features, temb in zip(points_list, features_list, temb_sample):
            dense_features, dense_coords, _ = sa((
                features.transpose(0, 1).unsqueeze(0),
                points.transpose(0, 1).unsqueeze(0),
                temb.view(1, -1, 1).expand(1, -1, points.shape[0]),
            ))
            expected_features.append(dense_features.squeeze(0).transpose(0, 1))
            expected_coords.append(dense_coords.squeeze(0).transpose(0, 1))
            expected_sizes.append(dense_coords.shape[-1])

        state_out, temb_out = sa((state, temb_sample))

        self.assertEqual(state_out.sample_sizes.tolist(), expected_sizes)
        self.assertTrue(torch.allclose(state_out.features, torch.cat(expected_features, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(state_out.coords, torch.cat(expected_coords, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(temb_out, temb_sample))

    def test_pointnet_a_state_matches_per_sample_dense_on_variable_lengths(self):
        module = load_pointnet_module()
        am = module.PointNetAModule(
            in_channels=3,
            out_channels=[8],
            include_coordinates=False,
        )
        am.eval()

        points_list = [
            torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=torch.float32),
            torch.tensor([[2.0, 2.1, 2.2], [3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
        ]
        features_list = [
            torch.tensor([[-5.0, -4.0, -3.0], [-2.0, -1.0, -0.5]], dtype=torch.float32),
            torch.tensor([[-6.0, -5.0, -4.0], [-3.0, -2.0, -1.0], [-0.8, -0.7, -0.6]], dtype=torch.float32),
        ]
        state = point_cloud_state_from_points_list(points_list, features_list=features_list, grid_size=0.25)
        temb_sample = torch.randn(2, 4)

        expected_features, expected_coords = [], []
        for points, features, temb in zip(points_list, features_list, temb_sample):
            dense_features, dense_coords, _ = am((
                features.transpose(0, 1).unsqueeze(0),
                points.transpose(0, 1).unsqueeze(0),
                temb.view(1, -1, 1).expand(1, -1, points.shape[0]),
            ))
            expected_features.append(dense_features.squeeze(0).transpose(0, 1))
            expected_coords.append(dense_coords.squeeze(0).transpose(0, 1))

        state_out, temb_out = am((state, temb_sample))

        self.assertEqual(state_out.sample_sizes.tolist(), [1, 1])
        self.assertTrue(torch.allclose(state_out.features, torch.cat(expected_features, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(state_out.coords, torch.cat(expected_coords, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(temb_out, temb_sample))

    def test_pointnet_fp_state_matches_per_sample_dense_on_variable_lengths(self):
        module = load_pointnet_module()
        fp = module.PointNetFPModule(in_channels=10, out_channels=[16])
        fp.eval()

        points_list = [
            torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32),
        ]
        point_features_list = [torch.randn(2, 4), torch.randn(3, 4)]
        center_points_list = [
            torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=torch.float32),
        ]
        center_features_list = [torch.randn(1, 6), torch.randn(2, 6)]
        points_state = point_cloud_state_from_points_list(points_list, features_list=point_features_list, grid_size=0.25)
        centers_state = point_cloud_state_from_points_list(center_points_list, features_list=center_features_list, grid_size=0.25)
        temb_sample = torch.randn(2, 4)

        expected_features = []
        for points, point_features, centers, center_features, temb in zip(
            points_list,
            point_features_list,
            center_points_list,
            center_features_list,
            temb_sample,
        ):
            dense_features, _, _ = fp((
                points.transpose(0, 1).unsqueeze(0),
                centers.transpose(0, 1).unsqueeze(0),
                center_features.transpose(0, 1).unsqueeze(0),
                point_features.transpose(0, 1).unsqueeze(0),
                temb.view(1, -1, 1).expand(1, -1, centers.shape[0]),
            ))
            expected_features.append(dense_features.squeeze(0).transpose(0, 1))

        state_out, temb_out = fp((points_state, centers_state, centers_state.features, points_state.features, temb_sample))

        self.assertEqual(state_out.sample_sizes.tolist(), points_state.sample_sizes.tolist())
        self.assertTrue(torch.allclose(state_out.features, torch.cat(expected_features, dim=0), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(temb_out, temb_sample))

    def test_pointnet_sa_state_matches_dense_and_keeps_sample_level_temb(self):
        module = load_pointnet_module()
        sa = module.PointNetSAModule(
            num_centers=2,
            radius=0.1,
            num_neighbors=3,
            in_channels=8,
            out_channels=[16],
            include_coordinates=True,
        )
        sa.eval()

        dense_features = torch.randn(2, 8, 5)
        dense_coords = torch.randn(2, 3, 5)
        temb_sample = torch.randn(2, 4)
        dense_temb = temb_sample[:, :, None].expand(-1, -1, dense_features.shape[-1])

        dense_features_out, dense_coords_out, dense_temb_out = sa((dense_features, dense_coords, dense_temb))

        state = dense_to_point_cloud_state(dense_coords, features=dense_features, grid_size=0.25).replace_query_context(
            query_coords=torch.zeros(4, 3),
            query_sample_inds=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 2], dtype=torch.long),
        )
        state_out, temb_out = sa((state, temb_sample))

        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.features, state_out.sample_sizes), dense_features_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.coords, state_out.sample_sizes), dense_coords_out, atol=1e-5, rtol=1e-5))
        self.assertEqual(state_out.sample_sizes.tolist(), [2, 2])
        self.assertEqual(state_out.grid_size, 0.25)
        self.assertIsNotNone(state_out.parent)
        self.assertIsNone(state_out.parent.features)
        self.assertTrue(torch.equal(state_out.parent.coords, state.coords))
        self.assertTrue(torch.allclose(temb_out, temb_sample))
        self.assertEqual(dense_temb_out.shape, torch.Size([2, 4, 2]))

    def test_pointnet_a_state_matches_dense_and_keeps_sample_level_temb(self):
        module = load_pointnet_module()
        am = module.PointNetAModule(
            in_channels=8,
            out_channels=[16],
            include_coordinates=True,
        )
        am.eval()

        dense_features = torch.randn(2, 8, 5)
        dense_coords = torch.randn(2, 3, 5)
        temb_sample = torch.randn(2, 4)
        dense_temb = temb_sample[:, :, None].expand(-1, -1, dense_features.shape[-1])

        dense_features_out, dense_coords_out, dense_temb_out = am((dense_features, dense_coords, dense_temb))

        state = dense_to_point_cloud_state(dense_coords, features=dense_features, grid_size=0.25)
        state_out, temb_out = am((state, temb_sample))

        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.features, state_out.sample_sizes), dense_features_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.coords, state_out.sample_sizes), dense_coords_out, atol=1e-5, rtol=1e-5))
        self.assertEqual(state_out.sample_sizes.tolist(), [1, 1])
        self.assertEqual(state_out.grid_size, 0.25)
        self.assertIsNotNone(state_out.parent)
        self.assertIsNone(state_out.parent.features)
        self.assertTrue(torch.equal(state_out.parent.coords, state.coords))
        self.assertTrue(torch.allclose(temb_out, temb_sample))
        self.assertEqual(dense_temb_out.shape, torch.Size([2, 4, 1]))

    def test_pointnet_fp_state_matches_dense_and_keeps_sample_level_temb(self):
        module = load_pointnet_module()
        fp = module.PointNetFPModule(in_channels=16, out_channels=[16])
        fp.eval()

        points_coords = torch.randn(2, 3, 5)
        centers_coords = torch.randn(2, 3, 2)
        centers_features = torch.randn(2, 10, 2)
        points_features = torch.randn(2, 6, 5)
        temb_sample = torch.randn(2, 4)
        dense_temb = temb_sample[:, :, None].expand(-1, -1, centers_coords.shape[-1])

        dense_features_out, dense_points_coords, dense_temb_out = fp(
            (points_coords, centers_coords, centers_features, points_features, dense_temb)
        )

        points_state = dense_to_point_cloud_state(points_coords, features=points_features, grid_size=0.25).replace_geometry(
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
        )
        centers_state = dense_to_point_cloud_state(centers_coords, features=centers_features, grid_size=0.25)
        state_out, temb_out = fp((points_state, centers_state, centers_state.features, points_state.features, temb_sample))

        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.features, state_out.sample_sizes), dense_features_out, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(point_cloud_state_to_dense(state_out.coords, state_out.sample_sizes), dense_points_coords, atol=1e-5, rtol=1e-5))
        self.assertEqual(state_out.grid_size, 0.25)
        self.assertTrue(torch.equal(state_out.i, points_state.i))
        self.assertTrue(torch.allclose(temb_out, temb_sample))
        self.assertEqual(dense_temb_out.shape, torch.Size([2, 4, 5]))


if __name__ == "__main__":
    unittest.main()
