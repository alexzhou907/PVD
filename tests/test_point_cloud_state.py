import unittest
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "modules"))

from point_cloud_state import (
    build_state_triplets,
    build_state_with_stride,
    PointCloudState,
    dense_to_point_cloud_state,
    metadata_to_state,
    expand_sample_tensor,
    point_cloud_state_to_dense,
    point_cloud_state_from_points_list,
    sample_sizes_to_offsets,
    state_to_metadata,
)


class PointCloudStateTests(unittest.TestCase):
    def test_dense_coords_round_trip_preserves_point_order(self):
        coords = torch.arange(24, dtype=torch.float32).view(2, 3, 4)

        state = dense_to_point_cloud_state(coords)
        dense_coords = point_cloud_state_to_dense(state.coords, state.sample_sizes)

        self.assertTrue(torch.equal(dense_coords, coords))
        self.assertTrue(torch.equal(state.points, state.coords))

    def test_dense_features_round_trip_preserves_channel_layout(self):
        coords = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        features = torch.arange(40, dtype=torch.float32).view(2, 5, 4)

        state = dense_to_point_cloud_state(coords, features=features, grid_size=0.125)
        dense_features = point_cloud_state_to_dense(state.features, state.sample_sizes)

        self.assertTrue(torch.equal(dense_features, features))
        self.assertEqual(state.grid_size, 0.125)

    def test_state_exposes_metadata_compatible_index_fields(self):
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4))

        self.assertIsInstance(state, PointCloudState)
        self.assertEqual(state.sample_inds.shape, torch.Size([8]))
        self.assertTrue(torch.equal(state.sample_sizes, torch.tensor([4, 4])))
        self.assertTrue(torch.equal(state.offsets, torch.tensor([0, 4, 8])))

    def test_expand_sample_tensor_broadcasts_to_points(self):
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4))
        sample_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        expanded = expand_sample_tensor(sample_values, state.sample_inds)

        self.assertEqual(expanded.shape, torch.Size([8, 2]))
        self.assertTrue(torch.equal(expanded[:4], sample_values[:1].expand(4, -1)))
        self.assertTrue(torch.equal(expanded[4:], sample_values[1:].expand(4, -1)))

    def test_sample_sizes_to_offsets_matches_state_offsets(self):
        state = dense_to_point_cloud_state(torch.randn(3, 3, 2))

        offsets = sample_sizes_to_offsets(state.sample_sizes)

        self.assertTrue(torch.equal(offsets, state.offsets))

    def test_state_to_metadata_round_trip_preserves_geometry_context(self):
        parent = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_features(None)
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4), features=torch.randn(2, 5, 4), grid_size=0.25)
        state = state.replace_query_context(
            query_coords=state.coords[:4].clone(),
            query_sample_inds=state.sample_inds[:4].clone(),
            query_sample_sizes=torch.tensor([4], dtype=torch.long),
        ).replace_geometry(
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([3, 4], dtype=torch.long),
            num_neighbors=torch.tensor([1, 1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2, 4, 6], dtype=torch.long),
            i_upsample=torch.tensor([0, 1], dtype=torch.long),
            j_upsample=torch.tensor([1, 0], dtype=torch.long),
            k_upsample=torch.tensor([4, 3], dtype=torch.long),
            parent=parent,
        )

        metadata = state_to_metadata(state)
        round_trip = metadata_to_state(metadata, features=state.features)

        self.assertEqual(round_trip.grid_size, state.grid_size)
        self.assertTrue(torch.equal(round_trip.coords, state.coords))
        self.assertTrue(torch.equal(round_trip.features, state.features))
        self.assertTrue(torch.equal(round_trip.query_points, state.query_points))
        self.assertTrue(torch.equal(round_trip.query_sample_inds, state.query_sample_inds))
        self.assertTrue(torch.equal(round_trip.query_sample_sizes, state.query_sample_sizes))
        self.assertTrue(torch.equal(round_trip.i, state.i))
        self.assertTrue(torch.equal(round_trip.j, state.j))
        self.assertTrue(torch.equal(round_trip.k, state.k))
        self.assertTrue(torch.equal(round_trip.downsample_indices, state.downsample_indices))
        self.assertTrue(torch.equal(round_trip.i_upsample, state.i_upsample))
        self.assertTrue(torch.equal(round_trip.j_upsample, state.j_upsample))
        self.assertTrue(torch.equal(round_trip.k_upsample, state.k_upsample))
        self.assertIsNotNone(round_trip.parent)
        self.assertIsNone(round_trip.parent.features)
        self.assertTrue(torch.equal(round_trip.parent.coords, parent.coords))

    def test_dirty_triplets_returns_triplet_free_copy(self):
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_geometry(
            i=torch.tensor([0], dtype=torch.long),
            j=torch.tensor([1], dtype=torch.long),
            k=torch.tensor([2], dtype=torch.long),
            downsample_indices=torch.tensor([0], dtype=torch.long),
            i_upsample=torch.tensor([0], dtype=torch.long),
            j_upsample=torch.tensor([1], dtype=torch.long),
            k_upsample=torch.tensor([2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        dirty = state.dirty_triplets()

        self.assertFalse(state.empty_triplets())
        self.assertTrue(dirty.empty_triplets())
        self.assertIsNone(dirty.i)
        self.assertIsNone(dirty.j)
        self.assertIsNone(dirty.k)
        self.assertIsNone(dirty.downsample_indices)
        self.assertIsNone(dirty.i_upsample)
        self.assertIsNone(dirty.j_upsample)
        self.assertIsNone(dirty.k_upsample)
        self.assertIsNone(dirty.kernel_size)
        self.assertEqual(dirty.receptive_field_scaler, 1.0)
        self.assertEqual(dirty.sort_by, "k")
        self.assertFalse(dirty.return_num_neighbors)
        self.assertEqual(dirty.distance_type, "ball")
        self.assertEqual(dirty.num_points(), state.coords.shape[0])

    def test_replace_coords_and_features_invalidates_geometry_context(self):
        parent = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.125).replace_features(None)
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_query_context(
            query_coords=torch.randn(5, 3),
            query_sample_inds=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([3, 2], dtype=torch.long),
        ).replace_geometry(
            parent=parent,
            i=torch.tensor([0], dtype=torch.long),
            j=torch.tensor([1], dtype=torch.long),
            k=torch.tensor([2], dtype=torch.long),
            num_neighbors=torch.tensor([1], dtype=torch.long),
            downsample_indices=torch.tensor([0], dtype=torch.long),
            i_upsample=torch.tensor([0], dtype=torch.long),
            j_upsample=torch.tensor([1], dtype=torch.long),
            k_upsample=torch.tensor([2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        replaced = state.replace_coords_and_features(torch.randn_like(state.coords))

        self.assertIsNone(replaced.parent)
        self.assertIsNone(replaced.query_coords)
        self.assertIsNone(replaced.query_sample_inds)
        self.assertIsNone(replaced.query_sample_sizes)
        self.assertIsNone(replaced.i)
        self.assertIsNone(replaced.j)
        self.assertIsNone(replaced.k)
        self.assertIsNone(replaced.num_neighbors)
        self.assertIsNone(replaced.downsample_indices)
        self.assertIsNone(replaced.i_upsample)
        self.assertIsNone(replaced.j_upsample)
        self.assertIsNone(replaced.k_upsample)
        self.assertIsNone(replaced.kernel_size)
        self.assertEqual(replaced.receptive_field_scaler, 1.0)
        self.assertEqual(replaced.sort_by, "k")
        self.assertFalse(replaced.return_num_neighbors)
        self.assertEqual(replaced.distance_type, "ball")

    def test_replace_query_context_invalidates_triplet_cache(self):
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_geometry(
            i=torch.tensor([0], dtype=torch.long),
            j=torch.tensor([1], dtype=torch.long),
            k=torch.tensor([2], dtype=torch.long),
            num_neighbors=torch.tensor([1], dtype=torch.long),
            downsample_indices=torch.tensor([0], dtype=torch.long),
            i_upsample=torch.tensor([0], dtype=torch.long),
            j_upsample=torch.tensor([1], dtype=torch.long),
            k_upsample=torch.tensor([2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        replaced = state.replace_query_context(
            query_coords=torch.randn(3, 3),
            query_sample_inds=torch.tensor([0, 0, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 1], dtype=torch.long),
        )

        self.assertIsNotNone(replaced.query_coords)
        self.assertTrue(replaced.empty_triplets())
        self.assertIsNone(replaced.downsample_indices)
        self.assertIsNone(replaced.i_upsample)
        self.assertIsNone(replaced.j_upsample)
        self.assertIsNone(replaced.k_upsample)
        self.assertIsNone(replaced.kernel_size)
        self.assertEqual(replaced.receptive_field_scaler, 1.0)
        self.assertEqual(replaced.sort_by, "k")
        self.assertFalse(replaced.return_num_neighbors)
        self.assertEqual(replaced.distance_type, "ball")

    def test_geometry_only_parent_is_allowed(self):
        parent = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.125).replace_features(None)
        child = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.125).replace_geometry(parent=parent)

        self.assertIsNone(child.parent.features)
        self.assertTrue(torch.equal(child.parent.points, parent.points))
        self.assertEqual(child.parent.num_points(), parent.coords.shape[0])

    def test_point_cloud_state_from_points_list_supports_variable_lengths(self):
        points_list = [
            torch.tensor([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=torch.float32),
            torch.tensor([[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]], dtype=torch.float32),
        ]
        features_list = [
            torch.tensor([[10.0, 10.1], [11.0, 11.1], [12.0, 12.1]], dtype=torch.float32),
            torch.tensor([[13.0, 13.1], [14.0, 14.1]], dtype=torch.float32),
        ]

        state = point_cloud_state_from_points_list(points_list, features_list=features_list, grid_size=0.5)

        self.assertEqual(state.sample_sizes.tolist(), [3, 2])
        self.assertTrue(torch.equal(state.sample_inds, torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)))
        self.assertEqual(state.grid_size, 0.5)
        self.assertTrue(torch.equal(state.coords[:3], points_list[0]))
        self.assertTrue(torch.equal(state.coords[3:], points_list[1]))
        self.assertTrue(torch.equal(state.features[:3], features_list[0]))
        self.assertTrue(torch.equal(state.features[3:], features_list[1]))

    def test_state_rejects_sample_inds_that_do_not_match_sample_sizes(self):
        with self.assertRaises(ValueError):
            PointCloudState(
                coords=torch.randn(4, 3),
                features=torch.randn(4, 2),
                sample_inds=torch.tensor([0, 1, 0, 1], dtype=torch.long),
                sample_sizes=torch.tensor([2, 2], dtype=torch.long),
                grid_size=0.25,
            )

    def test_state_rejects_query_sample_inds_that_do_not_match_query_sample_sizes(self):
        with self.assertRaises(ValueError):
            dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_query_context(
                query_coords=torch.randn(3, 3),
                query_sample_inds=torch.tensor([0, 1, 0], dtype=torch.long),
                query_sample_sizes=torch.tensor([2, 1], dtype=torch.long),
            )

    def test_replace_coords_and_features_invalidates_geometry_metadata(self):
        parent = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.125).replace_features(None)
        state = dense_to_point_cloud_state(
            torch.randn(2, 3, 4),
            features=torch.randn(2, 5, 4),
            grid_size=0.25,
        ).replace_query_context(
            query_coords=torch.randn(4, 3),
            query_sample_inds=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 2], dtype=torch.long),
        ).replace_geometry(
            parent=parent,
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
            num_neighbors=torch.tensor([1, 1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2], dtype=torch.long),
            i_upsample=torch.tensor([0, 1], dtype=torch.long),
            j_upsample=torch.tensor([1, 0], dtype=torch.long),
            k_upsample=torch.tensor([3, 2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        replaced = state.replace_coords_and_features(torch.randn_like(state.coords))

        self.assertIsNone(replaced.parent)
        self.assertIsNone(replaced.query_coords)
        self.assertIsNone(replaced.query_sample_inds)
        self.assertIsNone(replaced.query_sample_sizes)
        self.assertIsNone(replaced.i)
        self.assertIsNone(replaced.j)
        self.assertIsNone(replaced.k)
        self.assertIsNone(replaced.num_neighbors)
        self.assertIsNone(replaced.downsample_indices)
        self.assertIsNone(replaced.i_upsample)
        self.assertIsNone(replaced.j_upsample)
        self.assertIsNone(replaced.k_upsample)
        self.assertIsNone(replaced.kernel_size)
        self.assertEqual(replaced.receptive_field_scaler, 1.0)
        self.assertEqual(replaced.sort_by, "k")
        self.assertFalse(replaced.return_num_neighbors)
        self.assertEqual(replaced.distance_type, "ball")

    def test_replace_query_context_invalidates_triplets_and_kernel_metadata(self):
        state = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_geometry(
            i=torch.tensor([0], dtype=torch.long),
            j=torch.tensor([1], dtype=torch.long),
            k=torch.tensor([2], dtype=torch.long),
            num_neighbors=torch.tensor([1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2], dtype=torch.long),
            kernel_size=(3, 3, 3),
            receptive_field_scaler=2.0,
            sort_by="i",
            return_num_neighbors=True,
            distance_type="cube",
        )

        replaced = state.replace_query_context(
            query_coords=torch.randn(4, 3),
            query_sample_inds=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 2], dtype=torch.long),
        )

        self.assertTrue(replaced.empty_triplets())
        self.assertIsNone(replaced.downsample_indices)
        self.assertIsNone(replaced.kernel_size)
        self.assertEqual(replaced.receptive_field_scaler, 1.0)
        self.assertEqual(replaced.sort_by, "k")
        self.assertFalse(replaced.return_num_neighbors)
        self.assertEqual(replaced.distance_type, "ball")

    def test_state_to_moves_geometry_context_and_parent(self):
        parent = dense_to_point_cloud_state(torch.randn(2, 3, 4), grid_size=0.25).replace_features(None)
        state = dense_to_point_cloud_state(
            torch.randn(2, 3, 4),
            features=torch.randn(2, 5, 4),
            grid_size=0.25,
        ).replace_query_context(
            query_coords=torch.randn(3, 3),
            query_sample_inds=torch.tensor([0, 0, 1], dtype=torch.long),
            query_sample_sizes=torch.tensor([2, 1], dtype=torch.long),
        ).replace_geometry(
            i=torch.tensor([0, 1], dtype=torch.long),
            j=torch.tensor([1, 0], dtype=torch.long),
            k=torch.tensor([2, 3], dtype=torch.long),
            num_neighbors=torch.tensor([1, 1], dtype=torch.long),
            downsample_indices=torch.tensor([0, 2], dtype=torch.long),
            i_upsample=torch.tensor([0, 1], dtype=torch.long),
            j_upsample=torch.tensor([1, 0], dtype=torch.long),
            k_upsample=torch.tensor([3, 2], dtype=torch.long),
            parent=parent,
        )

        moved = state.to(device=torch.device("cpu"), dtype=torch.float64)

        self.assertEqual(moved.coords.dtype, torch.float64)
        self.assertEqual(moved.features.dtype, torch.float64)
        self.assertEqual(moved.query_coords.dtype, torch.float64)
        self.assertEqual(moved.sample_inds.dtype, torch.long)
        self.assertEqual(moved.i.dtype, torch.long)
        self.assertIsNotNone(moved.parent)
        self.assertIsNone(moved.parent.features)
        self.assertEqual(moved.parent.coords.dtype, torch.float64)
        self.assertTrue(torch.equal(moved.sample_sizes, state.sample_sizes))
        self.assertEqual(moved.grid_size, state.grid_size)


if __name__ == "__main__":
    unittest.main()
