import torch
import torch.nn as nn

import modules.functional as F
from modules.ball_query import BallQuery
from modules.ragged_dense_shims import (
    dense_features_from_state_or_flat,
    dense_to_state_like,
    replace_state_features_from_dense,
    sample_tensor_to_dense,
    sample_features_to_dense,
    sample_points_to_dense,
    sample_temporal_embedding_to_dense,
    split_flat_values_by_sample_sizes,
    split_state_to_sample_lists,
    state_from_sample_lists,
    state_has_uniform_sample_sizes,
    state_coords_to_dense,
    state_features_to_dense,
)
from modules.shared_mlp import SharedMLP
from point_cloud_state import PointCloudState

__all__ = ['PointNetAModule', 'PointNetSAModule', 'PointNetFPModule']


class PointNetAModule(nn.Module):
    def __init__(self, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]]
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels]

        mlps = []
        total_out_channels = 0
        for _out_channels in out_channels:
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0),
                          out_channels=_out_channels, dim=1)
            )
            total_out_channels += _out_channels[-1]

        self.include_coordinates = include_coordinates
        self.out_channels = total_out_channels
        self.mlps = nn.ModuleList(mlps)

    def _forward_dense(self, features, coords, temb=None):
        if self.include_coordinates:
            features = torch.cat([features, coords], dim=1)
        coords_out = torch.zeros((coords.size(0), 3, 1), device=coords.device)
        if len(self.mlps) > 1:
            features_list = []
            for mlp in self.mlps:
                features_list.append(mlp(features).max(dim=-1, keepdim=True).values)
            features_out = torch.cat(features_list, dim=1)
        else:
            features_out = self.mlps[0](features).max(dim=-1, keepdim=True).values
        if temb is None:
            return features_out, coords_out
        temb_out = temb.max(dim=-1, keepdim=True).values if temb.shape[1] > 0 else temb
        return features_out, coords_out, temb_out

    def forward(self, inputs):
        if len(inputs) == 2 and isinstance(inputs[0], PointCloudState):
            state, temb_sample = inputs
            if not state_has_uniform_sample_sizes(state):
                coords_list, features_list = split_state_to_sample_lists(state)
                features_out_list, coords_out_list = [], []
                for coords_i, features_i, temb_i in zip(coords_list, features_list, temb_sample):
                    dense_features_out, dense_coords_out, _ = self._forward_dense(
                        sample_features_to_dense(features_i),
                        sample_points_to_dense(coords_i),
                        sample_temporal_embedding_to_dense(temb_i, coords_i.shape[0]),
                    )
                    features_out_list.append(dense_features_out.squeeze(0).transpose(0, 1).contiguous())
                    coords_out_list.append(dense_coords_out.squeeze(0).transpose(0, 1).contiguous())
                return state_from_sample_lists(state, coords_out_list, features_out_list), temb_sample
            dense_features = state_features_to_dense(state)
            dense_coords = state_coords_to_dense(state)
            dense_temb = sample_tensor_to_dense(temb_sample, state)
            dense_features_out, dense_coords_out, _ = self._forward_dense(dense_features, dense_coords, dense_temb)
            return dense_to_state_like(state, dense_coords_out, dense_features_out), temb_sample
        if len(inputs) == 3:
            features, coords, temb = inputs
            return self._forward_dense(features, coords, temb)
        features, coords = inputs
        return self._forward_dense(features, coords)

    def extra_repr(self):
        return f'out_channels={self.out_channels}, include_coordinates={self.include_coordinates}'


class PointNetSAModule(nn.Module):
    def __init__(self, num_centers, radius, num_neighbors, in_channels, out_channels, include_coordinates=True):
        super().__init__()
        if not isinstance(radius, (list, tuple)):
            radius = [radius]
        if not isinstance(num_neighbors, (list, tuple)):
            num_neighbors = [num_neighbors] * len(radius)
        assert len(radius) == len(num_neighbors)
        if not isinstance(out_channels, (list, tuple)):
            out_channels = [[out_channels]] * len(radius)
        elif not isinstance(out_channels[0], (list, tuple)):
            out_channels = [out_channels] * len(radius)
        assert len(radius) == len(out_channels)

        groupers, mlps = [], []
        total_out_channels = 0
        for _radius, _out_channels, _num_neighbors in zip(radius, out_channels, num_neighbors):
            groupers.append(
                BallQuery(radius=_radius, num_neighbors=_num_neighbors, include_coordinates=include_coordinates)
            )
            mlps.append(
                SharedMLP(in_channels=in_channels + (3 if include_coordinates else 0),
                          out_channels=_out_channels, dim=2)
            )
            total_out_channels += _out_channels[-1]

        self.num_centers = num_centers
        self.out_channels = total_out_channels
        self.groupers = nn.ModuleList(groupers)
        self.mlps = nn.ModuleList(mlps)

    def _forward_dense(self, features, coords, temb):
        centers_coords = F.furthest_point_sample(coords, self.num_centers)
        features_list = []
        temb_out = temb
        for grouper, mlp in zip(self.groupers, self.mlps):
            grouped_features, grouped_temb = mlp(grouper(coords, centers_coords, temb, features))
            features_list.append(grouped_features.max(dim=-1).values)
            temb_out = grouped_temb.max(dim=-1).values if grouped_temb.shape[1] > 0 else grouped_temb
        return features_list[0], centers_coords, temb_out

    def forward(self, inputs):
        if len(inputs) == 2 and isinstance(inputs[0], PointCloudState):
            state, temb_sample = inputs
            if not state_has_uniform_sample_sizes(state):
                coords_list, features_list = split_state_to_sample_lists(state)
                features_out_list, coords_out_list = [], []
                for coords_i, features_i, temb_i in zip(coords_list, features_list, temb_sample):
                    dense_features_out, dense_coords_out, _ = self._forward_dense(
                        sample_features_to_dense(features_i),
                        sample_points_to_dense(coords_i),
                        sample_temporal_embedding_to_dense(temb_i, coords_i.shape[0]),
                    )
                    features_out_list.append(dense_features_out.squeeze(0).transpose(0, 1).contiguous())
                    coords_out_list.append(dense_coords_out.squeeze(0).transpose(0, 1).contiguous())
                return state_from_sample_lists(state, coords_out_list, features_out_list), temb_sample
            dense_features = state_features_to_dense(state)
            dense_coords = state_coords_to_dense(state)
            dense_temb = sample_tensor_to_dense(temb_sample, state)
            dense_features_out, dense_coords_out, _ = self._forward_dense(dense_features, dense_coords, dense_temb)
            return dense_to_state_like(state, dense_coords_out, dense_features_out), temb_sample
        features, coords, temb = inputs
        return self._forward_dense(features, coords, temb)

    def extra_repr(self):
        return f'num_centers={self.num_centers}, out_channels={self.out_channels}'


class PointNetFPModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mlp = SharedMLP(in_channels=in_channels, out_channels=out_channels, dim=1)

    def _forward_dense(self, points_coords, centers_coords, centers_features, points_features, temb):
        interpolated_features = F.nearest_neighbor_interpolate(points_coords, centers_coords, centers_features)
        interpolated_temb = F.nearest_neighbor_interpolate(points_coords, centers_coords, temb)
        if points_features is not None:
            interpolated_features = torch.cat(
                [interpolated_features, points_features], dim=1
            )
        return self.mlp(interpolated_features), points_coords, interpolated_temb

    def forward(self, inputs):
        if len(inputs) >= 4 and isinstance(inputs[0], PointCloudState):
            if len(inputs) == 4:
                points_state, centers_state, centers_features, temb_sample = inputs
                points_features = None
            else:
                points_state, centers_state, centers_features, points_features, temb_sample = inputs
            if not state_has_uniform_sample_sizes(points_state):
                points_coords_list, _ = split_state_to_sample_lists(points_state)
                centers_coords_list, _ = split_state_to_sample_lists(centers_state)
                centers_features_list = split_flat_values_by_sample_sizes(
                    centers_features.features if isinstance(centers_features, PointCloudState) else centers_features,
                    centers_features.sample_sizes if isinstance(centers_features, PointCloudState) else centers_state.sample_sizes,
                )
                points_features_list = None
                if points_features is not None:
                    points_features_list = split_flat_values_by_sample_sizes(
                        points_features.features if isinstance(points_features, PointCloudState) else points_features,
                        points_features.sample_sizes if isinstance(points_features, PointCloudState) else points_state.sample_sizes,
                    )

                features_out_list = []
                for sample_idx, (points_coords_i, centers_coords_i, centers_features_i, temb_i) in enumerate(
                    zip(points_coords_list, centers_coords_list, centers_features_list, temb_sample)
                ):
                    points_features_i = None if points_features_list is None else points_features_list[sample_idx]
                    dense_features_out, _, _ = self._forward_dense(
                        sample_points_to_dense(points_coords_i),
                        sample_points_to_dense(centers_coords_i),
                        sample_features_to_dense(centers_features_i),
                        None if points_features_i is None else sample_features_to_dense(points_features_i),
                        sample_temporal_embedding_to_dense(temb_i, centers_coords_i.shape[0]),
                    )
                    features_out_list.append(dense_features_out.squeeze(0).transpose(0, 1).contiguous())
                return points_state.replace_features(torch.cat(features_out_list, dim=0)), temb_sample
            dense_points_coords = state_coords_to_dense(points_state)
            dense_centers_coords = state_coords_to_dense(centers_state)
            dense_centers_features = dense_features_from_state_or_flat(centers_features, centers_state.sample_sizes)
            dense_points_features = None if points_features is None else dense_features_from_state_or_flat(points_features, points_state.sample_sizes)
            dense_temb = sample_tensor_to_dense(temb_sample, centers_state)
            dense_features_out, _, _ = self._forward_dense(
                dense_points_coords,
                dense_centers_coords,
                dense_centers_features,
                dense_points_features,
                dense_temb,
            )
            return replace_state_features_from_dense(points_state, dense_features_out), temb_sample
        if len(inputs) == 4:
            points_coords, centers_coords, centers_features, temb = inputs
            points_features = None
        else:
            points_coords, centers_coords, centers_features, points_features, temb = inputs
        return self._forward_dense(points_coords, centers_coords, centers_features, points_features, temb)
