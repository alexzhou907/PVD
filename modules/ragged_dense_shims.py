import torch

from point_cloud_state import (
    PointCloudState,
    dense_values_to_flat as flatten_dense_values,
    expand_sample_tensor,
    offsets_to_sample_inds,
    point_cloud_state_from_points_list,
    point_cloud_state_to_dense,
    sample_sizes_to_offsets,
)


def state_coords_to_dense(state: PointCloudState):
    return point_cloud_state_to_dense(state.coords, state.sample_sizes)


def state_features_to_dense(state: PointCloudState):
    if state.features is None:
        raise ValueError("state.features must be present")
    return point_cloud_state_to_dense(state.features, state.sample_sizes)


def flat_to_dense(values, sample_sizes):
    return point_cloud_state_to_dense(values, sample_sizes)


def sample_tensor_to_dense(sample_tensor, state: PointCloudState):
    expanded = expand_sample_tensor(sample_tensor, state.sample_inds)
    return point_cloud_state_to_dense(expanded, state.sample_sizes)


def dense_values_to_flat(values, sample_sizes):
    return flatten_dense_values(values, sample_sizes)


def replace_state_features_from_dense(state: PointCloudState, dense_features):
    return state.replace_features(dense_values_to_flat(dense_features, state.sample_sizes))


def state_has_uniform_sample_sizes(state: PointCloudState) -> bool:
    if state.sample_sizes.numel() <= 1:
        return True
    first = state.sample_sizes[0]
    return bool(torch.all(state.sample_sizes == first))


def iter_sample_slices(sample_sizes):
    start = 0
    for count in sample_sizes.tolist():
        end = start + count
        yield start, end
        start = end


def split_flat_values_by_sample_sizes(values, sample_sizes):
    return [values[start:end].contiguous() for start, end in iter_sample_slices(sample_sizes)]


def split_state_to_sample_lists(state: PointCloudState):
    coords_list = split_flat_values_by_sample_sizes(state.coords, state.sample_sizes)
    features_list = None if state.features is None else split_flat_values_by_sample_sizes(state.features, state.sample_sizes)
    return coords_list, features_list


def sample_points_to_dense(points):
    return points.transpose(0, 1).unsqueeze(0).contiguous()


def sample_features_to_dense(features):
    return features.transpose(0, 1).unsqueeze(0).contiguous()


def sample_temporal_embedding_to_dense(sample_temporal_embedding, num_points):
    return sample_temporal_embedding.view(1, -1, 1).expand(1, -1, num_points)


def state_from_sample_lists(template_state: PointCloudState, coords_list, features_list, *, parent=None):
    return point_cloud_state_from_points_list(
        coords_list,
        features_list=features_list,
        grid_size=template_state.grid_size,
        parent=template_state.geometry_only() if parent is None else parent,
    )


def dense_to_state_like(template_state: PointCloudState, dense_coords, dense_features):
    batch_size, _, num_points = dense_coords.shape
    sample_sizes = torch.full(
        (batch_size,),
        num_points,
        dtype=template_state.sample_sizes.dtype,
        device=dense_coords.device,
    )
    sample_inds = offsets_to_sample_inds(sample_sizes_to_offsets(sample_sizes))
    flat_coords = flatten_dense_values(dense_coords, sample_sizes)
    flat_features = flatten_dense_values(dense_features, sample_sizes)

    return template_state.replace_geometry(
        coords=flat_coords,
        features=flat_features,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        parent=template_state.geometry_only(),
        query_coords=None,
        query_sample_inds=None,
        query_sample_sizes=None,
        i=None,
        j=None,
        k=None,
        num_neighbors=None,
        downsample_indices=None,
        i_upsample=None,
        j_upsample=None,
        k_upsample=None,
        kernel_size=None,
        receptive_field_scaler=1.0,
        sort_by="k",
        return_num_neighbors=False,
        distance_type="ball",
    )


def dense_features_from_state_or_flat(values, sample_sizes):
    if isinstance(values, PointCloudState):
        return state_features_to_dense(values)
    return flat_to_dense(values, sample_sizes)
