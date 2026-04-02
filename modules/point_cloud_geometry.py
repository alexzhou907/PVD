from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Optional

import torch

from point_cloud_state import PointCloudState


_POINTELLIGENCE_CACHE = None


def _fallback_indexed_distance(queries, q_inds, points, p_inds, distance_type="ball"):
    q_inds = q_inds.to(dtype=torch.long, device=queries.device)
    p_inds = p_inds.to(dtype=torch.long, device=points.device)
    deltas = points.index_select(0, p_inds) - queries.index_select(0, q_inds)
    if distance_type == "cube":
        return deltas.abs().amax(dim=1)
    return deltas.square().sum(dim=1).sqrt()


def _wrap_cpu_safe_indexed_distance(indexed_distance_mod):
    indexed_distance = getattr(indexed_distance_mod, "indexed_distance", None)
    if indexed_distance is None or getattr(indexed_distance, "_pvd_cpu_safe", False):
        return

    def cpu_safe_indexed_distance(queries, q_inds, points, p_inds, distance_type="ball"):
        if queries.device.type == "cpu" or points.device.type == "cpu":
            return _fallback_indexed_distance(queries, q_inds, points, p_inds, distance_type=distance_type)
        return indexed_distance(queries, q_inds, points, p_inds, distance_type=distance_type)

    cpu_safe_indexed_distance._pvd_cpu_safe = True
    indexed_distance_mod.indexed_distance = cpu_safe_indexed_distance


def _fallback_bucket_arrange(bucket_indices, num_buckets):
    bucket_indices = bucket_indices.to(dtype=torch.long)
    bucket_sizes = torch.bincount(bucket_indices, minlength=num_buckets)
    bucket_slots = torch.empty_like(bucket_indices)
    next_slot = torch.zeros(num_buckets, dtype=torch.long, device=bucket_indices.device)
    for idx, bucket in enumerate(bucket_indices.tolist()):
        bucket_slots[idx] = next_slot[bucket]
        next_slot[bucket] += 1
    return bucket_sizes.to(bucket_indices.dtype), bucket_slots


def _wrap_cpu_safe_bucket_arrange(sparse_cuda_ops):
    bucket_arrange = getattr(sparse_cuda_ops, "bucket_arrange", None)
    if bucket_arrange is None or getattr(bucket_arrange, "_pvd_cpu_safe", False):
        return

    def cpu_safe_bucket_arrange(bucket_indices, num_buckets):
        if bucket_indices.device.type == "cpu":
            return _fallback_bucket_arrange(bucket_indices, num_buckets)
        return bucket_arrange(bucket_indices, num_buckets)

    cpu_safe_bucket_arrange._pvd_cpu_safe = True
    sparse_cuda_ops.bucket_arrange = cpu_safe_bucket_arrange


def _resolve_pointelligence_root(module_file: Path) -> Path:
    candidates = [
        module_file.parents[2],
        module_file.parents[2] / "pointelligence",
    ]
    for root in candidates:
        if (root / "layers" / "metadata.py").exists():
            return root
    raise ModuleNotFoundError(
        f"pointelligence layers directory not found from {module_file}; "
        f"checked {[str(root / 'layers') for root in candidates]}"
    )


def _ensure_pointelligence_import_paths(root: Path) -> None:
    for path in (root / "extensions", root):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def _install_sparse_engine_shim():
    try:
        sparse_pkg = importlib.import_module("sparse_engines")
    except ModuleNotFoundError:
        sparse_pkg = types.ModuleType("sparse_engines")
        sparse_pkg.__path__ = []
        sys.modules["sparse_engines"] = sparse_pkg
    try:
        indexed_distance_mod = importlib.import_module("sparse_engines.indexed_distance_triton")
    except ModuleNotFoundError:
        indexed_distance_mod = types.ModuleType("sparse_engines.indexed_distance_triton")
        indexed_distance_mod.indexed_distance = _fallback_indexed_distance
        sys.modules["sparse_engines.indexed_distance_triton"] = indexed_distance_mod
    sparse_pkg.indexed_distance_triton = indexed_distance_mod
    _wrap_cpu_safe_indexed_distance(indexed_distance_mod)

    try:
        sparse_cuda_pkg = importlib.import_module("sparse_engines_cuda")
    except Exception:
        sys.modules.pop("sparse_engines_cuda", None)
        sys.modules.pop("sparse_engines_cuda.ops", None)
        sparse_cuda_pkg = types.ModuleType("sparse_engines_cuda")
        sparse_cuda_pkg.__path__ = []
        sys.modules["sparse_engines_cuda"] = sparse_cuda_pkg
    try:
        sparse_cuda_ops = importlib.import_module("sparse_engines_cuda.ops")
    except Exception:
        sys.modules.pop("sparse_engines_cuda.ops", None)
        sparse_cuda_ops = types.ModuleType("sparse_engines_cuda.ops")
        sparse_cuda_ops.bucket_arrange = _fallback_bucket_arrange
        sys.modules["sparse_engines_cuda.ops"] = sparse_cuda_ops
    sparse_cuda_pkg.ops = sparse_cuda_ops
    _wrap_cpu_safe_bucket_arrange(sparse_cuda_ops)


def _load_file_module(module_name: str, path: Path):
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_pointelligence_geometry_api():
    global _POINTELLIGENCE_CACHE
    if _POINTELLIGENCE_CACHE is not None:
        return _POINTELLIGENCE_CACHE

    root = _resolve_pointelligence_root(Path(__file__).resolve())
    layers_dir = root / "layers"
    _ensure_pointelligence_import_paths(root)

    _install_sparse_engine_shim()

    package_name = "_pvd_pointelligence_layers"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(layers_dir)]
        sys.modules[package_name] = package

    metadata_module = _load_file_module(f"{package_name}.metadata", layers_dir / "metadata.py")
    _load_file_module(f"{package_name}.downsample", layers_dir / "downsample.py")
    triplets_module = _load_file_module(f"{package_name}.triplets", layers_dir / "triplets.py")

    _POINTELLIGENCE_CACHE = (
        metadata_module.MetaData,
        triplets_module.build_triplets,
        triplets_module.radius_scaler_for_kernel_size,
        triplets_module.handle_stride_and_build_triplets,
        triplets_module.voxelize_3d,
    )
    return _POINTELLIGENCE_CACHE


def _require_grid_size(state: PointCloudState) -> float:
    if state.grid_size is None:
        raise ValueError("PointCloudState.grid_size must be set before building MetaData geometry")
    return float(state.grid_size)


def state_to_metadata(state: PointCloudState):
    MetaData, _, _, _, _ = _load_pointelligence_geometry_api()
    parent_meta = state_to_metadata(state.parent) if state.parent is not None else None
    return MetaData(
        points=state.points,
        sample_inds=state.sample_inds,
        sample_sizes=state.sample_sizes,
        grid_size=_require_grid_size(state),
        i=state.i,
        j=state.j,
        k=state.k,
        num_neighbors=state.num_neighbors,
        downsample_indices=state.downsample_indices,
        i_upsample=state.i_upsample,
        j_upsample=state.j_upsample,
        k_upsample=state.k_upsample,
        parent=parent_meta,
        kernel_size=state.kernel_size,
        receptive_field_scaler=state.receptive_field_scaler,
        sort_by=state.sort_by,
        return_num_neighbors=state.return_num_neighbors,
        query_points=state.query_points,
        query_sample_inds=state.query_sample_inds,
        query_sample_sizes=state.query_sample_sizes,
        distance_type=state.distance_type,
        auto_build_triplets=False,
    )


def metadata_to_state(metadata, features: Optional[torch.Tensor] = None) -> PointCloudState:
    parent_state = metadata_to_state(metadata.parent) if metadata.parent is not None else None
    return PointCloudState(
        coords=metadata.points,
        features=features,
        sample_inds=metadata.sample_inds,
        sample_sizes=metadata.sample_sizes,
        grid_size=metadata.grid_size,
        parent=parent_state,
        query_coords=metadata.query_points,
        query_sample_inds=metadata.query_sample_inds,
        query_sample_sizes=metadata.query_sample_sizes,
        i=metadata.i,
        j=metadata.j,
        k=metadata.k,
        num_neighbors=metadata.num_neighbors,
        downsample_indices=metadata.downsample_indices,
        i_upsample=metadata.i_upsample,
        j_upsample=metadata.j_upsample,
        k_upsample=metadata.k_upsample,
        kernel_size=metadata.kernel_size,
        receptive_field_scaler=metadata.receptive_field_scaler,
        sort_by=metadata.sort_by,
        return_num_neighbors=metadata.return_num_neighbors,
        distance_type=metadata.distance_type,
    )


def build_state_triplets(
    state: PointCloudState,
    kernel_size,
    receptive_field_scaler: float = 1.0,
    sort_by: str = "k",
    return_num_neighbors: bool = False,
    distance_type: str = "ball",
) -> PointCloudState:
    metadata = state_to_metadata(state)
    _, build_triplets, radius_scaler_for_kernel_size, _, voxelize_3d = _load_pointelligence_geometry_api()
    radius_scaler = radius_scaler_for_kernel_size(kernel_size, receptive_field_scaler, distance_type)
    neighbor_radius = metadata.grid_size * radius_scaler
    i, j, k, num_neighbors = build_triplets(
        points=metadata.points,
        sample_inds=metadata.sample_inds,
        sample_sizes=metadata.sample_sizes,
        neighbor_radius=neighbor_radius,
        kernel_indexer=lambda points, grid_size: voxelize_3d(kernel_size=kernel_size, points=points, grid_size=grid_size),
        query_points=metadata.query_points,
        query_sample_inds=metadata.query_sample_inds,
        query_sample_sizes=metadata.query_sample_sizes,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
        radius_scaler=radius_scaler,
    )
    metadata.i = i
    metadata.j = j
    metadata.k = k
    metadata.num_neighbors = num_neighbors
    return metadata_to_state(metadata, features=state.features).replace_geometry(
        kernel_size=kernel_size,
        receptive_field_scaler=receptive_field_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
        distance_type=distance_type,
    )


def build_state_with_stride(
    state: PointCloudState,
    stride: float,
    kernel_size,
    receptive_field_scaler: float = 1.0,
    sort_by: str = "k",
    return_num_neighbors: bool = False,
    distance_type: str = "ball",
) -> PointCloudState:
    _, _, _, handle_stride_and_build_triplets, _ = _load_pointelligence_geometry_api()
    metadata = state_to_metadata(state)
    metadata = handle_stride_and_build_triplets(
        metadata,
        stride=stride,
        kernel_size=kernel_size,
        receptive_field_scaler=receptive_field_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
        distance_type=distance_type,
    )
    features = state.features if metadata.points.shape[0] == state.points.shape[0] else None
    return metadata_to_state(metadata, features=features).replace_geometry(
        kernel_size=kernel_size,
        receptive_field_scaler=receptive_field_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
        distance_type=distance_type,
    )
