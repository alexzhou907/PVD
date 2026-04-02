from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types

import torch
import torch.nn as nn

from modules.shared_mlp import SharedMLP
from point_cloud_state import (
    PointCloudState,
    build_state_triplets,
    dense_to_point_cloud_state,
    point_cloud_state_to_dense,
)

__all__ = ["PointelligencePVConv"]


_POINTELLIGENCE_NN_CACHE = None
_POINTELLIGENCE_FORWARD_CACHE = {}


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


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def clear_pointelligence_forward_cache():
    _POINTELLIGENCE_FORWARD_CACHE.clear()


def _resolve_pointelligence_root(module_file: Path) -> Path:
    candidates = [
        module_file.parents[2],
        module_file.parents[2] / "pointelligence",
    ]
    for root in candidates:
        if (root / "layers" / "conv.py").exists():
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


def _load_pointelligence_pvconv_api():
    global _POINTELLIGENCE_NN_CACHE
    if _POINTELLIGENCE_NN_CACHE is not None:
        return _POINTELLIGENCE_NN_CACHE

    root = _resolve_pointelligence_root(Path(__file__).resolve())
    layers_dir = root / "layers"
    _ensure_pointelligence_import_paths(root)

    _install_sparse_engine_shim()

    package_name = "_pvd_pointelligence_layers"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(layers_dir)]
        sys.modules[package_name] = package

    _load_file_module(f"{package_name}.metadata", layers_dir / "metadata.py")
    _load_file_module(f"{package_name}.downsample", layers_dir / "downsample.py")
    _load_file_module(f"{package_name}.triplets", layers_dir / "triplets.py")
    conv_module = _load_file_module(f"{package_name}.conv", layers_dir / "conv.py")
    upsample_module = _load_file_module(f"{package_name}.upsample", layers_dir / "upsample.py")

    _POINTELLIGENCE_NN_CACHE = (
        conv_module.PointConv3d,
        conv_module.conv_with_stride,
        upsample_module.Upsample,
    )
    return _POINTELLIGENCE_NN_CACHE


class _PointelligenceSameScaleBranch(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        receptive_field_scaler: float = 1.0,
        dropout: float | None = 0.1,
    ):
        super().__init__()
        PointConv3d, _, _ = _load_pointelligence_pvconv_api()

        self.receptive_field_scaler = receptive_field_scaler
        self.conv1 = PointConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        )
        self.conv2 = PointConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.swish1 = Swish()
        self.swish2 = Swish()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None

    def prepare_state(self, state: PointCloudState) -> PointCloudState:
        expected_kernel_size = tuple(self.conv1.kernel_size_3)
        if (
            not state.empty_triplets()
            and state.kernel_size == expected_kernel_size
            and state.receptive_field_scaler == self.receptive_field_scaler
            and state.sort_by == "k"
            and state.return_num_neighbors is False
            and state.distance_type == "ball"
            and state.query_coords is None
            and state.query_sample_inds is None
            and state.query_sample_sizes is None
        ):
            return state

        if not state.empty_triplets():
            state = state.dirty_triplets()
        return build_state_triplets(
            state,
            kernel_size=expected_kernel_size,
            receptive_field_scaler=self.receptive_field_scaler,
        )

    def forward(self, state: PointCloudState) -> torch.Tensor:
        if state.features is None:
            raise ValueError("Pointelligence voxel branch requires state.features")

        state = self.prepare_state(state)
        features = self.conv1(
            state.features,
            state.i,
            state.j,
            state.k,
            state.num_points(),
        )
        features = self.swish1(self.norm1(features))
        if self.dropout is not None:
            features = self.dropout(features)
        features = self.conv2(
            features,
            state.i,
            state.j,
            state.k,
            state.num_points(),
        )
        features = self.swish2(self.norm2(features))
        return features


class PointelligencePVConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        resolution,
        attention=False,
        dropout=0.1,
        with_se=False,
        with_se_relu=False,
        normalize=True,
        eps=0,
    ):
        super().__init__()
        if attention:
            raise ValueError("pointelligence backend does not support PVConv local attention")
        del with_se, with_se_relu
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = int(resolution)
        self.normalize = normalize
        self.eps = eps

        self.voxel_branch = _PointelligenceSameScaleBranch(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.point_features = SharedMLP(in_channels, out_channels)

    def _proxy_cache_key(self, state: PointCloudState):
        return (
            state.coords.data_ptr(),
            state.sample_inds.data_ptr(),
            tuple(state.sample_sizes.tolist()),
            self.resolution,
            self.normalize,
            float(self.eps),
            tuple(self.voxel_branch.conv1.kernel_size_3),
            float(self.voxel_branch.receptive_field_scaler),
        )

    def _normalize_coords(self, state: PointCloudState) -> PointCloudState:
        coords = state.coords.detach()
        normalized_chunks = []
        start = 0
        for count in state.sample_sizes.tolist():
            end = start + count
            sample_coords = coords[start:end]
            sample_coords = sample_coords - sample_coords.mean(dim=0, keepdim=True)
            if self.normalize:
                scale = sample_coords.norm(dim=1).max() * 2.0 + self.eps
                sample_coords = sample_coords / scale + 0.5
            else:
                sample_coords = (sample_coords + 1.0) / 2.0
            sample_coords = torch.clamp(sample_coords * self.resolution, 0, self.resolution - 1)
            sample_coords = sample_coords / float(self.resolution)
            normalized_chunks.append(sample_coords)
            start = end
        normalized_coords = torch.cat(normalized_chunks, dim=0).contiguous()
        return PointCloudState(
            coords=normalized_coords,
            features=state.features,
            sample_inds=state.sample_inds,
            sample_sizes=state.sample_sizes,
            grid_size=1.0 / float(self.resolution),
        )

    def _forward_state(self, state: PointCloudState, temb_sample):
        if state.features is None:
            raise ValueError("PointelligencePVConv requires point features")
        cache_key = self._proxy_cache_key(state)
        proxy_geometry = _POINTELLIGENCE_FORWARD_CACHE.get(cache_key)
        if proxy_geometry is None:
            proxy_geometry = self.voxel_branch.prepare_state(self._normalize_coords(state).geometry_only())
            _POINTELLIGENCE_FORWARD_CACHE[cache_key] = proxy_geometry
        proxy_state = proxy_geometry.replace_features(state.features)
        point_features = self.point_features(state.features)
        voxel_features = self.voxel_branch(proxy_state)
        return state.replace_features(point_features + voxel_features), temb_sample

    def _forward_dense(self, features, coords, temb):
        state = dense_to_point_cloud_state(
            coords,
            features=features,
            grid_size=1.0 / float(self.resolution),
        )
        state_out, _ = self._forward_state(state, None)
        dense_out = point_cloud_state_to_dense(state_out.features, state_out.sample_sizes)
        return dense_out, coords, temb

    def forward(self, inputs):
        if len(inputs) == 2 and isinstance(inputs[0], PointCloudState):
            return self._forward_state(*inputs)
        features, coords, temb = inputs
        return self._forward_dense(features, coords, temb)
