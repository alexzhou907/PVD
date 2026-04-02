from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Optional

import torch


_KEEP = object()
_GEOMETRY_MODULE = None


def _load_geometry_module():
    global _GEOMETRY_MODULE
    if _GEOMETRY_MODULE is None:
        module_path = Path(__file__).resolve().parent / "point_cloud_geometry.py"
        spec = importlib.util.spec_from_file_location("_pvd_point_cloud_geometry_impl", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        _GEOMETRY_MODULE = module
    return _GEOMETRY_MODULE


def _validate_sample_sizes(sample_sizes: torch.Tensor) -> torch.Tensor:
    if sample_sizes.dim() != 1:
        raise ValueError("sample_sizes must be a 1D tensor")
    if torch.any(sample_sizes < 0):
        raise ValueError("sample_sizes must be nonnegative")
    return sample_sizes.to(dtype=torch.long)


def _validate_triplet_tensor(name: str, value: Optional[torch.Tensor], *, device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() != 1:
        raise ValueError(f"{name} must be a 1D tensor")
    return value.to(dtype=torch.long, device=device)


def _validate_sample_layout(
    sample_inds: torch.Tensor,
    sample_sizes: torch.Tensor,
    *,
    num_values: int,
    device,
    value_name: str,
):
    sample_sizes = _validate_sample_sizes(sample_sizes).to(device=device)
    sample_inds = sample_inds.to(dtype=torch.long, device=device)
    if sample_inds.dim() != 1 or sample_inds.shape[0] != num_values:
        raise ValueError(f"{value_name}_sample_inds must have shape [{num_values}]")
    if sample_sizes.sum().item() != num_values:
        raise ValueError(f"{value_name}_sample_sizes must sum to the number of values")
    if torch.any(sample_inds < 0):
        raise ValueError(f"{value_name}_sample_inds must be nonnegative")
    if sample_sizes.numel() == 0:
        if sample_inds.numel() != 0:
            raise ValueError(f"{value_name}_sample_inds must be empty when there are no samples")
        return sample_inds, sample_sizes
    if torch.any(sample_inds >= sample_sizes.numel()):
        raise ValueError(f"{value_name}_sample_inds must be smaller than the batch size")
    if sample_inds.numel() > 1 and torch.any(sample_inds[1:] < sample_inds[:-1]):
        raise ValueError(f"{value_name}_sample_inds must be nondecreasing")
    expected_sample_inds = offsets_to_sample_inds(sample_sizes_to_offsets(sample_sizes)).to(device=device)
    if not torch.equal(sample_inds, expected_sample_inds):
        raise ValueError(f"{value_name}_sample_inds must match the run-length encoding implied by {value_name}_sample_sizes")
    return sample_inds, sample_sizes


def sample_sizes_to_offsets(sample_sizes: torch.Tensor) -> torch.Tensor:
    sample_sizes = _validate_sample_sizes(sample_sizes)
    zeros = torch.zeros(1, dtype=sample_sizes.dtype, device=sample_sizes.device)
    return torch.cat([zeros, sample_sizes.cumsum(dim=0)], dim=0)


def offsets_to_sample_inds(offsets: torch.Tensor) -> torch.Tensor:
    if offsets.dim() != 1:
        raise ValueError("offsets must be a 1D tensor")
    if offsets.numel() == 0:
        raise ValueError("offsets must contain at least one element")
    if offsets[0].item() != 0:
        raise ValueError("offsets must start at 0")
    if torch.any(offsets[1:] < offsets[:-1]):
        raise ValueError("offsets must be nondecreasing")

    sample_inds = []
    for sample_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        count = int((end - start).item())
        sample_inds.append(
            torch.full((count,), sample_idx, dtype=torch.long, device=offsets.device)
        )
    if not sample_inds:
        return torch.empty(0, dtype=torch.long, device=offsets.device)
    return torch.cat(sample_inds, dim=0)


def expand_sample_tensor(sample_tensor: torch.Tensor, sample_inds: torch.Tensor) -> torch.Tensor:
    if sample_tensor.dim() < 1:
        raise ValueError("sample_tensor must have at least one dimension")
    if sample_inds.dim() != 1:
        raise ValueError("sample_inds must be a 1D tensor")
    return sample_tensor.index_select(0, sample_inds.to(dtype=torch.long, device=sample_tensor.device))


def segment_mean(values: torch.Tensor, sample_sizes: torch.Tensor) -> torch.Tensor:
    if values.dim() < 1:
        raise ValueError("values must have at least one dimension")

    sample_sizes = _validate_sample_sizes(sample_sizes).to(device=values.device)
    if sample_sizes.sum().item() != values.shape[0]:
        raise ValueError("sample_sizes must sum to the number of values")

    reduced = []
    start = 0
    for count in sample_sizes.tolist():
        end = start + count
        segment = values[start:end]
        if segment.shape[0] == 0:
            raise ValueError("segment_mean is undefined for empty samples")
        reduced.append(segment.mean(dim=0))
        start = end
    if not reduced:
        return values.new_empty((0,) + tuple(values.shape[1:]))
    return torch.stack(reduced, dim=0)


def dense_values_to_flat(values: torch.Tensor, sample_sizes: torch.Tensor) -> torch.Tensor:
    if values.dim() != 3:
        raise ValueError("values must have shape [B, C, N]")

    sample_sizes = _validate_sample_sizes(sample_sizes).to(device=values.device)
    if values.shape[0] != sample_sizes.numel():
        raise ValueError("sample_sizes must match batch size")

    flat = []
    for batch_idx, count in enumerate(sample_sizes.tolist()):
        flat.append(values[batch_idx, :, :count].transpose(0, 1))
    if not flat:
        return values.new_empty((0, values.shape[1]))
    return torch.cat(flat, dim=0)


def point_cloud_state_from_points_list(
    points_list: list[torch.Tensor],
    *,
    features_list: Optional[list[torch.Tensor]] = None,
    grid_size: Optional[float] = None,
    **geometry_kwargs,
) -> "PointCloudState":
    if len(points_list) == 0:
        raise ValueError("points_list must not be empty")

    device = points_list[0].device
    sample_sizes = []
    flat_points = []
    flat_features = []

    if features_list is not None and len(features_list) != len(points_list):
        raise ValueError("features_list must match points_list length")

    for sample_idx, points in enumerate(points_list):
        if points.dim() != 2 or points.shape[1] != 3:
            raise ValueError("each points tensor must have shape [Ni, 3]")
        if points.device != device:
            raise ValueError("all points tensors must be on the same device")

        point_count = int(points.shape[0])
        sample_sizes.append(point_count)
        flat_points.append(points)

        if features_list is not None:
            features = features_list[sample_idx]
            if features.dim() != 2 or features.shape[0] != point_count:
                raise ValueError("each features tensor must have shape [Ni, C]")
            if features.device != device:
                raise ValueError("all features tensors must be on the same device")
            flat_features.append(features)

    sample_sizes = torch.tensor(sample_sizes, dtype=torch.long, device=device)
    sample_inds = torch.repeat_interleave(
        torch.arange(sample_sizes.numel(), dtype=torch.long, device=device),
        sample_sizes,
    )
    coords = torch.cat(flat_points, dim=0).contiguous()
    features = torch.cat(flat_features, dim=0).contiguous() if features_list is not None else coords

    return PointCloudState(
        coords=coords,
        features=features,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
        **geometry_kwargs,
    )


def _move_tensor(
    value: Optional[torch.Tensor],
    *,
    device=None,
    dtype=None,
    non_blocking: bool = False,
    preserve_dtype: bool = False,
):
    if value is None:
        return None
    kwargs = {"non_blocking": non_blocking}
    if device is not None:
        kwargs["device"] = device
    if not preserve_dtype and dtype is not None and (value.is_floating_point() or value.is_complex()):
        kwargs["dtype"] = dtype
    return value.to(**kwargs)


@dataclass(frozen=True)
class PointCloudState:
    coords: torch.Tensor
    features: Optional[torch.Tensor]
    sample_inds: torch.Tensor
    sample_sizes: torch.Tensor
    grid_size: Optional[float] = None
    parent: Optional["PointCloudState"] = None
    query_coords: Optional[torch.Tensor] = None
    query_sample_inds: Optional[torch.Tensor] = None
    query_sample_sizes: Optional[torch.Tensor] = None
    i: Optional[torch.Tensor] = None
    j: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    num_neighbors: Optional[torch.Tensor] = None
    downsample_indices: Optional[torch.Tensor] = None
    i_upsample: Optional[torch.Tensor] = None
    j_upsample: Optional[torch.Tensor] = None
    k_upsample: Optional[torch.Tensor] = None
    kernel_size: Optional[tuple[int, int, int]] = None
    receptive_field_scaler: float = 1.0
    sort_by: str = "k"
    return_num_neighbors: bool = False
    distance_type: str = "ball"

    def __post_init__(self):
        if self.coords.dim() != 2 or self.coords.shape[1] != 3:
            raise ValueError("coords must have shape [M, 3]")
        sample_inds, sample_sizes = _validate_sample_layout(
            self.sample_inds,
            self.sample_sizes,
            num_values=self.coords.shape[0],
            device=self.coords.device,
            value_name="sample",
        )
        if self.features is not None:
            if self.features.dim() != 2 or self.features.shape[0] != self.coords.shape[0]:
                raise ValueError("features must have shape [M, C]")
            object.__setattr__(self, "features", self.features.to(device=self.coords.device))

        query_fields = (self.query_coords, self.query_sample_inds, self.query_sample_sizes)
        if any(field is not None for field in query_fields):
            if any(field is None for field in query_fields):
                raise ValueError("query_coords, query_sample_inds, and query_sample_sizes must be all set or all None")
            query_coords = self.query_coords.to(device=self.coords.device)
            if query_coords.dim() != 2 or query_coords.shape[1] != 3:
                raise ValueError("query_coords must have shape [Q, 3]")
            query_sample_inds, query_sample_sizes = _validate_sample_layout(
                self.query_sample_inds,
                self.query_sample_sizes,
                num_values=query_coords.shape[0],
                device=self.coords.device,
                value_name="query",
            )
            object.__setattr__(self, "query_coords", query_coords)
            object.__setattr__(self, "query_sample_inds", query_sample_inds)
            object.__setattr__(self, "query_sample_sizes", query_sample_sizes)

        triplets = {
            "i": self.i,
            "j": self.j,
            "k": self.k,
            "num_neighbors": self.num_neighbors,
            "downsample_indices": self.downsample_indices,
            "i_upsample": self.i_upsample,
            "j_upsample": self.j_upsample,
            "k_upsample": self.k_upsample,
        }
        converted_triplets = {
            name: _validate_triplet_tensor(name, value, device=self.coords.device)
            for name, value in triplets.items()
        }
        if any(converted_triplets[name] is None for name in ("i", "j", "k")):
            if any(converted_triplets[name] is not None for name in ("i", "j", "k")):
                raise ValueError("i, j, and k must be all set or all None")
        if any(converted_triplets[name] is None for name in ("i_upsample", "j_upsample", "k_upsample")):
            if any(converted_triplets[name] is not None for name in ("i_upsample", "j_upsample", "k_upsample")):
                raise ValueError("i_upsample, j_upsample, and k_upsample must be all set or all None")
        for name, value in converted_triplets.items():
            object.__setattr__(self, name, value)

        object.__setattr__(self, "sample_inds", sample_inds)
        object.__setattr__(self, "sample_sizes", sample_sizes)

    @property
    def points(self) -> torch.Tensor:
        return self.coords

    @property
    def query_points(self) -> Optional[torch.Tensor]:
        return self.query_coords

    @property
    def offsets(self) -> torch.Tensor:
        return sample_sizes_to_offsets(self.sample_sizes)

    @property
    def batch_ids(self) -> torch.Tensor:
        return self.sample_inds

    def num_points(self) -> int:
        return int(self.coords.shape[0])

    def empty_triplets(self) -> bool:
        if self.i is None or self.j is None or self.k is None:
            return True
        return False

    def geometry_only(self) -> "PointCloudState":
        return self.replace_features(None)

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "PointCloudState":
        if device is None and dtype is None:
            return self
        return self.replace_geometry(
            coords=_move_tensor(self.coords, device=device, dtype=dtype, non_blocking=non_blocking),
            features=_move_tensor(self.features, device=device, dtype=dtype, non_blocking=non_blocking),
            sample_inds=_move_tensor(self.sample_inds, device=device, non_blocking=non_blocking, preserve_dtype=True),
            sample_sizes=_move_tensor(self.sample_sizes, device=device, non_blocking=non_blocking, preserve_dtype=True),
            parent=None if self.parent is None else self.parent.to(device=device, dtype=dtype, non_blocking=non_blocking),
            query_coords=_move_tensor(self.query_coords, device=device, dtype=dtype, non_blocking=non_blocking),
            query_sample_inds=_move_tensor(self.query_sample_inds, device=device, non_blocking=non_blocking, preserve_dtype=True),
            query_sample_sizes=_move_tensor(self.query_sample_sizes, device=device, non_blocking=non_blocking, preserve_dtype=True),
            i=_move_tensor(self.i, device=device, non_blocking=non_blocking, preserve_dtype=True),
            j=_move_tensor(self.j, device=device, non_blocking=non_blocking, preserve_dtype=True),
            k=_move_tensor(self.k, device=device, non_blocking=non_blocking, preserve_dtype=True),
            num_neighbors=_move_tensor(self.num_neighbors, device=device, non_blocking=non_blocking, preserve_dtype=True),
            downsample_indices=_move_tensor(self.downsample_indices, device=device, non_blocking=non_blocking, preserve_dtype=True),
            i_upsample=_move_tensor(self.i_upsample, device=device, non_blocking=non_blocking, preserve_dtype=True),
            j_upsample=_move_tensor(self.j_upsample, device=device, non_blocking=non_blocking, preserve_dtype=True),
            k_upsample=_move_tensor(self.k_upsample, device=device, non_blocking=non_blocking, preserve_dtype=True),
        )

    def replace_features(self, new_features: Optional[torch.Tensor]) -> "PointCloudState":
        return self.replace_geometry(features=new_features)

    def replace_coords_and_features(
        self,
        new_coords: torch.Tensor,
        new_features: Optional[torch.Tensor] = None,
    ) -> "PointCloudState":
        if new_coords.shape != self.coords.shape:
            raise ValueError("replace_coords_and_features requires the same flattened geometry")
        if new_features is None:
            new_features = new_coords
        return self.replace_geometry(
            coords=new_coords,
            features=new_features,
            parent=None,
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

    def replace_query_context(
        self,
        query_coords: Optional[torch.Tensor],
        query_sample_inds: Optional[torch.Tensor],
        query_sample_sizes: Optional[torch.Tensor],
    ) -> "PointCloudState":
        return self.replace_geometry(
            query_coords=query_coords,
            query_sample_inds=query_sample_inds,
            query_sample_sizes=query_sample_sizes,
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

    def dirty_triplets(self) -> "PointCloudState":
        return self.replace_geometry(
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

    def replace_geometry(
        self,
        *,
        coords=_KEEP,
        features=_KEEP,
        sample_inds=_KEEP,
        sample_sizes=_KEEP,
        grid_size=_KEEP,
        parent=_KEEP,
        query_coords=_KEEP,
        query_sample_inds=_KEEP,
        query_sample_sizes=_KEEP,
        i=_KEEP,
        j=_KEEP,
        k=_KEEP,
        num_neighbors=_KEEP,
        downsample_indices=_KEEP,
        i_upsample=_KEEP,
        j_upsample=_KEEP,
        k_upsample=_KEEP,
        kernel_size=_KEEP,
        receptive_field_scaler=_KEEP,
        sort_by=_KEEP,
        return_num_neighbors=_KEEP,
        distance_type=_KEEP,
    ) -> "PointCloudState":
        return PointCloudState(
            coords=self.coords if coords is _KEEP else coords,
            features=self.features if features is _KEEP else features,
            sample_inds=self.sample_inds if sample_inds is _KEEP else sample_inds,
            sample_sizes=self.sample_sizes if sample_sizes is _KEEP else sample_sizes,
            grid_size=self.grid_size if grid_size is _KEEP else grid_size,
            parent=self.parent if parent is _KEEP else parent,
            query_coords=self.query_coords if query_coords is _KEEP else query_coords,
            query_sample_inds=self.query_sample_inds if query_sample_inds is _KEEP else query_sample_inds,
            query_sample_sizes=self.query_sample_sizes if query_sample_sizes is _KEEP else query_sample_sizes,
            i=self.i if i is _KEEP else i,
            j=self.j if j is _KEEP else j,
            k=self.k if k is _KEEP else k,
            num_neighbors=self.num_neighbors if num_neighbors is _KEEP else num_neighbors,
            downsample_indices=self.downsample_indices if downsample_indices is _KEEP else downsample_indices,
            i_upsample=self.i_upsample if i_upsample is _KEEP else i_upsample,
            j_upsample=self.j_upsample if j_upsample is _KEEP else j_upsample,
            k_upsample=self.k_upsample if k_upsample is _KEEP else k_upsample,
            kernel_size=self.kernel_size if kernel_size is _KEEP else kernel_size,
            receptive_field_scaler=self.receptive_field_scaler if receptive_field_scaler is _KEEP else receptive_field_scaler,
            sort_by=self.sort_by if sort_by is _KEEP else sort_by,
            return_num_neighbors=self.return_num_neighbors if return_num_neighbors is _KEEP else return_num_neighbors,
            distance_type=self.distance_type if distance_type is _KEEP else distance_type,
        )


def dense_to_point_cloud_state(
    coords: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    grid_size: Optional[float] = None,
    **geometry_kwargs,
) -> PointCloudState:
    if coords.dim() != 3 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [B, 3, N]")

    batch_size, _, num_points = coords.shape
    sample_sizes = torch.full(
        (batch_size,),
        num_points,
        dtype=torch.long,
        device=coords.device,
    )
    offsets = sample_sizes_to_offsets(sample_sizes)
    sample_inds = offsets_to_sample_inds(offsets)
    flat_coords = coords.permute(0, 2, 1).reshape(batch_size * num_points, 3).contiguous()

    flat_features = None
    if features is not None:
        if features.dim() != 3 or features.shape[0] != batch_size or features.shape[2] != num_points:
            raise ValueError("features must have shape [B, C, N]")
        flat_features = features.permute(0, 2, 1).reshape(batch_size * num_points, -1).contiguous()

    return PointCloudState(
        coords=flat_coords,
        features=flat_features if flat_features is not None else flat_coords,
        sample_inds=sample_inds,
        sample_sizes=sample_sizes,
        grid_size=grid_size,
        **geometry_kwargs,
    )


def point_cloud_state_to_dense(
    values: torch.Tensor,
    sample_sizes: torch.Tensor,
    fill_value: float = 0,
) -> torch.Tensor:
    if values.dim() != 2:
        raise ValueError("values must have shape [M, C]")

    sample_sizes = _validate_sample_sizes(sample_sizes).to(device=values.device)
    if sample_sizes.sum().item() != values.shape[0]:
        raise ValueError("sample_sizes must sum to the number of values")

    batch_size = sample_sizes.numel()
    max_points = int(sample_sizes.max().item()) if batch_size > 0 else 0
    dense = values.new_full((batch_size, values.shape[1], max_points), fill_value)

    start = 0
    for batch_idx, count in enumerate(sample_sizes.tolist()):
        end = start + count
        if count > 0:
            dense[batch_idx, :, :count] = values[start:end].transpose(0, 1)
        start = end
    return dense


def state_to_metadata(state: PointCloudState):
    return _load_geometry_module().state_to_metadata(state)


def metadata_to_state(metadata, features: Optional[torch.Tensor] = None) -> PointCloudState:
    return _load_geometry_module().metadata_to_state(metadata, features=features)


def build_state_triplets(
    state: PointCloudState,
    kernel_size,
    receptive_field_scaler: float = 1.0,
    sort_by: str = "k",
    return_num_neighbors: bool = False,
    distance_type: str = "ball",
) -> PointCloudState:
    return _load_geometry_module().build_state_triplets(
        state,
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
    return _load_geometry_module().build_state_with_stride(
        state,
        stride=stride,
        kernel_size=kernel_size,
        receptive_field_scaler=receptive_field_scaler,
        sort_by=sort_by,
        return_num_neighbors=return_num_neighbors,
        distance_type=distance_type,
    )
