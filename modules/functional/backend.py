import os
import shutil

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))


def _resolve_nvcc_host_compiler():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    candidates = [
        os.environ.get("PVD_NVCC_COMPILER"),
        os.environ.get("CC"),
        os.path.join(conda_prefix, "bin", "x86_64-conda-linux-gnu-gcc"),
        os.path.join(conda_prefix, "bin", "x86_64-conda_cos6-linux-gnu-gcc"),
        shutil.which("x86_64-conda-linux-gnu-gcc"),
        shutil.which("x86_64-conda_cos6-linux-gnu-gcc"),
        shutil.which("gcc"),
        shutil.which("gcc-8"),
        "/usr/bin/gcc-8",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _collect_extra_ldflags():
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    candidates = [
        os.path.join(conda_prefix, "lib"),
        os.path.join(conda_prefix, "lib64"),
        os.path.join(conda_prefix, "targets", "x86_64-linux", "lib"),
        os.path.join(
            conda_prefix,
            "lib",
            f"python{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "site-packages",
            "nvidia",
            "cuda_runtime",
            "lib",
        ),
    ]
    ldflags = []
    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            ldflags.append(f"-L{candidate}")
    return ldflags


_extra_cuda_cflags = []
_nvcc_host_compiler = _resolve_nvcc_host_compiler()
if _nvcc_host_compiler is not None:
    _extra_cuda_cflags.append(f"--compiler-bindir={_nvcc_host_compiler}")

_backend = load(
    name="_pvcnn_backend",
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=_extra_cuda_cflags,
    extra_ldflags=_collect_extra_ldflags(),
    sources=[
        os.path.join(_src_path, "src", f)
        for f in [
            "ball_query/ball_query.cpp",
            "ball_query/ball_query.cu",
            "grouping/grouping.cpp",
            "grouping/grouping.cu",
            "interpolate/neighbor_interpolate.cpp",
            "interpolate/neighbor_interpolate.cu",
            "interpolate/trilinear_devox.cpp",
            "interpolate/trilinear_devox.cu",
            "sampling/sampling.cpp",
            "sampling/sampling.cu",
            "voxelization/vox.cpp",
            "voxelization/vox.cu",
            "bindings.cpp",
        ]
    ],
)

__all__ = ["_backend"]
