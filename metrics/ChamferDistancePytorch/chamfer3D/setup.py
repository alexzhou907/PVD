import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],

    extra_cuda_cflags=([f"--compiler-bindir={os.environ.get('PVD_NVCC_COMPILER') or os.environ.get('CC')}"] if (os.environ.get('PVD_NVCC_COMPILER') or os.environ.get('CC')) else []),
    cmdclass={
        'build_ext': BuildExtension
    })