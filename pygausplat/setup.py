from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pygausplat',
    ext_modules=[
        CUDAExtension('pygausplat', [
            'ext.cpp',
            'kernel.cu',
            'gausplat.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })