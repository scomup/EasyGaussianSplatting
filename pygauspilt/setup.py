from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pygauspilt',
    ext_modules=[
        CUDAExtension('pygauspilt', [
            'ext.cpp',
            'forward.cu',
            'backward.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })