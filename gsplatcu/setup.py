from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gsplatcu',
    ext_modules=[
        CUDAExtension('gsplatcu', [
            'ext.cpp',
            'kernel.cu',
            'gausplat.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })