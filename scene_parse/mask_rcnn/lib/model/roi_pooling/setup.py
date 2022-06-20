  
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='roi_pooling',
    ext_modules=[
        CUDAExtension(
            name='roi_pooling_cuda',
            sources=['src/roi_pooling.c', 'src/roi_pooling_cuda.c'
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

