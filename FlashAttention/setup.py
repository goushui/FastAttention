from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_cuda',
    ext_modules=[
        CUDAExtension(
            name='flash_attention_cuda',
            sources=['flash_attention_cuda.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    '-arch=sm_80',  # Adjust based on your GPU (sm_70 for V100, sm_80 for A100, sm_86 for RTX 30xx)
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
