
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'  # Add this line
setup(
    name='KTransformersOps',
    ext_modules=[
        CUDAExtension(
            'KTransformersOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
                # 'gptq_marlin_repack.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '--use_fast_math',
                    # '-lineinfo',
                    '-Xcompiler', '-fPIC',
                    # '-D TORCH_USE_CUDA_DSA'  # Add this line
                ]
            },
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)