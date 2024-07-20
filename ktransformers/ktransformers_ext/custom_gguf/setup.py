from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup marlin gemm
setup(name='cudaops',
      ext_modules=[
          CUDAExtension('cudaops', [
              'dequant.cu',
              'binding.cpp',
            #   'gptq_marlin_repack.cu',
      ])
      ],
      cmdclass={'build_ext': BuildExtension
      })