from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup marlin gemm
setup(name='qlib',
      ext_modules=[
          CUDAExtension('qlib', [
              'qlib.cpp',
              'gptq_marlin/gptq_marlin.cu',
            #   'gptq_marlin_repack.cu',
      ])
      ],
      cmdclass={'build_ext': BuildExtension
      })

