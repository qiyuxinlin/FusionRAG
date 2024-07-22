
from setuptools import setup, Extension
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# setup marlin gemm
setup(name='KCudaOps',
      ext_modules=[
          CUDAExtension('KCudaOps', [
                'custom_gguf/dequant.cu',
                'binding.cpp',
                'gptq_marlin/gptq_marlin.cu',
                # 'gptq_marlin_repack.cu',
      ])
      ],
      cmdclass={'build_ext': BuildExtension
})

