# DeepSeek-V2-local

A heterogeneous solution to serve the full DeepSeek-V2 model with a single node and a single GPU.

By splitting the model into the MoE and non-MoE part, and placing them on CPU and GPU, respectively, we can achieve
a more efficient serving of the model. Specifically, the MoE part enjoys the large memory capacity of the CPU, and the 
non-MoE part enjoys the high memory bandwidth and computation power of the GPU.

Aside from running the non-quantized version of DeepSeek-V2, the following quantization methods are supported:
- GPU Part: `qint8` with `optimum.quanto`
- CPU Part: `quint8` with ad-hoc quantization

## Getting Started

### Compiling the C++ Extension

``` bash
git submodule init
git submodule update
mkdir csrc/build
cd csrc/build
cmake .. && make -j
```

### Running the Chat

```
python3 local_chat.py <gpu_part_path> <cpu_part_path> [ --gpu-quant <gpu_quant_type> ]
```

## System Requirements

The following table shows the minimum and recommended hardware requirements to run the non-quantized DeepSeek-V2 Model 
locally using this project:

|          | Minimum | Recommended |
|:--------:|:--------:|:--------:|
| **CPU**  | x86-64 CPU with AVX2 support | Intel(R) Xeon(R) Scalable 4th Gen / AMD EPYC 4th Gen |
| **RAM**  | 512 GB | 1024 GB DDR5 |
| **GPU**  | NVIDIA Ampere GPU with 32 GB memory or better | NVIDIA A100-40G |
| **Free space** | 700 GB | 1 TB |

The project is tested under the following environments:
- CPU: Intel(R) Xeon(R) Platinum 8452Y
- RAM: 2048 GB DDR5-4400
- GPU: NVIDIA A100-PCIE-40GB

With quantization enabled, the system requirements can be relaxed.

The CPU part may additionally benefit from the following instruction set extensions:
- **AVX512**: for all operators.
- **AVX512-BF16**: for some bf16-related operators.
- **AVX512-VNNI** or **AVX-VNNI**: for some operators with quantized operands.

## Preparing The Heterogeneous Model

The heterogeneous version contains a GPU part (of all non-MoE modules) and a CPU part (of all MoE experts). 
Both parts can be quantized, but the way of quantization is different. The GPU part is quantized during model
loading with `optimum.quanto`, and the CPU part is quantized during preprocessing in an ad-hoc manner.

### Preparing the GPU Part

``` bash
python3 -m tools.prepare_gpu --save-directory ${path_to_save}
```

### Preparing the CPU Part with Optional Quantization

The weights of the MoE experts are quantized row-wise with affine quantizer. The activations are dynamically quantized 
with symmetric quantizer.

``` bash
python3 -m tools.prepare_cpu --quant ${quant_type} --save-directory ${path_to_save} --use-cuda
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The source code, scripts, and other materials in this project can be used to prepare, run, or host the DeepSeek-V2 
Model, its derivatives, or the similar. We kindly remind you that the use of the Model and the Derivatives of the Model 
from DeepSeek may be subject to the terms and conditions from DeepSeek. You may use the Model and the Derivatives of 
the Model from DeepSeek in accordance with DEEPSEEK LICENSE AGREEMENT - see the [LICENSE.DeepSeek](LICENSE.DeepSeek) 
file for details.
