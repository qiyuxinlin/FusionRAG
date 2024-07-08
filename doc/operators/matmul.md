# CPUInfer Operators Documentation

## Overview
CPUInfer is an inference framework optimized for CPUs that leverages the llamafile sgemm module to implement key operators such as linear layers, MLP, and MoE. These operators are fundamental components for building large models. CPUInfer uses a backend work-stealing thread pool and asynchronous task queue execution logic to efficiently offload parts of model parameters to the CPU, thereby maintaining high inference performance. It supports adjustments based on hardware capabilities or user configurations, providing enhanced inference performance and making it an ideal tool for running deep learning models on CPUs.

## Llamafile Sgemm
The llamafile sgemm module is a core component for implementing general matrix multiplication (GEMM). This module optimizes performance by using various processor-specific instruction sets. For example, it includes checks for different x86 instruction sets such as AVX, FMA, and AVX512, and leverages these advanced instructions to accelerate computation. Additionally, the llamafile sgemm module supports multiple quantization types, such as q8_0, q6_k, q5_k, among a dozen others. This module aims to adapt to different hardware capabilities, ensuring that it utilizes the most advanced instructions available in a given computing environment, thereby achieving high computational efficiency. You can view the file and its details on [GitHub](https://github.com/Mozilla-Ocho/llamafile/blob/main/llamafile/sgemm.cpp) for more information.

## Microbenchmark

### Evaluation Setup

We conducted our evaluations on an Intel (R) Xeon (R) Gold 6454S processor. To ensure the validity of our test results, we used the real parameters of the DeepSeek-Coder-V2-Instruct model. The microbenchmark was performed on three operators: Linear, MLP, and MoE. Here are the specific configurations:

1. **Linear Operator (corresponding to self_attn.o_proj)**
   - Input Size: 16384
   - Output Size: 5120

2. **MLP Operator (corresponding to mlp.shared_experts)**
   - Hidden Size: 5120
   - Intermediate Size: 3072

3. **MoE Operator (corresponding to mlp.experts)**
   - Number of Experts: 160
   - Hidden Size: 5120
   - Intermediate Size: 1536
   - Number of Routed Experts: 6

### Benchmark Results

| Framework | Data type | Linear time per iteration (µs) | Linear bandwidth (GB/s) | MLP time per iteration (µs) | MLP bandwidth (GB/s) | MoE time per iteration (µs) | MoE bandwidth (GB/s) |
|-----------|-----------|-------------------------------|-------------------------|-----------------------------|-----------------------|-----------------------------|-----------------------|
| Torch     | fp32      | 1474.89                       | 211.88                  | 900.34                      | 195.24                | 2985.58                     | 176.63                |
| Torch     | fp16      | 1244.81                       | 125.52                  | 787.61                      | 111.59                | 3177.17                     | 82.99                 |
| Torch     | bf16      | 1052.27                       | 148.49                  | 687.56                      | 127.83                | 2693.89                     | 97.88                 |
| Torch     | qint8     | 439.31                        | 177.84                  | 492.12                      | 89.3                  | 2054.65                     | 64.16                 |
| CPUInfer  | fp32      | 1550.09                       | 201.6                   | 879.74                      | 199.81                | 2644.29                     | 199.43                |
| CPUInfer  | fp16      | 802.16                        | 194.79                  | 479.55                      | 183.28                | 1560.48                     | 168.97                |
| CPUInfer  | bf16      | 817.48                        | 191.14                  | 486.19                      | 180.77                | 1588.45                     | 165.99                |
| CPUInfer  | q8_0      | 459                           | 180.85                  | 305.99                      | 152.59                | 1063.13                     | 131.76                |
| CPUInfer  | q6_k      | 404.85                        | 158.3                   | 296.43                      | 121.61                | 1023.21                     | 105.69                |
| CPUInfer  | q5_k_m    | 353.11                        | 152.11                  | 281.03                      | 114.43                | 969.92                      | 99.47                 |
| CPUInfer  | q4_k_m    | 303.04                        | 145.01                  | 257.16                      | 110.81                | 914.3                       | 93.5                  |
| CPUInfer  | q3_k_m    | 253.24                        | 132.56                  | 231.83                      | 97.74                 | 792.31                      | 85.8                  |
| CPUInfer  | q2_k      | 212.51                        | 120.63                  | 205.43                      | 70.19                 | 645.13                      | 67.05                 |

Our comparison against Torch across various data types shows that CPUInfer achieves significantly faster inference speeds. Specifically, in half-precision floating-point operations, CPUInfer is 1.29 to 2.04 times faster than Torch. Moreover, across different quantization formats, CPUInfer outperforms Torch by up to 2.07 to 3.18 times. These results underscore CPUInfer's superior performance and versatility, making it an optimal choice for high-performance computing tasks that require support for diverse quantization levels and hardware environments.