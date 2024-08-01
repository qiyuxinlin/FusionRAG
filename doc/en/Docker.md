# Docker

## Prerequisites
* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (ex. /mnt/models)

## Images
There are Docker images available for our project：

**Uploading**

## Building docker locally
 - Download Dockerfile in [there](../../Dockerfile)
 - update CPU_INSTRUCT in Dockerfile with your cpu instruct
   - FANCY: support AVX512VL, AVX512BW, AVX512DQ, AVX512VNNI
   - AVX512: support AVX512F
   - AVX2: support AVX2
   - NATIVE: detect your cpu instruct

 - update TORCH_CUDA_ARCH_LIST in Dockerfile with your gpu architecture
   - you can see your gpu architecture with
     ```python
     import torch
     torch.cuda.get_arch_list()
     ```

 - finish, execute
   ```bash
   docker build  -t approachingai/ktransformers:v0.1.1 .
   ```

## Usage

Assuming you have the [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) that you can use the GPU in a Docker container.
```
docker run --gpus all -v /path/to/models:/models -p 10002:10002 approachingai/ktransformers:v0.1.1 --port 10002 --gguf_path /models/path/to/gguf_path --model_path /models/path/to/model_path --web True
```

More operators you can see in the [readme](../../README.md)