#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

__global__ void dequantize_q8_0_kernel(float* output, const float* scales, const int8_t* qs, int num_blocks, int blk_size) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (auto block_id=global_idx; block_id<num_blocks;block_id+=blockDim.x * gridDim.x){
        for(int i=0;i<blk_size;i++){
            float scale = scales[block_id];
            output[block_id * blk_size + i] = scale * qs[block_id * blk_size + i];
            if (block_id >= 532 && block_id < 535){
                printf("block_id: %d, offset: %d, scale: %f, qs: %d, output: %f \n", block_id, i, scale, qs[block_id*blk_size+i],output[block_id*blk_size+i]);
            }
        }
    }
}

torch::Tensor dequantize_q8_0(torch::Tensor data, int blk_size, torch::Device device) {
    int num_blocks = data.numel() / blk_size;
    std::cout<< "num_blocks" << num_blocks << std::endl;
    // create gpu
    auto options_scales = torch::TensorOptions().dtype(torch::kFloat32).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto options_qs = torch::TensorOptions().dtype(torch::kInt8).device(device).memory_format(torch::MemoryFormat::Contiguous);
    auto scales_gpu = torch::empty({{num_blocks, 1}}, options_scales);
    auto qs_gpu = torch::empty({num_blocks, 32}, options_qs);

    // read on cpu
    options_scales = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
    options_qs = torch::TensorOptions().dtype(torch::kInt8).device(torch::kCPU);

    // // reinterpret
    auto scales = torch::from_blob(data.data_ptr(), {num_blocks, 1 + 16}, options_scales).slice(1, 0, 1);
    auto qs = torch::from_blob(data.data_ptr(), {num_blocks, 2 + 32}, options_qs).slice(1, 2);
    
    auto scales_f32 = scales.to(torch::kFloat32);
    scales_gpu.copy_(scales_f32, false);
    qs_gpu.copy_(qs, false);

    cudaDeviceSynchronize();
    // Create output tensor
    auto output = torch::zeros_like(qs, torch::dtype(torch::kFloat32).device(device));

    // Launch kernel
    dequantize_q8_0_kernel<<< 512, 256 >>>(
        output.data_ptr<float>(), scales_gpu.data_ptr<float>(), qs_gpu.data_ptr<int8_t>(), num_blocks, 32);

    cudaDeviceSynchronize();
    return output;
}
