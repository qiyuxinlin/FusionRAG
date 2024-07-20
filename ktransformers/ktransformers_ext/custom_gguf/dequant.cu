#include <cuda_runtime.h>
#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
// #include "dequant.cuh"
// __global__ void dequantize_q8_0_kernel(float* output, const float* scales, const int8_t* qs, int num_blocks, int blk_size) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx >= num_blocks) {
//         return;
//     }
//     float scale = scales[idx];
//     int8_t q = qs[idx];
//     output[idx] = scale * q;
// }

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

    // check is contiguous
    std::cout<< "output is contiguous: " << output.is_contiguous() << output.sizes() << std::endl;
    std::cout<< "scales_gpu is contiguous: " << scales_gpu.is_contiguous() << scales_gpu.sizes() << std::endl;
    std::cout<< "qs_gpu is contiguous: " << qs_gpu.is_contiguous() << qs_gpu.sizes() << std::endl;

    // for(int i=0; i< num_blocks; i++){
    //     for(int j=0; j<32; j++){
    //         if (i >= 532 && i < 533){
    //             printf("i: %d, j: %d, scales: %f, qs: %d , qs addr: %lld , scale addr: %lld, +1 addr: %lld\n", i, j, scales_gpu[i][0].item<float>(), qs_gpu[i][j].item<int8_t>(), qs_gpu[i][j].data_ptr<int8_t>(), scales_gpu[i][0].data_ptr<float>(), scales_gpu[i+1][0].data_ptr<float>());
    //         }
    //     }
    // }
    
    float* a=(float*)malloc(sizeof(float)*num_blocks*32);
    float* b=(float*)malloc(sizeof(float)*num_blocks);
    int8_t* c=(int8_t*)malloc(sizeof(int8_t)*num_blocks*32);
    // cudaMemcpy(a, output.data_ptr<float>(), num_blocks*32*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(b, scales_gpu.data_ptr<float>(), num_blocks*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(c, qs_gpu.data_ptr<int8_t>(), num_blocks*32*sizeof(int8_t),cudaMemcpyDeviceToHost);
    for(int i=0; i< num_blocks; i++){
        for(int j=0; j<32; j++){
            if (i >= 532 && i < 533){
                // printf("i: %d, j: %d, scales: %f, qs: %d \n", i, j, b[i], (int)c[i*32+j]);
                std::cout<< "i: " << i << ", j: " << j << ", scales: " << b[i] << ", qs: " << (int)c[i*32+j] << std::endl;
            }
        }
    }


    // Launch kernel
    dequantize_q8_0_kernel<<< 512, 256 >>>(
        output.data_ptr<float>(), scales_gpu.data_ptr<float>(), qs_gpu.data_ptr<int8_t>(), num_blocks, 32);

    cudaDeviceSynchronize();
    return output;
}


int compare(torch::Tensor data, int blk_size, torch::Device device, torch::Tensor cmp_scales, torch::Tensor cmp_qs) {
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

    // reinterpret
    auto scales = torch::from_blob(data.data_ptr(), {num_blocks, 1 + 16}, options_scales).slice(1, 0, 1);
    auto qs = torch::from_blob(data.data_ptr(), {num_blocks, 2 + 32}, options_qs).slice(1, 2);

    auto scales_cp = scales.to(torch::kFloat32);
    scales_gpu.copy_(scales_cp, false);
    qs_gpu.copy_(qs, false);

    cmp_qs = cmp_qs.to(device);
    cmp_scales = cmp_scales.to(device);
    cudaDeviceSynchronize();
    int ne_count = 0;
    for(int i=0; i< num_blocks; i++){
        for(int j=0; j<32; j++){
            if (i >= 532 && i < 533){
                printf("i: %d, j: %d, scales: %f, qs: %d \n", i, j, scales_gpu[i][0].item<float>(), qs_gpu[i][j].item<int8_t>());
                printf("i: %d, j: %d, cmp_scales: %f, cmp_qs: %d \n", i, j, cmp_scales[i][0].item<float>(), cmp_qs[i][j].item<int8_t>());
            }
        }
    }
    // 使用torch::allclose检查两个张量是否近似相等
    bool close = torch::allclose(qs_gpu, cmp_qs, 0.001, 1e-6);
    bool scale_close = torch::allclose(scales_gpu, cmp_scales, 0.001, 1e-6);
    std::cout << "The qs are close: " << (close ? "Yes" : "No") << std::endl;
    std::cout << "The scales are close: " << (scale_close ? "Yes" : "No") << std::endl;

    return ne_count;
}
