#pragma once

#include <torch/library.h>
#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor dequantize_q8_0(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q6_k(torch::Tensor data, int blk_size, torch::Device device);
torch::Tensor dequantize_q4_k(torch::Tensor data, int blk_size, torch::Device device);
// torch::Tensor dequantize_q4_k(torch::Tensor data);
// torch::Tensor dequantize_q6_k(torch::Tensor data);
// int compare(torch::Tensor data, int blk_size, torch::Device device, torch::Tensor cmp_scales, torch::Tensor cmp_qs);