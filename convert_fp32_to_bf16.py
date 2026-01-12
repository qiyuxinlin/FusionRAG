#!/usr/bin/env python3
"""
将 FP32 模型转换为 BF16 格式

Usage:
    python convert_fp32_to_bf16.py --input /mnt/data/models/DistilQwen2.5-3B-Instruct-FP32 \
                                   --output /mnt/data/models/DistilQwen2.5-3B-Instruct-BF16
"""

import os
import json
import shutil
import argparse
import torch
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file


def convert_model(input_dir: str, output_dir: str):
    """
    将 FP32 模型转换为 BF16
    """
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有文件
    files = os.listdir(input_dir)

    # 分类文件
    safetensor_files = [f for f in files if f.endswith('.safetensors')]
    bin_files = [f for f in files if f.endswith('.bin') and 'optimizer' not in f.lower()]
    other_files = [f for f in files if not f.endswith('.safetensors') and not f.endswith('.bin')]

    print(f"\nFound {len(safetensor_files)} safetensor files")
    print(f"Found {len(bin_files)} bin files")
    print(f"Found {len(other_files)} other files")

    # 复制非模型文件（config, tokenizer 等）
    print("\n[1/3] Copying non-model files...")
    for f in tqdm(other_files, desc="Copying"):
        src = os.path.join(input_dir, f)
        dst = os.path.join(output_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    # 转换 safetensors 文件
    if safetensor_files:
        print("\n[2/3] Converting safetensors files to BF16...")
        for sf_file in tqdm(safetensor_files, desc="Converting safetensors"):
            src_path = os.path.join(input_dir, sf_file)
            dst_path = os.path.join(output_dir, sf_file)

            # 读取 safetensors
            tensors = {}
            with safe_open(src_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    # 只转换浮点类型
                    if tensor.dtype in [torch.float32, torch.float16]:
                        tensors[key] = tensor.to(torch.bfloat16)
                    else:
                        tensors[key] = tensor

            # 保存为 BF16
            save_file(tensors, dst_path)

            # 打印大小对比
            src_size = os.path.getsize(src_path) / (1024**3)
            dst_size = os.path.getsize(dst_path) / (1024**3)
            print(f"  {sf_file}: {src_size:.2f} GB -> {dst_size:.2f} GB")

    # 转换 .bin 文件（如果有）
    if bin_files:
        print("\n[3/3] Converting .bin files to BF16...")
        for bin_file in tqdm(bin_files, desc="Converting bin"):
            src_path = os.path.join(input_dir, bin_file)
            dst_path = os.path.join(output_dir, bin_file)

            # 读取 checkpoint
            state_dict = torch.load(src_path, map_location="cpu", weights_only=True)

            # 转换
            converted = {}
            for key, tensor in state_dict.items():
                if isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.float32, torch.float16]:
                    converted[key] = tensor.to(torch.bfloat16)
                else:
                    converted[key] = tensor

            # 保存
            torch.save(converted, dst_path)

            src_size = os.path.getsize(src_path) / (1024**3)
            dst_size = os.path.getsize(dst_path) / (1024**3)
            print(f"  {bin_file}: {src_size:.2f} GB -> {dst_size:.2f} GB")

    # 更新 config.json 中的 torch_dtype
    config_path = os.path.join(output_dir, "config.json")
    if os.path.exists(config_path):
        print("\n[4/4] Updating config.json...")
        with open(config_path, 'r') as f:
            config = json.load(f)

        old_dtype = config.get('torch_dtype', 'unknown')
        config['torch_dtype'] = 'bfloat16'

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  torch_dtype: {old_dtype} -> bfloat16")

    print("\n" + "="*60)
    print("Conversion complete!")
    print(f"Output saved to: {output_dir}")

    # 计算总大小
    total_src = sum(os.path.getsize(os.path.join(input_dir, f)) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))
    total_dst = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))

    print(f"\nTotal size: {total_src/(1024**3):.2f} GB -> {total_dst/(1024**3):.2f} GB")
    print(f"Compression ratio: {total_dst/total_src*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Convert FP32 model to BF16")
    parser.add_argument('--input', type=str, required=True, help='Input model directory (FP32)')
    parser.add_argument('--output', type=str, required=True, help='Output model directory (BF16)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        return 1

    convert_model(args.input, args.output)
    return 0


if __name__ == '__main__':
    exit(main())
