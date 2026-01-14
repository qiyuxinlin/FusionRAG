#!/usr/bin/env python3
"""
图片拼接脚本

将多张图片按 N×M 网格布局拼接成一张大图

直接修改下面的配置区域，然后运行：
    python merge_images.py
"""

import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

#############################################################################
# 配置区域 - 在这里修改输入输出路径和布局
#############################################################################

# 输入：要拼接的图片列表（按顺序从左到右，从上到下排列）
IMAGE_PATHS = [
    "/home/shm/document/exp/FusionRAG/result/fig/no_preprocess_accuracy_curves.png",
    "/home/shm/document/exp/FusionRAG/result/fig/FusionRAG_global_normal_topk10_accuracy_curves.png",
    "/home/shm/document/exp/FusionRAG/result/fig/FusionRAG_global_ramdom_topk10_accuracy_curves.png",
]

# 输出：拼接后的图片保存路径
OUTPUT_FILE = "/home/shm/document/exp/FusionRAG/result/fig/merged_comparison.png"

# 布局：(行数, 列数)
# 例如: (1, 3) = 1行3列, (2, 2) = 2行2列, (3, 1) = 3行1列
LAYOUT = (1, 3)  # 1行3列

# 子图标签（可选，留空则不添加标签）
# 如果提供，数量应与图片数量相同
LABELS = [
    "(a) No Preprocess",
    "(b) BGE Similarity",
    "(c) Random Recall"
]

# 图片间距（像素）
SPACING = 20

# 外边距（像素）
MARGIN = 30

# 标签字体大小
LABEL_FONT_SIZE = 36

# 标签位置：'top', 'bottom', 'top-left', 'top-right'
LABEL_POSITION = 'top'

# 背景颜色 (R, G, B)
BACKGROUND_COLOR = (255, 255, 255)  # 白色

#############################################################################
# 以下是脚本逻辑，通常不需要修改
#############################################################################

def load_and_resize_images(image_paths, target_width=None, target_height=None):
    """
    加载并调整图片大小，使所有图片具有相同的尺寸

    Args:
        image_paths: 图片路径列表
        target_width: 目标宽度（None表示使用最大宽度）
        target_height: 目标高度（None表示使用最大高度）

    Returns:
        list: PIL Image 对象列表
    """
    images = []

    # 加载所有图片
    print(f"加载 {len(image_paths)} 张图片...")
    for i, path in enumerate(image_paths, 1):
        if not os.path.exists(path):
            print(f"  ❌ 错误: 文件不存在: {path}")
            continue

        try:
            img = Image.open(path)
            images.append(img)
            print(f"  ✓ {i}. {os.path.basename(path)} - {img.size[0]}×{img.size[1]}")
        except Exception as e:
            print(f"  ❌ 错误: 无法加载 {path}: {e}")
            continue

    if not images:
        return []

    # 确定目标尺寸
    if target_width is None:
        target_width = max(img.size[0] for img in images)
    if target_height is None:
        target_height = max(img.size[1] for img in images)

    print(f"\n统一图片尺寸为: {target_width}×{target_height}")

    # 调整所有图片到相同尺寸
    resized_images = []
    for i, img in enumerate(images, 1):
        if img.size != (target_width, target_height):
            resized = img.resize((target_width, target_height), Image.LANCZOS)
            print(f"  ✓ 调整图片 {i}: {img.size} → {resized.size}")
        else:
            resized = img
            print(f"  ✓ 图片 {i}: 尺寸已符合")
        resized_images.append(resized)

    return resized_images

def add_label_to_image(img, label, position='top', font_size=36, bg_color=(255, 255, 255)):
    """
    在图片上添加标签

    Args:
        img: PIL Image 对象
        label: 标签文本
        position: 标签位置 ('top', 'bottom', 'top-left', 'top-right')
        font_size: 字体大小
        bg_color: 背景颜色

    Returns:
        PIL Image: 添加标签后的图片
    """
    # 创建一个新图片，上下留出标签空间
    if position in ['top', 'top-left', 'top-right']:
        label_height = font_size + 20
        new_img = Image.new('RGB', (img.size[0], img.size[1] + label_height), bg_color)
        new_img.paste(img, (0, label_height))
        text_y = 10
    else:  # bottom
        label_height = font_size + 20
        new_img = Image.new('RGB', (img.size[0], img.size[1] + label_height), bg_color)
        new_img.paste(img, (0, 0))
        text_y = img.size[1] + 10

    # 绘制文字
    draw = ImageDraw.Draw(new_img)

    # 尝试加载字体（如果失败则使用默认字体）
    try:
        # 尝试常见的字体路径
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 获取文本边界框
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]

    # 确定文本位置
    if position == 'top' or position == 'bottom':
        text_x = (new_img.size[0] - text_width) // 2  # 居中
    elif position == 'top-left':
        text_x = 20  # 左对齐
    else:  # top-right
        text_x = new_img.size[0] - text_width - 20  # 右对齐

    # 绘制文字
    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return new_img

def merge_images(images, layout, labels=None, spacing=20, margin=30,
                label_position='top', label_font_size=36, bg_color=(255, 255, 255)):
    """
    将多张图片拼接成网格布局

    Args:
        images: PIL Image 对象列表
        layout: (rows, cols) 布局
        labels: 标签列表（可选）
        spacing: 图片间距
        margin: 外边距
        label_position: 标签位置
        label_font_size: 标签字体大小
        bg_color: 背景颜色

    Returns:
        PIL Image: 拼接后的图片
    """
    rows, cols = layout

    if len(images) > rows * cols:
        print(f"⚠ 警告: 图片数量 ({len(images)}) 超过布局容量 ({rows}×{cols}={rows*cols})，将只使用前 {rows*cols} 张")
        images = images[:rows * cols]

    # 添加标签
    if labels:
        if len(labels) < len(images):
            print(f"⚠ 警告: 标签数量 ({len(labels)}) 少于图片数量 ({len(images)})")
            labels = labels + [None] * (len(images) - len(labels))

        labeled_images = []
        print(f"\n添加标签...")
        for i, (img, label) in enumerate(zip(images, labels), 1):
            if label:
                labeled_img = add_label_to_image(img, label, label_position, label_font_size, bg_color)
                print(f"  ✓ 图片 {i}: 添加标签 '{label}'")
            else:
                labeled_img = img
            labeled_images.append(labeled_img)
        images = labeled_images

    # 计算单个图片的尺寸（假设所有图片已经统一尺寸）
    img_width = images[0].size[0]
    img_height = images[0].size[1]

    # 计算总尺寸
    total_width = cols * img_width + (cols - 1) * spacing + 2 * margin
    total_height = rows * img_height + (rows - 1) * spacing + 2 * margin

    print(f"\n创建画布: {total_width}×{total_height}")

    # 创建空白画布
    canvas = Image.new('RGB', (total_width, total_height), bg_color)

    # 粘贴图片
    print(f"\n拼接图片 ({rows}×{cols})...")
    img_idx = 0
    for row in range(rows):
        for col in range(cols):
            if img_idx >= len(images):
                break

            x = margin + col * (img_width + spacing)
            y = margin + row * (img_height + spacing)

            canvas.paste(images[img_idx], (x, y))
            print(f"  ✓ 位置 [{row+1},{col+1}]: 图片 {img_idx+1}")

            img_idx += 1

    return canvas

def main():
    print("="*70)
    print("图片拼接脚本")
    print("="*70)
    print(f"输入图片数量: {len(IMAGE_PATHS)}")
    print(f"布局: {LAYOUT[0]}行×{LAYOUT[1]}列")
    print(f"输出文件: {OUTPUT_FILE}")
    print("="*70)
    print()

    # 检查图片文件
    missing_files = [p for p in IMAGE_PATHS if not os.path.exists(p)]
    if missing_files:
        print("❌ 以下文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请检查 IMAGE_PATHS 配置")
        return

    # 加载并调整图片大小
    images = load_and_resize_images(IMAGE_PATHS)

    if not images:
        print("\n❌ 错误: 没有成功加载任何图片")
        return

    if len(images) < len(IMAGE_PATHS):
        print(f"\n⚠ 警告: 只成功加载了 {len(images)}/{len(IMAGE_PATHS)} 张图片")

    # 拼接图片
    print()
    merged = merge_images(
        images,
        layout=LAYOUT,
        labels=LABELS if LABELS else None,
        spacing=SPACING,
        margin=MARGIN,
        label_position=LABEL_POSITION,
        label_font_size=LABEL_FONT_SIZE,
        bg_color=BACKGROUND_COLOR
    )

    # 保存结果
    print()
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    merged.save(OUTPUT_FILE, quality=95)

    print(f"✓ 拼接完成！")
    print(f"  输出尺寸: {merged.size[0]}×{merged.size[1]}")
    print(f"  保存路径: {OUTPUT_FILE}")

    # 显示文件大小
    file_size = os.path.getsize(OUTPUT_FILE)
    if file_size > 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.2f} MB"
    else:
        size_str = f"{file_size / 1024:.2f} KB"
    print(f"  文件大小: {size_str}")

    print("\n✅ 完成！")

if __name__ == '__main__':
    main()
