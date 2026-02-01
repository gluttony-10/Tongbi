"""
图库管理工具函数

包含图库加载、刷新、选择等功能
"""

import os
from PIL import Image


def load_gallery():
    """加载图库中的所有图片"""
    outputs_dir = "outputs"
    if not os.path.exists(outputs_dir):
        return [], "❌ outputs 文件夹不存在"
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
    
    image_files = []
    for file in os.listdir(outputs_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(outputs_dir, file))

    image_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not image_files:
        return [], "📁 outputs 文件夹中没有图片文件"
    
    # 直接返回文件路径列表，不再加载为PIL图像
    return image_files, f"✅ 成功加载 {len(image_files)} 张图片"


def refresh_gallery():
    """刷新图库"""
    return load_gallery()


def load_image_info(selected_index, gallery):
    """加载选中图片的信息"""
    if selected_index is None or selected_index < 0 or selected_index >= len(gallery):
        return ""
    
    # gallery 可能是文件路径列表或元组列表
    if isinstance(gallery[selected_index], tuple):
        filepath = gallery[selected_index][0]
    else:
        filepath = gallery[selected_index]
    
    img = Image.open(filepath)
    # 读取PNG文本信息块
    if img.format == 'PNG' and hasattr(img, 'text'):
        info = "".join([f"{k}: {v}" for k, v in img.text.items()])
    else:
        info = "None"
    return info
