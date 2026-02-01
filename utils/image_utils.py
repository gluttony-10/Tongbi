"""
图片处理工具函数

包含图片编码、上传、元数据创建、尺寸调整等功能
"""

import io
import base64
import requests
import datetime
import math
from PIL import Image, PngImagePlugin
from diffusers.utils import load_image


def encode_file(img):
    """将PIL图片编码为base64 data URI"""
    format = (img.format or "PNG").upper()
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    byte_data = buffer.getvalue()
    mime_type = f"image/{format.lower()}"
    encoded_string = base64.b64encode(byte_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def upload_image_to_smms(image_pil):
    """
    上传图片到SM.MS图床（无需注册）
    返回图片URL，失败返回None
    """
    try:
        # 将PIL图片转换为bytes
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        buffer.seek(0)
        
        # 上传到SM.MS
        files = {'smfile': ('image.png', buffer, 'image/png')}
        proxies = {'http': None, 'https': None}  # 禁用代理
        response = requests.post('https://sm.ms/api/v2/upload', files=files, proxies=proxies)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                return data['data']['url']
            else:
                print(f"SM.MS上传失败: {data.get('message', '未知错误')}")
                return None
        else:
            print(f"SM.MS上传失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"上传图片到SM.MS时出错: {str(e)}")
        return None


def create_pnginfo(mode, prompt, negative_prompt, seed, transformer_dropdown, 
                   num_inference_steps=None, true_cfg_scale=None, 
                   width=None, height=None, strength=None,
                   lora_dropdown=None, lora_weights=None, 
                   image=None, generation_time=None):
    """
    创建 PNG 元数据信息
    
    参数:
        mode: 生成模式 (t2i, i2i, inp, con, editplus 等)
        prompt: 正向提示词
        negative_prompt: 负向提示词
        seed: 随机种子
        transformer_dropdown: 使用的模型
        num_inference_steps: 推理步数
        true_cfg_scale: CFG 强度
        width: 图像宽度
        height: 图像高度
        strength: 强度参数（i2i/inp/con 模式）
        lora_dropdown: LoRA 列表
        lora_weights: LoRA 权重
        image: 输入图像（仅作标记）
        generation_time: 生成耗时（秒）
    
    返回:
        PngImagePlugin.PngInfo 对象
    """
    pnginfo = PngImagePlugin.PngInfo()
    
    # 基础信息
    pnginfo.add_text("mode", mode)
    pnginfo.add_text("prompt", prompt)
    pnginfo.add_text("negative_prompt", negative_prompt or "")
    pnginfo.add_text("seed", str(seed))
    pnginfo.add_text("model", transformer_dropdown)
    
    # 生成参数
    if num_inference_steps is not None:
        pnginfo.add_text("num_inference_steps", str(num_inference_steps))
    if true_cfg_scale is not None:
        pnginfo.add_text("true_cfg_scale", str(true_cfg_scale))
    if width is not None and mode != "edit2":
        pnginfo.add_text("width", str(width))
    if height is not None:
        pnginfo.add_text("height", str(height))
    if strength is not None:
        pnginfo.add_text("strength", str(strength))
    
    # LoRA 信息
    if lora_dropdown:
        lora_str = ", ".join(lora_dropdown)
        pnginfo.add_text("lora", lora_str)
        if lora_weights:
            lora_weights_str = ", ".join(lora_weights) if isinstance(lora_weights, list) else lora_weights
            pnginfo.add_text("lora_weights", lora_weights_str)
    
    # 是否使用输入图像
    if image is not None:
        pnginfo.add_text("has_input_image", "true")
    
    # 生成时间
    if generation_time is not None:
        pnginfo.add_text("generation_time", f"{generation_time:.2f}s")
    
    # 生成日期时间
    pnginfo.add_text("created_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return pnginfo


def exchange_width_height(width, height):
    """交换宽高"""
    return height, width, "✅ 宽高交换完毕"


def adjust_width_height(image):
    """根据图片调整宽高（用于文生图、图生图等）"""
    if isinstance(image, dict):
        image = image["background"]
    image = load_image(image)
    width, height = image.size
    original_area = width * height
    default_area = 1328*1328
    ratio = math.sqrt(original_area / default_area)
    width = width / ratio // 16 * 16
    height = height / ratio // 16 * 16
    return int(width), int(height), "✅ 根据图片调整宽高"


def calculate_dimensions(target_area, ratio):
    """计算目标尺寸"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return int(width), int(height)


def adjust_width_height_editplus2(image):
    """根据图片调整宽高（用于多图编辑）"""
    image_width, image_height = image.size
    vae_width, vae_height = calculate_dimensions(1024*1024, image_width / image_height)
    calculated_height = vae_height // 32 * 32
    calculated_width = vae_width // 32 * 32
    return int(calculated_width), int(calculated_height), "✅ 根据图片调整宽高"


def snap_to_nearest(value, options):
    """将值对齐到最近的选项"""
    return min(options, key=lambda x: abs(x - value))
