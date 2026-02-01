"""
LoRA 工具函数

用于 Qwen-Image 模型的 LoRA 权重加载和合并功能。
"""

import torch
import torch.nn as nn
import safetensors.torch


def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
    """
    构建 LoRA 权重的名称
    
    参数:
        key: 模型参数的键名
        lora_down_key: LoRA down 层的键名后缀
        lora_up_key: LoRA up 层的键名后缀
        is_native_weight: 是否为原生权重格式
    
    返回:
        (lora_down, lora_up, lora_alpha) 三个键名的元组
    """
    base = "diffusion_model." if is_native_weight else ""
    lora_down = base + key.replace(".weight", lora_down_key)
    lora_up = base + key.replace(".weight", lora_up_key)
    lora_alpha = base + key.replace(".weight", ".alpha")
    return lora_down, lora_up, lora_alpha


def load_and_merge_lora_weight(
    model: nn.Module,
    lora_state_dict: dict,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    """
    加载并合并 LoRA 权重到模型中
    
    参数:
        model: 要合并 LoRA 权重的模型
        lora_state_dict: LoRA 权重字典
        lora_down_key: LoRA down 层的键名后缀（默认 ".lora_down.weight"）
        lora_up_key: LoRA up 层的键名后缀（默认 ".lora_up.weight"）
    
    返回:
        合并了 LoRA 权重的模型
    """
    is_native_weight = any("diffusion_model." in key for key in lora_state_dict)
    
    for key, value in model.named_parameters():
        lora_down_name, lora_up_name, lora_alpha_name = build_lora_names(
            key, lora_down_key, lora_up_key, is_native_weight
        )
        
        if lora_down_name in lora_state_dict:
            lora_down = lora_state_dict[lora_down_name]
            lora_up = lora_state_dict[lora_up_name]
            lora_alpha = float(lora_state_dict[lora_alpha_name])
            rank = lora_down.shape[0]
            scaling_factor = lora_alpha / rank
            
            assert lora_up.dtype == torch.float32, f"LoRA up 层应为 float32，实际为 {lora_up.dtype}"
            assert lora_down.dtype == torch.float32, f"LoRA down 层应为 float32，实际为 {lora_down.dtype}"
            
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            value.data = (value.data + delta_W).type_as(value.data)
    
    return model


def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    """
    从 safetensors 文件加载并合并 LoRA 权重
    
    参数:
        model: 要合并 LoRA 权重的模型
        lora_weight_path: LoRA 权重文件路径（.safetensors）
        lora_down_key: LoRA down 层的键名后缀（默认 ".lora_down.weight"）
        lora_up_key: LoRA up 层的键名后缀（默认 ".lora_up.weight"）
    
    返回:
        合并了 LoRA 权重的模型
    """
    lora_state_dict = {}
    
    with safetensors.torch.safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    
    model = load_and_merge_lora_weight(
        model, lora_state_dict, lora_down_key, lora_up_key
    )
    
    return model
