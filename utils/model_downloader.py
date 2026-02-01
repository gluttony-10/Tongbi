"""
模型下载工具函数

包含模型检测和自动下载功能
"""

import os
import subprocess
import requests
from pathlib import Path
from tqdm import tqdm


def check_model_exists(model_path):
    """检查模型文件是否存在"""
    return os.path.exists(model_path)


def download_with_modelscope(model_id, target_dir):
    """
    使用 modelscope 命令下载模型
    
    参数:
        model_id: 模型ID，如 "Gluttony10/Qwen-Image-Tongbi"
        target_dir: 目标目录
    
    返回:
        (success: bool, message: str)
    """
    try:
        os.makedirs(target_dir, exist_ok=True)
        cmd = ["modelscope", "download", "--model", model_id, "--local_dir", target_dir]
        print(f"📥 开始下载模型: {model_id} 到 {target_dir}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"✅ 模型下载成功: {model_id}")
            return True, f"✅ 模型下载成功: {model_id}"
        else:
            error_msg = result.stderr or result.stdout
            print(f"❌ 模型下载失败: {error_msg}")
            return False, f"❌ 模型下载失败: {error_msg}"
    except subprocess.TimeoutExpired:
        return False, "❌ 下载超时"
    except FileNotFoundError:
        return False, "❌ 未找到 modelscope 命令，请先安装: pip install modelscope"
    except Exception as e:
        return False, f"❌ 下载出错: {str(e)}"


def download_file_from_url(url, target_path):
    """
    从 URL 下载文件（带进度条显示）
    
    参数:
        url: 文件URL
        target_path: 目标文件路径
    
    返回:
        (success: bool, message: str)
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        print(f"📥 开始下载文件: {os.path.basename(target_path)}")
        
        proxies = {'http': None, 'https': None}  # 禁用代理
        response = requests.get(url, stream=True, proxies=proxies, timeout=3600)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            if total_size > 0:
                # 使用 tqdm 显示进度条
                with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc="下载中", ncols=80) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            else:
                # 如果无法获取文件大小，仍然下载但不显示进度条
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        print(f"✅ 文件下载成功: {os.path.basename(target_path)}")
        return True, f"✅ 文件下载成功: {os.path.basename(target_path)}"
    except requests.exceptions.RequestException as e:
        return False, f"❌ 下载失败: {str(e)}"
    except Exception as e:
        return False, f"❌ 下载出错: {str(e)}"


def ensure_base_model():
    """
    确保基础模型（Qwen-Image-Tongbi）已下载
    
    返回:
        (success: bool, message: str)
    """
    base_model_dir = "models/Qwen-Image"
    
    # 检查关键文件是否存在
    key_files = [
        "text_encoder/text_encoder-mmgp.safetensors",
        "transformer/config.json",
        "vae/config.json"
    ]
    
    all_exist = all(os.path.exists(os.path.join(base_model_dir, f)) for f in key_files)
    
    if all_exist:
        return True, "✅ 基础模型已存在"
    
    # 下载基础模型
    return download_with_modelscope("Gluttony10/Qwen-Image-Tongbi", base_model_dir)


def ensure_transformer_model(model_name):
    """
    确保 Transformer 模型已下载
    
    参数:
        model_name: 模型名称（不带 .safetensors 后缀），如 "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp"
    
    返回:
        (success: bool, message: str)
    """
    # 添加 .safetensors 后缀
    if not model_name.endswith(".safetensors"):
        model_name_full = f"{model_name}.safetensors"
    else:
        model_name_full = model_name
        model_name = model_name.replace(".safetensors", "")
    
    transformer_dir = "models/transformer"
    model_path = os.path.join(transformer_dir, model_name_full)
    
    # 检查模型是否已存在
    if os.path.exists(model_path):
        return True, f"✅ 模型已存在: {model_name_full}"
    
    # 根据模型名称确定下载URL
    model_urls = {
        "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp": "https://modelscope.cn/models/Gluttony10/Tongbi-transformer/resolve/master/Qwen-Image-2512-Lightning-4steps-V1.0-mmgp.safetensors",
        "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-mmgp": "https://modelscope.cn/models/Gluttony10/Tongbi-transformer/resolve/master/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-mmgp.safetensors",
    }
    
    if model_name not in model_urls:
        return False, f"❌ 不支持的模型: {model_name}"
    
    url = model_urls[model_name]
    
    # 下载模型文件
    return download_file_from_url(url, model_path)


def check_and_download_model(model_name):
    """
    检查并下载模型（包括基础模型和 Transformer 模型）
    
    参数:
        model_name: 模型名称（不带 .safetensors 后缀），如 "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp"
    
    返回:
        (success: bool, message: str)
    """
    # 如果是云端模型，不需要下载
    if model_name.startswith("MS-"):
        return True, "✅ 使用云端模型，无需下载"
    
    # 确保基础模型已下载
    success, msg = ensure_base_model()
    if not success:
        return False, msg
    
    # 确保 Transformer 模型已下载
    return ensure_transformer_model(model_name)
