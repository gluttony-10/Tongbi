"""
UI工具函数模块

包含UI相关的工具函数
"""

import os
import json
import socket
import gradio as gr
import datetime
from PIL import Image
from controlnet_aux.processor import Processor
import utils.state as state
CONFIG_FILE = state.CONFIG_FILE
from utils.config_utils import save_tab_model
from utils.prompt_enhancer import update_config
from utils.gallery_utils import load_image_info


def stop_generate():
    """停止生成"""
    import utils.state as state
    state.stop_generation = True
    return "🛑 等待生成中止"


def change_reference_count(reference_count):
    """根据参考图片数量更新UI显示"""
    if reference_count == 0:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    elif reference_count == 1:
        return gr.update(visible=True), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    elif reference_count == 2:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=None)
    elif reference_count == 3:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def scale_resolution_1_5(width, height):
    """
    将宽度和高度都放大1.5倍，并按照16的倍数向下取整
    """
    new_width = int(width * 1.5) // 16 * 16
    new_height = int(height * 1.5) // 16 * 16
    return new_width, new_height, "✅ 分辨率已调整为1.5倍"


def find_port(port: int) -> int:
    """解决冲突端口（感谢licyk酱提供的代码~）"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port


def update_selection(selected_state: gr.SelectData):
    """更新图库选择"""
    return selected_state.index


def load_image_info_wrapper(selected_index, gallery):
    """包装 load_image_info 以返回 gr.update"""
    info = load_image_info(selected_index, gallery)
    return gr.update(value=info)


def generate_cont(image, processor_id):
    """ControlNet预处理生成"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/{timestamp}.{state.image_format}"
    processor = Processor(processor_id)
    img = image.convert("RGBA")
    white_bg = Image.new("RGB", img.size, (255, 255, 255))
    white_bg.paste(img, mask=img.split()[3])
    img_rgb = white_bg.convert("RGB")
    width, height = img_rgb.size
    width = (width // 64) * 64
    height = (height // 64) * 64
    img_rgb = img_rgb.resize((width, height), Image.LANCZOS)
    processed_image = processor(img_rgb, to_pil=True)
    processed_image.save(filename, format=state.image_format.upper())
    msg = f"✅ 预处理完成,保存地址{filename}"
    print(msg)
    yield processed_image, msg


def save_openai_config(transformer_dropdown, res_vram_tb, base_url_tb, api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb, image_format_tb):
    """保存OpenAI配置"""
    import utils.state as state
    
    state.res_vram, state.base_url, state.api_key, state.model_name, state.temperature, state.top_p, state.max_tokens, state.modelscope_api_key, state.image_format = res_vram_tb, base_url_tb, api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb, image_format_tb.lower()
    state.openai_base_url, state.openai_api_key = base_url_tb, api_key_tb
    
    # 更新prompt_enhancer模块中的配置
    new_config = {
        "OPENAI_BASE_URL": base_url_tb,
        "OPENAI_API_KEY": api_key_tb,
        "MODEL_NAME": model_name_tb,
        "TEMPERATURE": temperature_tb,
        "TOP_P": top_p_tb,
        "MAX_TOKENS": max_tokens_tb,
    }
    update_config(new_config)
    
    config = {
        "RES_VRAM": res_vram_tb,
        "OPENAI_BASE_URL": base_url_tb,
        "OPENAI_API_KEY": api_key_tb,
        "MODEL_NAME": model_name_tb,
        "TEMPERATURE": temperature_tb,
        "TOP_P": top_p_tb,
        "MAX_TOKENS": max_tokens_tb,
        "MODELSCOPE_API_KEY": modelscope_api_key_tb,
        "IMAGE_FORMAT": image_format_tb.lower(),
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return "✅ 配置已保存到本地文件"
