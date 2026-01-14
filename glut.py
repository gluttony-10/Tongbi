import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.registry")
warnings.filterwarnings("ignore", category=UserWarning, module="controlnet_aux.segment_anything.modeling.tiny_vit_sam")
import gc
import io
import os
from PIL import Image, PngImagePlugin
import json
import math
import time
from mmgp import offload, profile_type
import base64
import torch
import torch.nn as nn
import numpy as np
import socket
import psutil
import random
import gradio as gr
from openai import OpenAI
import requests
import argparse
import datetime
import mimetypes
from diffusers import QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler, QwenImageImg2ImgPipeline, QwenImagePipeline, QwenImageInpaintPipeline, QwenImageControlNetPipeline, QwenImageControlNetModel, QwenImageEditPlusPipeline
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
import safetensors.torch
from transformers import Qwen2_5_VLForConditionalGeneration
from controlnet_aux.processor import Processor
from prompt_enhancer import enhance_prompt, enhance_prompt_edit2, update_config

parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7891, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument("--compile", action="store_true", help="是否启用compile加速")
args = parser.parse_args()

print(" 启动中，请耐心等待 bilibili@十字鱼 https://space.bilibili.com/893892")
print(f'\033[32mPytorch版本：{torch.__version__}\033[0m')
if torch.cuda.is_available():
    device = "cuda" 
    print(f'\033[32m显卡型号：{torch.cuda.get_device_name()}\033[0m')
    total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824
    print(f'\033[32m显存大小：{total_vram_in_gb:.2f}GB\033[0m')
    mem = psutil.virtual_memory()
    print(f'\033[32m内存大小：{mem.total/1073741824:.2f}GB\033[0m')
    if torch.cuda.get_device_capability()[0] >= 8:
        print(f'\033[32m支持BF16\033[0m')
        dtype = torch.bfloat16
    else:
        print(f'\033[32m不支持BF16，使用FP32\033[0m')
        dtype = torch.float32
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"


#初始化
config = {}
transformer_choices = []
transformer_choices2 = []
transformer_loaded = None
lora_choices = []
lora_loaded = None
lora_loaded_weights = None
image_loaded = None
mode = None
mode_loaded = None
pipe = None
prompt_cache = None
negative_prompt_cache = None
model_id = "models/Qwen-Image"
stop_generation = False
mmgp = None
EXAMPLES_FILE = "examples.json"
#确保输出文件夹存在
os.makedirs("outputs", exist_ok=True)
#读取设置
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
#默认设置
transformer_ = config.get("TRANSFORMER_DROPDOWN", "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp.safetensors")
transformer_2 = config.get("TRANSFORMER_DROPDOWN2", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-mmgp.safetensors")
res_vram = float(config.get("RES_VRAM", "1000"))
openai_base_url = config.get("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
openai_api_key = config.get("OPENAI_API_KEY", "")
model_name = config.get("MODEL_NAME", "GLM-4.6V-Flash")
temperature = float(config.get("TEMPERATURE", "0.8"))
top_p = float(config.get("TOP_P", "0.6"))
max_tokens = float(config.get("MAX_TOKENS", "16384"))
modelscope_api_key = config.get("MODELSCOPE_API_KEY", "")


def refresh_model():
    global transformer_choices, transformer_choices2, lora_choices
    transformer_dir = "models/transformer"
    lora_dir = "models/lora"
    if os.path.exists(transformer_dir):
        transformer_files = [f for f in os.listdir(transformer_dir) if f.endswith(".safetensors")]
        transformer_choices = sorted([f for f in transformer_files] + ["ModelScope-QI.safetensors"])
        transformer_choices2 = sorted([f for f in transformer_files if "edit" in f or "Edit" in f])
    else:
        print("transformer文件夹不存在")
    if os.path.exists(lora_dir):
        lora_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
        lora_choices = sorted(lora_files)
    else:
        lora_choices = []
    return gr.Dropdown(choices=transformer_choices), gr.Dropdown(choices=transformer_choices2), gr.Dropdown(choices=lora_choices)

def initialize_examples_file():
    """
    初始化 EXAMPLES_FILE 文件，确保每个 TabItem 都有默认提示词
    """
    default_examples = {
        "t2i": ["选择保存过的提示词"],
        "i2i": ["选择保存过的提示词"],
        "inp": ["选择保存过的提示词"],
        "con": ["选择保存过的提示词"],
        "editplus": ["选择保存过的提示词"],
        "camera": ["选择保存过的提示词"]
    }
    
    if not os.path.exists(EXAMPLES_FILE):
        with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
            json.dump(default_examples, f, ensure_ascii=False, indent=2)
        print(f"✅ 已创建 {EXAMPLES_FILE} 并写入默认示例")
    else:
        # 检查现有文件是否包含所有必需的标签
        with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                # 如果文件损坏或为空，重新创建
                existing_data = {}
        
        # 确保所有标签都存在
        updated = False
        for tab in default_examples.keys():
            if tab not in existing_data:
                existing_data[tab] = default_examples[tab]
                updated = True
                
        if updated:
            with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 已更新 {EXAMPLES_FILE}，添加缺失的标签")


initialize_examples_file()
refresh_model()


def build_lora_names(key, lora_down_key, lora_up_key, is_native_weight):
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
            assert lora_up.dtype == torch.float32
            assert lora_down.dtype == torch.float32
            delta_W = scaling_factor * torch.matmul(lora_up, lora_down)
            value.data = (value.data + delta_W).type_as(value.data)
    return model


def load_and_merge_lora_weight_from_safetensors(
    model: nn.Module,
    lora_weight_path: str,
    lora_down_key: str = ".lora_down.weight",
    lora_up_key: str = ".lora_up.weight",
):
    lora_state_dict = {}
    with safetensors.torch.safe_open(lora_weight_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            lora_state_dict[key] = f.get_tensor(key)
    model = load_and_merge_lora_weight(
        model, lora_state_dict, lora_down_key, lora_up_key
    )
    return model


def load_model(mode, transformer_dropdown, lora_dropdown, lora_weights, res_vram):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    global pipe, mode_loaded, transformer_loaded, lora_loaded, lora_loaded_weights, mmgp
    res_vram = float(res_vram)
    # 使用 mem_get_info 获取可用显存（单位：字节）
    free_memory, _ = torch.cuda.mem_get_info(0)
    budgets = int(free_memory / 1048576 - res_vram)
    scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
    # 判断是否需要重新加载模型
    if (pipe is None or mode_loaded != mode or transformer_loaded != transformer_dropdown or 
        lora_loaded != lora_dropdown or lora_loaded_weights != lora_weights):
        if pipe is not None:
            pipe.unload_lora_weights()
            mmgp.release()
        # 更新全局状态
        mode_loaded, transformer_loaded, lora_loaded, lora_loaded_weights = (
            mode, transformer_dropdown, lora_dropdown, lora_weights
        )
        text_encoder = offload.fast_load_transformers_model(
            f"{model_id}/text_encoder/text_encoder-mmgp.safetensors",
            do_quantize=False,
            modelClass=Qwen2_5_VLForConditionalGeneration,
            forcedConfigPath=f"{model_id}/text_encoder/config.json",
        )
        # 加载transformer
        # 转换mmgp模型，需要加载原始模型+lora，或者直接加载训练或融好的模型
        # 方法一，加载原始diffusers模型
        #transformer = QwenImageTransformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype)
        # 方法二，加载单文件模型
        #transformer = QwenImageTransformer2DModel.from_single_file("models/Real-Qwen-Image-V1.safetensors", config=f"{model_id}/transformer/config.json", torch_dtype=dtype)
        # 方法三，加载单文件模型
        """transformer = QwenImageTransformer2DModel.from_config(f"{model_id}/transformer/config.json", torch_dtype=dtype)
        state_dict = safetensors.torch.load_file("Real-Qwen-Image-V1.safetensors")
        transformer.load_state_dict(state_dict)"""
        # 加载加速lora
        #transformer = load_and_merge_lora_weight_from_safetensors(transformer, "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors")
        # Tongbi中的加载mmgp模型方式，转换时请注释掉
        if "mmgp" in transformer_dropdown:
            transformer = offload.fast_load_transformers_model(
                f"models/transformer/{transformer_dropdown}",
                do_quantize=False,
                modelClass=QwenImageTransformer2DModel,
                forcedConfigPath=f"{model_id}/transformer/config.json",
            )
        else:
            raise ValueError("请使用mmgp转化后保存的模型")
        # 加载scheduler
        if "Lightning" in transformer_dropdown:
            print("使用Lightning scheduler")
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        else:
            print("使用原始scheduler")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
        # 根据模式初始化pipeline
        pipeline_class = {
            "t2i": QwenImagePipeline,
            "i2i": QwenImageImg2ImgPipeline,
            "inp": QwenImageInpaintPipeline,
            "con": QwenImageControlNetPipeline,
            "editplus":QwenImageEditPlusPipeline,
        }.get(mode)
        if pipeline_class is None:
            raise ValueError(f"Unsupported mode: {mode}")
        if mode != "con":
            pipe = pipeline_class.from_pretrained(
                model_id, 
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=dtype,
            )
        elif mode == "con":
            controlnet = QwenImageControlNetModel.from_pretrained("models/Qwen-Image-ControlNet-Union", torch_dtype=dtype,)
            pipe = pipeline_class.from_pretrained(
                model_id, 
                controlnet=controlnet,
                text_encoder=text_encoder,
                transformer=transformer,
                scheduler=scheduler,
                torch_dtype=dtype,
            )
        if mode in ["editplus"]:
            pipe.set_progress_bar_config(disable=None)
        # 加载LoRA并配置显存
        load_lora(lora_dropdown, lora_weights)
        mmgp = offload.all(
            pipe, 
            pinnedMemory = ["text_encoder", "transformer"] if mem.total/1073741824 > 60 else "transformer",
            budgets={'*': budgets}, 
            extraModelsToQuantize = ["text_encoder"],
            #quantizeTransformer = False,
            compile=True if args.compile else False,
        )
        #offload.save_model(pipe.transformer, "models/transformer-mmgp.safetensors")
        if torch.cuda.get_device_capability()[0] >= 8:
            pipe.transformer.set_attention_backend("flash")
        else:
            pipe.transformer.set_attention_backend("native")


def load_lora(lora_dropdown, lora_weights):
    if lora_dropdown != []:
        global pipe
        adapter_names = []
        weightss = []
        weights = [float(w) for w in lora_weights.split(',')] if lora_weights else []
        for idx, lora_name in enumerate(lora_dropdown):
            try:
                adapter_name = os.path.splitext(os.path.basename(lora_name))[0]
                adapter_names.append(adapter_name)
                weight = weights[idx] if idx < len(weights) else 1.0
                weightss.append(weight)
                pipe.load_lora_weights(f"models/lora/{lora_name}", adapter_name=adapter_name)
                print(f"✅ 已加载LoRA模型: {lora_name} (权重: {weight})")
            except Exception as e:
                print(f"❌ 加载{adapter_name}失败: {str(e)}")
        pipe.set_adapters(adapter_names, adapter_weights=weightss)
        print("LoRA加载完成")


def encode_file(img):
    format = (img.format or "PNG").upper()
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    byte_data = buffer.getvalue()
    mime_type = f"image/{format.lower()}"
    encoded_string = base64.b64encode(byte_data).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"


def modelscope_generate(
    mode,
    prompt,
    negative_prompt,
    width,
    height,
    batch_images, 
    seed_param,
    transformer_dropdown,
    image=None, 
):
    global stop_generation
    num_inference_steps = 50  
    true_cfg_scale = 4.0 
    results = []
    resolutions = [
        (928, 1664),
        (1104, 1472),
        (1328, 1328),
        (1472, 1104),
        (1664, 928)
    ]
    if image:
        pil_img = image.convert("RGB")
        """pil_img = image.convert("RGB")
        filename = f"outputs/20.PNG"
        pil_img.save(filename, format='PNG')
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError("不支持或无法识别的图像格式")
        with open(filename, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')"""
        format = (pil_img.format or "PNG").upper()
        buffer = io.BytesIO()
        pil_img.save(buffer, format=format)
        byte_data = buffer.getvalue()
        mime_type = f"image/{format.lower()}"
        encoded_string = base64.b64encode(byte_data).decode("utf-8")
        width, height = load_image(pil_img).size
    min_distance = float('inf') # 初始化最小距离为正无穷大
    for res_width, res_height in resolutions:
        # 使用欧几里得距离计算相似度
        distance = ((width - res_width) ** 2 + (height - res_height) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            closest_resolution = (res_width, res_height)
    width, height = closest_resolution[0], closest_resolution[1]
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    base_url = 'https://api-inference.modelscope.cn/'
    common_headers = {
        "Authorization": f"Bearer {modelscope_api_key}",
        "Content-Type": "application/json",
    }
    for i in range(batch_images):
        if stop_generation:
            stop_generation = False
            yield results, f"✅ 生成已中止，最后种子数{seed+i-1}"
            break
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}.png"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("mode", f"{mode}\n")
        if image:
            pnginfo.add_text("image", f"{str(image)}\n")
        pnginfo.add_text("prompt", f"{prompt}\n")
        pnginfo.add_text("negative_prompt", f"{negative_prompt}\n")
        pnginfo.add_text("num_inference_steps", f"{str(num_inference_steps)}\n")
        pnginfo.add_text("true_cfg_scale", f"{str(true_cfg_scale)}\n")
        pnginfo.add_text("seed", f"{str(seed + i)}\n")
        pnginfo.add_text("models", f"{transformer_dropdown}\n")
        if mode == "t2i_ms":
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps({
                    "model": "Qwen/Qwen-Image", # ModelScope Model-Id, required
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "true_cfg_scale": true_cfg_scale,
                    "size": f"{width}x{height}",
                    "seed": seed + i,
                }, ensure_ascii=False).encode('utf-8')
            )
        elif mode == "edit_ms":
            response = requests.post(
                f"{base_url}v1/images/generations",
                headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                data=json.dumps({
                    "model": "Qwen/Qwen-Image-Edit", # ModelScope Model-Id, required
                    "image": f"data:{mime_type};base64,{encoded_string}",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "true_cfg_scale": true_cfg_scale,
                    "size": f"{width}x{height}",
                    "seed": seed + i,
                }, ensure_ascii=False).encode('utf-8')
            )
        response.raise_for_status()
        task_id = response.json()["task_id"]
        while True:
            result = requests.get(
                f"{base_url}v1/tasks/{task_id}",
                headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
            )
            result.raise_for_status()
            data = result.json()
            if data["task_status"] == "SUCCEED":
                image = Image.open(io.BytesIO(requests.get(data["output_images"][0]).content))
                break
            elif data["task_status"] == "FAILED":
                print("Image Generation Failed.")
                break
            time.sleep(1)
        image.save(filename, pnginfo=pnginfo)
        results.append(image)
        yield results, f"种子数{seed+i}，保存地址{filename}"


def exchange_width_height(width, height):
    return height, width, "✅ 宽高交换完毕"


def adjust_width_height(image):
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


def adjust_width_height_editplus2(image):
    image_width, image_height = image.size
    vae_width, vae_height = calculate_dimensions(1024*1024, image_width / image_height)
    calculated_height = vae_height // 32 * 32
    calculated_width = vae_width // 32 * 32
    return int(calculated_width), int(calculated_height), "✅ 根据图片调整宽高"


def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    return width, height


# --- Camera Control Functions ---

# Azimuth mappings (8 positions)
AZIMUTH_MAP = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view"
}

# Elevation mappings (4 positions)
ELEVATION_MAP = {
    -30: "low-angle shot",
    0: "eye-level shot",
    30: "elevated shot",
    60: "high-angle shot"
}

# Distance mappings (3 positions)
DISTANCE_MAP = {
    0.6: "close-up",
    1.0: "medium shot",
    1.8: "wide shot"
}


def snap_to_nearest(value, options):
    """Snap a value to the nearest option in a list."""
    return min(options, key=lambda x: abs(x - value))


def build_camera_prompt(azimuth: float, elevation: float, distance: float) -> str:
    """
    Build a camera prompt from azimuth, elevation, and distance values.
    
    Args:
        azimuth: Horizontal rotation in degrees (0-360)
        elevation: Vertical angle in degrees (-30 to 60)
        distance: Distance factor (0.6 to 1.8)
    
    Returns:
        Formatted prompt string for the LoRA
    """
    # Snap to nearest valid values
    azimuth_snapped = snap_to_nearest(azimuth, list(AZIMUTH_MAP.keys()))
    elevation_snapped = snap_to_nearest(elevation, list(ELEVATION_MAP.keys()))
    distance_snapped = snap_to_nearest(distance, list(DISTANCE_MAP.keys()))
    
    azimuth_name = AZIMUTH_MAP[azimuth_snapped]
    elevation_name = ELEVATION_MAP[elevation_snapped]
    distance_name = DISTANCE_MAP[distance_snapped]
    
    return f"<sks> {azimuth_name} {elevation_name} {distance_name}"


def update_dimensions_on_upload_camera(image):
    """Compute recommended dimensions preserving aspect ratio for camera control."""
    if image is None:
        return 1024, 1024, "✅ 根据图片调整宽高"
    
    image_width, image_height = image.size
    vae_width, vae_height = calculate_dimensions(1024*1024, image_width / image_height)
    calculated_height = vae_height // 32 * 32
    calculated_width = vae_width // 32 * 32
    return int(calculated_width), int(calculated_height), "✅ 根据图片调整宽高"


# --- 3D Camera Control Component ---
class CameraControl3D(gr.HTML):
    """
    A 3D camera control component using Three.js.
    Outputs: { azimuth: number, elevation: number, distance: number }
    Accepts imageUrl prop to display user's uploaded image on the plane.
    """
    def __init__(self, value=None, imageUrl=None, **kwargs):
        if value is None:
            value = {"azimuth": 0, "elevation": 0, "distance": 1.0}
        
        html_template = """
        <div id="camera-control-wrapper" style="width: 100%; height: 450px; position: relative; background: #1a1a1a; border-radius: 12px; overflow: hidden;">
            <div id="prompt-overlay" style="position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.8); padding: 8px 16px; border-radius: 8px; font-family: monospace; font-size: 12px; color: #00ff88; white-space: nowrap; z-index: 10;"></div>
        </div>
        """
        
        js_on_load = """
        (() => {
            const wrapper = element.querySelector('#camera-control-wrapper');
            const promptOverlay = element.querySelector('#prompt-overlay');
            
            // Wait for THREE to load
            const initScene = () => {
                if (typeof THREE === 'undefined') {
                    setTimeout(initScene, 100);
                    return;
                }
                
                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0x1a1a1a);
                
                const camera = new THREE.PerspectiveCamera(50, wrapper.clientWidth / wrapper.clientHeight, 0.1, 1000);
                camera.position.set(4.5, 3, 4.5);
                camera.lookAt(0, 0.75, 0);
                
                const renderer = new THREE.WebGLRenderer({ antialias: true });
                renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
                wrapper.insertBefore(renderer.domElement, promptOverlay);
                
                // Lighting
                scene.add(new THREE.AmbientLight(0xffffff, 0.6));
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
                dirLight.position.set(5, 10, 5);
                scene.add(dirLight);
                
                // Grid
                scene.add(new THREE.GridHelper(8, 16, 0x333333, 0x222222));
                
                // Constants - reduced distances for tighter framing
                const CENTER = new THREE.Vector3(0, 0.75, 0);
                const BASE_DISTANCE = 1.6;
                const AZIMUTH_RADIUS = 2.4;
                const ELEVATION_RADIUS = 1.8;
                
                // State
                let azimuthAngle = props.value?.azimuth || 0;
                let elevationAngle = props.value?.elevation || 0;
                let distanceFactor = props.value?.distance || 1.0;
                
                // Mappings - reduced wide shot multiplier
                const azimuthSteps = [0, 45, 90, 135, 180, 225, 270, 315];
                const elevationSteps = [-30, 0, 30, 60];
                const distanceSteps = [0.6, 1.0, 1.4];
                
                const azimuthNames = {
                    0: 'front view', 45: 'front-right quarter view', 90: 'right side view',
                    135: 'back-right quarter view', 180: 'back view', 225: 'back-left quarter view',
                    270: 'left side view', 315: 'front-left quarter view'
                };
                const elevationNames = { '-30': 'low-angle shot', '0': 'eye-level shot', '30': 'elevated shot', '60': 'high-angle shot' };
                const distanceNames = { '0.6': 'close-up', '1': 'medium shot', '1.4': 'wide shot' };
                
                function snapToNearest(value, steps) {
                    return steps.reduce((prev, curr) => Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
                }
                
                // Create placeholder texture (smiley face)
                function createPlaceholderTexture() {
                    const canvas = document.createElement('canvas');
                    canvas.width = 256;
                    canvas.height = 256;
                    const ctx = canvas.getContext('2d');
                    ctx.fillStyle = '#3a3a4a';
                    ctx.fillRect(0, 0, 256, 256);
                    ctx.fillStyle = '#ffcc99';
                    ctx.beginPath();
                    ctx.arc(128, 128, 80, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.fillStyle = '#333';
                    ctx.beginPath();
                    ctx.arc(100, 110, 10, 0, Math.PI * 2);
                    ctx.arc(156, 110, 10, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.strokeStyle = '#333';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(128, 130, 35, 0.2, Math.PI - 0.2);
                    ctx.stroke();
                    return new THREE.CanvasTexture(canvas);
                }
                
                // Target image plane
                let currentTexture = createPlaceholderTexture();
                const planeMaterial = new THREE.MeshBasicMaterial({ map: currentTexture, side: THREE.DoubleSide });
                let targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                targetPlane.position.copy(CENTER);
                scene.add(targetPlane);
                
                // Function to update texture from image URL
                function updateTextureFromUrl(url) {
                    if (!url) {
                        // Reset to placeholder
                        planeMaterial.map = createPlaceholderTexture();
                        planeMaterial.needsUpdate = true;
                        // Reset plane to square
                        scene.remove(targetPlane);
                        targetPlane = new THREE.Mesh(new THREE.PlaneGeometry(1.2, 1.2), planeMaterial);
                        targetPlane.position.copy(CENTER);
                        scene.add(targetPlane);
                        return;
                    }
                    
                    const loader = new THREE.TextureLoader();
                    loader.crossOrigin = 'anonymous';
                    loader.load(url, (texture) => {
                        texture.minFilter = THREE.LinearFilter;
                        texture.magFilter = THREE.LinearFilter;
                        planeMaterial.map = texture;
                        planeMaterial.needsUpdate = true;
                        
                        // Adjust plane aspect ratio to match image
                        const img = texture.image;
                        if (img && img.width && img.height) {
                            const aspect = img.width / img.height;
                            const maxSize = 1.5;
                            let planeWidth, planeHeight;
                            if (aspect > 1) {
                                planeWidth = maxSize;
                                planeHeight = maxSize / aspect;
                            } else {
                                planeHeight = maxSize;
                                planeWidth = maxSize * aspect;
                            }
                            scene.remove(targetPlane);
                            targetPlane = new THREE.Mesh(
                                new THREE.PlaneGeometry(planeWidth, planeHeight),
                                planeMaterial
                            );
                            targetPlane.position.copy(CENTER);
                            scene.add(targetPlane);
                        }
                    }, undefined, (err) => {
                        console.error('Failed to load texture:', err);
                    });
                }
                
                // Check for initial imageUrl
                if (props.imageUrl) {
                    updateTextureFromUrl(props.imageUrl);
                }
                
                // Camera model
                const cameraGroup = new THREE.Group();
                const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 });
                const body = new THREE.Mesh(new THREE.BoxGeometry(0.3, 0.22, 0.38), bodyMat);
                cameraGroup.add(body);
                const lens = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.09, 0.11, 0.18, 16),
                    new THREE.MeshStandardMaterial({ color: 0x6699cc, metalness: 0.5, roughness: 0.3 })
                );
                lens.rotation.x = Math.PI / 2;
                lens.position.z = 0.26;
                cameraGroup.add(lens);
                scene.add(cameraGroup);
                
                // GREEN: Azimuth ring
                const azimuthRing = new THREE.Mesh(
                    new THREE.TorusGeometry(AZIMUTH_RADIUS, 0.04, 16, 64),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.3 })
                );
                azimuthRing.rotation.x = Math.PI / 2;
                azimuthRing.position.y = 0.05;
                scene.add(azimuthRing);
                
                const azimuthHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0x00ff88, emissive: 0x00ff88, emissiveIntensity: 0.5 })
                );
                azimuthHandle.userData.type = 'azimuth';
                scene.add(azimuthHandle);
                
                // PINK: Elevation arc
                const arcPoints = [];
                for (let i = 0; i <= 32; i++) {
                    const angle = THREE.MathUtils.degToRad(-30 + (90 * i / 32));
                    arcPoints.push(new THREE.Vector3(-0.8, ELEVATION_RADIUS * Math.sin(angle) + CENTER.y, ELEVATION_RADIUS * Math.cos(angle)));
                }
                const arcCurve = new THREE.CatmullRomCurve3(arcPoints);
                const elevationArc = new THREE.Mesh(
                    new THREE.TubeGeometry(arcCurve, 32, 0.04, 8, false),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.3 })
                );
                scene.add(elevationArc);
                
                const elevationHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xff69b4, emissive: 0xff69b4, emissiveIntensity: 0.5 })
                );
                elevationHandle.userData.type = 'elevation';
                scene.add(elevationHandle);
                
                // YELLOW: Distance line & handle
                const distanceLineGeo = new THREE.BufferGeometry();
                const distanceLine = new THREE.Line(distanceLineGeo, new THREE.LineBasicMaterial({ color: 0xffff00 }));
                scene.add(distanceLine);
                
                const distanceHandle = new THREE.Mesh(
                    new THREE.SphereGeometry(0.18, 16, 16),
                    new THREE.MeshStandardMaterial({ color: 0xffff00, emissive: 0xffff00, emissiveIntensity: 0.5 })
                );
                distanceHandle.userData.type = 'distance';
                scene.add(distanceHandle);
                
                function updatePositions() {
                    const distance = BASE_DISTANCE * distanceFactor;
                    const azRad = THREE.MathUtils.degToRad(azimuthAngle);
                    const elRad = THREE.MathUtils.degToRad(elevationAngle);
                    
                    const camX = distance * Math.sin(azRad) * Math.cos(elRad);
                    const camY = distance * Math.sin(elRad) + CENTER.y;
                    const camZ = distance * Math.cos(azRad) * Math.cos(elRad);
                    
                    cameraGroup.position.set(camX, camY, camZ);
                    cameraGroup.lookAt(CENTER);
                    
                    azimuthHandle.position.set(AZIMUTH_RADIUS * Math.sin(azRad), 0.05, AZIMUTH_RADIUS * Math.cos(azRad));
                    elevationHandle.position.set(-0.8, ELEVATION_RADIUS * Math.sin(elRad) + CENTER.y, ELEVATION_RADIUS * Math.cos(elRad));
                    
                    const orangeDist = distance - 0.5;
                    // 距离手柄固定在正前方，只受仰角和距离影响，不受方位角影响
                    distanceHandle.position.set(
                        0,
                        orangeDist * Math.sin(elRad) + CENTER.y,
                        orangeDist * Math.cos(elRad)
                    );
                    distanceLineGeo.setFromPoints([cameraGroup.position.clone(), CENTER.clone()]);
                    
                    // Update prompt
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    const distKey = distSnap === 1 ? '1' : distSnap.toFixed(1);
                    const prompt = '<sks> ' + azimuthNames[azSnap] + ' ' + elevationNames[String(elSnap)] + ' ' + distanceNames[distKey];
                    promptOverlay.textContent = prompt;
                }
                
                function updatePropsAndTrigger() {
                    const azSnap = snapToNearest(azimuthAngle, azimuthSteps);
                    const elSnap = snapToNearest(elevationAngle, elevationSteps);
                    const distSnap = snapToNearest(distanceFactor, distanceSteps);
                    
                    props.value = { azimuth: azSnap, elevation: elSnap, distance: distSnap };
                    trigger('change', props.value);
                }
                
                // Raycasting
                const raycaster = new THREE.Raycaster();
                const mouse = new THREE.Vector2();
                let isDragging = false;
                let dragTarget = null;
                let dragStartMouse = new THREE.Vector2();
                let dragStartDistance = 1.0;
                const intersection = new THREE.Vector3();
                
                const canvas = renderer.domElement;
                
                canvas.addEventListener('mousedown', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                        dragStartDistance = distanceFactor;
                        canvas.style.cursor = 'grabbing';
                    }
                });
                
                canvas.addEventListener('mousemove', (e) => {
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance + deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    } else {
                        raycaster.setFromCamera(mouse, camera);
                        const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                        [azimuthHandle, elevationHandle, distanceHandle].forEach(h => {
                            h.material.emissiveIntensity = 0.5;
                            h.scale.setScalar(1);
                        });
                        if (intersects.length > 0) {
                            intersects[0].object.material.emissiveIntensity = 0.8;
                            intersects[0].object.scale.setScalar(1.1);
                            canvas.style.cursor = 'grab';
                        } else {
                            canvas.style.cursor = 'default';
                        }
                    }
                });
                
                const onMouseUp = () => {
                    if (dragTarget) {
                        dragTarget.material.emissiveIntensity = 0.5;
                        dragTarget.scale.setScalar(1);
                        
                        // Snap and animate
                        const targetAz = snapToNearest(azimuthAngle, azimuthSteps);
                        const targetEl = snapToNearest(elevationAngle, elevationSteps);
                        const targetDist = snapToNearest(distanceFactor, distanceSteps);
                        
                        const startAz = azimuthAngle, startEl = elevationAngle, startDist = distanceFactor;
                        const startTime = Date.now();
                        
                        function animateSnap() {
                            const t = Math.min((Date.now() - startTime) / 200, 1);
                            const ease = 1 - Math.pow(1 - t, 3);
                            
                            let azDiff = targetAz - startAz;
                            if (azDiff > 180) azDiff -= 360;
                            if (azDiff < -180) azDiff += 360;
                            azimuthAngle = startAz + azDiff * ease;
                            if (azimuthAngle < 0) azimuthAngle += 360;
                            if (azimuthAngle >= 360) azimuthAngle -= 360;
                            
                            elevationAngle = startEl + (targetEl - startEl) * ease;
                            distanceFactor = startDist + (targetDist - startDist) * ease;
                            
                            updatePositions();
                            if (t < 1) requestAnimationFrame(animateSnap);
                            else updatePropsAndTrigger();
                        }
                        animateSnap();
                    }
                    isDragging = false;
                    dragTarget = null;
                    canvas.style.cursor = 'default';
                };
                
                canvas.addEventListener('mouseup', onMouseUp);
                canvas.addEventListener('mouseleave', onMouseUp);
                // Touch support for mobile
                canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    raycaster.setFromCamera(mouse, camera);
                    const intersects = raycaster.intersectObjects([azimuthHandle, elevationHandle, distanceHandle]);
                    
                    if (intersects.length > 0) {
                        isDragging = true;
                        dragTarget = intersects[0].object;
                        dragTarget.material.emissiveIntensity = 1.0;
                        dragTarget.scale.setScalar(1.3);
                        dragStartMouse.copy(mouse);
                        dragStartDistance = distanceFactor;
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const rect = canvas.getBoundingClientRect();
                    mouse.x = ((touch.clientX - rect.left) / rect.width) * 2 - 1;
                    mouse.y = -((touch.clientY - rect.top) / rect.height) * 2 + 1;
                    
                    if (isDragging && dragTarget) {
                        raycaster.setFromCamera(mouse, camera);
                        
                        if (dragTarget.userData.type === 'azimuth') {
                            const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -0.05);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                azimuthAngle = THREE.MathUtils.radToDeg(Math.atan2(intersection.x, intersection.z));
                                if (azimuthAngle < 0) azimuthAngle += 360;
                            }
                        } else if (dragTarget.userData.type === 'elevation') {
                            const plane = new THREE.Plane(new THREE.Vector3(1, 0, 0), -0.8);
                            if (raycaster.ray.intersectPlane(plane, intersection)) {
                                const relY = intersection.y - CENTER.y;
                                const relZ = intersection.z;
                                elevationAngle = THREE.MathUtils.clamp(THREE.MathUtils.radToDeg(Math.atan2(relY, relZ)), -30, 60);
                            }
                        } else if (dragTarget.userData.type === 'distance') {
                            const deltaY = mouse.y - dragStartMouse.y;
                            distanceFactor = THREE.MathUtils.clamp(dragStartDistance + deltaY * 1.5, 0.6, 1.4);
                        }
                        updatePositions();
                    }
                }, { passive: false });
                
                canvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                canvas.addEventListener('touchcancel', (e) => {
                    e.preventDefault();
                    onMouseUp();
                }, { passive: false });
                
                // Initial update
                updatePositions();
                
                // Render loop
                function render() {
                    requestAnimationFrame(render);
                    renderer.render(scene, camera);
                }
                render();
                
                // Handle resize
                new ResizeObserver(() => {
                    camera.aspect = wrapper.clientWidth / wrapper.clientHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(wrapper.clientWidth, wrapper.clientHeight);
                }).observe(wrapper);
                
                // Store update functions for external calls
                wrapper._updateFromProps = (newVal) => {
                    if (newVal && typeof newVal === 'object') {
                        azimuthAngle = newVal.azimuth ?? azimuthAngle;
                        elevationAngle = newVal.elevation ?? elevationAngle;
                        distanceFactor = newVal.distance ?? distanceFactor;
                        updatePositions();
                    }
                };
                
                wrapper._updateTexture = updateTextureFromUrl;
                
                // Watch for prop changes (imageUrl and value)
                let lastImageUrl = props.imageUrl;
                let lastValue = JSON.stringify(props.value);
                setInterval(() => {
                    // Check imageUrl changes
                    if (props.imageUrl !== lastImageUrl) {
                        lastImageUrl = props.imageUrl;
                        updateTextureFromUrl(props.imageUrl);
                    }
                    // Check value changes (from sliders)
                    const currentValue = JSON.stringify(props.value);
                    if (currentValue !== lastValue) {
                        lastValue = currentValue;
                        if (props.value && typeof props.value === 'object') {
                            azimuthAngle = props.value.azimuth ?? azimuthAngle;
                            elevationAngle = props.value.elevation ?? elevationAngle;
                            distanceFactor = props.value.distance ?? distanceFactor;
                            updatePositions();
                        }
                    }
                }, 100);
            };
            
            initScene();
        })();
        """
        
        super().__init__(
            value=value,
            html_template=html_template,
            js_on_load=js_on_load,
            imageUrl=imageUrl,
            **kwargs
        )


def stop_generate():
    global stop_generation
    stop_generation = True
    return "🛑 等待生成中止"


def _generate_common(
    mode, 
    prompt, 
    negative_prompt, 
    width, 
    height, 
    num_inference_steps, 
    batch_images, 
    true_cfg_scale, 
    seed_param, 
    transformer_dropdown, 
    lora_dropdown, 
    lora_weights, 
    res_vram, 
    image=None, 
    mask_image=None, 
    strength=None,
    size_edit2=None, 
    reserve_edit2=None,
):
    global mode_loaded, prompt_cache, negative_prompt_cache, stop_generation, prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, image_loaded
    results = []
    if seed_param < 0:
        seed = random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = seed_param
    if mode in ["editplus"]:
        CONDITION_IMAGE_SIZE = 384 * 384
        VAE_IMAGE_SIZE = 1024 * 1024
        image_processor = VaeImageProcessor(vae_scale_factor=16)
        if not isinstance(image, list):
            image = [image]
        calculated_images = []
        condition_images = []
        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            calculated_height = vae_height // 32 * 32
            calculated_width = vae_width // 32 * 32
            calculated_images.append(image_processor.resize(img, calculated_height, calculated_width))
            condition_images.append(image_processor.resize(img, condition_height, condition_width))
    if (mode != mode_loaded or prompt_cache != prompt or negative_prompt_cache != negative_prompt or 
        transformer_loaded != transformer_dropdown or lora_loaded != lora_dropdown or
          lora_loaded_weights != lora_weights or image_loaded!=image):
        load_model(mode, transformer_dropdown, lora_dropdown, lora_weights, res_vram)
        prompt_cache, negative_prompt_cache, image_loaded = prompt, negative_prompt, image
        if mode == "t2i" or mode == "i2i" or mode == "inp" or mode == "con":
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(prompt)
            if true_cfg_scale > 1:
                negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(negative_prompt)
        elif mode in ["editplus"]:
            prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(image=condition_images, prompt=prompt)
            if true_cfg_scale > 1:
                negative_prompt_embeds, negative_prompt_embeds_mask = pipe.encode_prompt(image=condition_images, prompt=negative_prompt)
    for i in range(batch_images):
        if stop_generation:
            stop_generation = False
            yield results, f"✅ 生成已中止，最后种子数{seed+i-1}"
            break
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}.png"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("mode", f"{mode}\n")
        if image:
            pnginfo.add_text("image", f"{str(image)}\n")
        pnginfo.add_text("prompt", f"{prompt}\n")
        pnginfo.add_text("negative_prompt", f"{negative_prompt}\n")
        if width and mode != "edit2":
            pnginfo.add_text("width", f"{str(width)}\n")
        if height:
            pnginfo.add_text("height", f"{str(height)}\n")
        pnginfo.add_text("num_inference_steps", f"{str(num_inference_steps)}\n")
        pnginfo.add_text("true_cfg_scale", f"{str(true_cfg_scale)}\n")
        pnginfo.add_text("seed", f"{str(seed + i)}\n")
        pnginfo.add_text("models", f"{transformer_dropdown}\n")
        lora_str = ", ".join(lora_dropdown) if lora_dropdown else ""
        pnginfo.add_text("lora", f"{lora_str}\n")
        lora_weights_str = ", ".join(lora_weights) if lora_weights else ""
        pnginfo.add_text("lora_weights", f"{lora_weights_str}\n")
        if strength:
            pnginfo.add_text("strength", f"{str(strength)}\n")
        with torch.no_grad():
            if mode == "t2i":
                output = pipe(
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if true_cfg_scale > 1 else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if true_cfg_scale > 1 else None,
                    generator=torch.Generator().manual_seed(seed + i),
                )
            elif mode == "i2i":
                output = pipe(
                    image=image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    true_cfg_scale=true_cfg_scale,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if true_cfg_scale > 1 else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if true_cfg_scale > 1 else None,
                    generator=torch.Generator().manual_seed(seed + i),
                )
            elif mode == "inp":
                output = pipe(
                    image=image,
                    mask_image=mask_image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    true_cfg_scale=true_cfg_scale,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if true_cfg_scale > 1 else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if true_cfg_scale > 1 else None,
                    generator=torch.Generator().manual_seed(seed + i),
                )
            elif mode == "con":
                output = pipe(
                    control_image=image,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    controlnet_conditioning_scale=strength,
                    true_cfg_scale=true_cfg_scale,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if true_cfg_scale > 1 else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if true_cfg_scale > 1 else None,
                    generator=torch.Generator().manual_seed(seed + i),
                )
            elif mode == "editplus":
                output = pipe(
                    image=calculated_images,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=true_cfg_scale,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds if true_cfg_scale > 1 else None,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask if true_cfg_scale > 1 else None,
                    generator=torch.Generator().manual_seed(seed + i),
                )
        image = output.images[0]
        image.save(filename, pnginfo=pnginfo)
        results.append(image)
        yield results, f"种子数{seed+i}，保存地址{filename}"
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

#, progress=gr.Progress(track_tqdm=True)
def generate_t2i(prompt, negative_prompt, width, height, num_inference_steps, 
                 batch_images, true_cfg_scale, seed_param, transformer_dropdown, 
                 lora_dropdown, lora_weights, res_vram):
    if "ModelScope" in transformer_dropdown:
        yield from modelscope_generate(
            mode="t2i_ms",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            batch_images=batch_images, 
            seed_param=seed_param,
            transformer_dropdown=transformer_dropdown,
        )
    else:
        yield from _generate_common(
            mode="t2i",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            batch_images=batch_images,
            true_cfg_scale=true_cfg_scale,
            seed_param=seed_param,
            transformer_dropdown=transformer_dropdown,
            lora_dropdown=lora_dropdown,
            lora_weights=lora_weights,
            res_vram=res_vram
        )


def generate_i2i(image, prompt, negative_prompt, width, height, num_inference_steps,
                 strength, batch_images, true_cfg_scale, seed_param, transformer_dropdown, 
                 lora_dropdown, lora_weights, res_vram):
    image = load_image(image)
    yield from _generate_common(
        mode="i2i",
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        strength=strength,
        batch_images=batch_images,
        true_cfg_scale=true_cfg_scale,
        seed_param=seed_param,
        transformer_dropdown=transformer_dropdown,
        lora_dropdown=lora_dropdown,
        lora_weights=lora_weights,
        res_vram=res_vram
    )


def generate_inp(image, prompt, negative_prompt, width, height, num_inference_steps,
                 strength, batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                 lora_dropdown, lora_weights, res_vram):
    # 处理蒙版图像
    mask_image = image["layers"][0]
    mask_image = mask_image .convert("RGBA")
    data = np.array(mask_image)
    # 修改蒙版颜色（黑色->白色，透明->黑色）
    black_pixels = (data[:, :, 0] == 0) & (data[:, :, 1] == 0) & (data[:, :, 2] == 0)
    data[black_pixels, :3] = [255, 255, 255]
    transparent_pixels = (data[:, :, 3] == 0)
    data[transparent_pixels, :3] = [0, 0, 0]
    mask_image = Image.fromarray(data)
    # 提取背景图像
    background_image = load_image(image["background"])
    yield from _generate_common(
        mode="inp",
        image=background_image,
        mask_image=mask_image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        strength=strength,
        batch_images=batch_images,
        true_cfg_scale=true_cfg_scale,
        seed_param=seed_param,
        transformer_dropdown=transformer_dropdown,
        lora_dropdown=lora_dropdown,
        lora_weights=lora_weights,
        res_vram=res_vram
    )


def generate_con(image, prompt, negative_prompt, width, height, num_inference_steps,
                 strength, batch_images, true_cfg_scale, seed_param, transformer_dropdown, 
                 lora_dropdown, lora_weights, res_vram):
    image = load_image(image)
    yield from _generate_common(
        mode="con",
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        strength=strength,
        batch_images=batch_images,
        true_cfg_scale=true_cfg_scale,
        seed_param=seed_param,
        transformer_dropdown=transformer_dropdown,
        lora_dropdown=lora_dropdown,
        lora_weights=lora_weights,
        res_vram=res_vram
    )


def generate_editplus2(image_editplus2, image_editplus3, image_editplus4, image_editplus5, prompt, negative_prompt, width, height, num_inference_steps,
                  batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                  lora_dropdown, lora_weights, res_vram):
    if "ModelScope" in transformer_dropdown:
        yield from modelscope_generate(
            mode="edit_ms",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=None,
            height=None,
            batch_images=batch_images, 
            seed_param=seed_param,
            transformer_dropdown=transformer_dropdown,
            image=image_editplus2,
        )
    else:
        image = [image_editplus2, image_editplus3, image_editplus4, image_editplus5]
        image = [img for img in image if img is not None]
        images = []  # 用于存储所有处理后的图片
        for img in image:  # 遍历图片地址列表
            # 转换为RGBA
            img = img.convert("RGBA")
            # 创建白色背景
            white_bg = Image.new("RGB", img.size, (255, 255, 255))
            # 使用alpha通道作为掩码进行粘贴
            white_bg.paste(img, mask=img.split()[3])
            # 转换为RGB
            img_rgb = white_bg.convert("RGB")
            # 添加到结果列表
            images.append(img_rgb)
        yield from _generate_common(
            mode="editplus",
            image=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            batch_images=batch_images,
            true_cfg_scale=true_cfg_scale,
            seed_param=seed_param,
            transformer_dropdown=transformer_dropdown, 
            lora_dropdown=lora_dropdown,
            lora_weights=lora_weights,
            res_vram=res_vram
        )


def generate_camera_edit(image_camera, azimuth, elevation, distance, negative_prompt, width, height, num_inference_steps,
                  batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                  lora_dropdown, lora_weights, res_vram, additional_prompt=""):
    """
    Edit the camera angle of an image using Qwen Image Edit Plus with multi-angles LoRA.
    """
    if image_camera is None:
        raise gr.Error("请先上传图片")
    
    # Build camera prompt
    camera_prompt = build_camera_prompt(azimuth, elevation, distance)
    
    # Merge additional prompt if provided
    if additional_prompt and additional_prompt.strip():
        camera_prompt = f"{camera_prompt}, {additional_prompt.strip()}"
    
    # Convert image to RGB
    pil_image = image_camera.convert("RGB") if isinstance(image_camera, Image.Image) else Image.open(image_camera).convert("RGB")
    
    # Use editplus mode with camera prompt
    yield from _generate_common(
        mode="editplus",
        image=[pil_image],
        prompt=camera_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        batch_images=batch_images,
        true_cfg_scale=true_cfg_scale,
        seed_param=seed_param,
        transformer_dropdown=transformer_dropdown, 
        lora_dropdown=lora_dropdown,
        lora_weights=lora_weights,
        res_vram=res_vram
    )


def generate_cont(image, processor_id):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"outputs/{timestamp}.png"
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
    processed_image.save(filename)
    yield processed_image, f"✅ 预处理完成,保存地址{filename}"


def change_reference_count(reference_count):
    if reference_count == 0:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    elif reference_count == 1:
        return gr.update(visible=True), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    elif reference_count == 2:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False, value=None)
    elif reference_count == 3:
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def convert_lora(lora_in):
    global pipe
    results = []
    for lora_path in lora_in:
        # 读取LoRA文件
        pipe = safetensors.torch.load_file(lora_path, device="cpu")
        # 打印所有key值
        """print("LoRA文件包含的key值:")
        for key in pipe.keys():
            print(f"  {key}")"""
        converted_dict = {}
        for key, value in pipe.items():
            if 'lora' not in key:
                continue
            elif 'alpha' in key:
                continue
            fixed_key = key
            if fixed_key.endswith(".lora_A.default.weight"):
                fixed_key = fixed_key.replace(".lora_A.default.weight", ".lora.down.weight")
            elif fixed_key.endswith(".lora_B.default.weight"):
                fixed_key = fixed_key.replace(".lora_B.default.weight", ".lora.up.weight")
            elif fixed_key.endswith(".lora_A.weight"):
                fixed_key = fixed_key.replace(".lora_A.weight", ".lora.down.weight") 
            elif fixed_key.endswith(".lora_B.weight"):
                fixed_key = fixed_key.replace(".lora_B.weight", ".lora.up.weight")
            elif fixed_key.endswith(".lora_down.weight"):
                fixed_key = fixed_key.replace(".lora_down.weight", ".lora.down.weight")
            elif fixed_key.endswith(".lora_up.weight"):
                fixed_key = fixed_key.replace(".lora_up.weight", ".lora.up.weight")
    
            if fixed_key.startswith("diffusion_model.transformer_blocks."):
                fixed_key = fixed_key.replace("diffusion_model.transformer_blocks.", "transformer.transformer_blocks.")
            elif fixed_key.startswith("lora_unet_transformer_blocks_"):
                fixed_key = fixed_key.replace("lora_unet_transformer_blocks_", "transformer.transformer_blocks.")
                fixed_key = fixed_key.replace("_attn_", ".attn.")
                fixed_key = fixed_key.replace("_img_mlp_net_", ".img_mlp.net.")
                fixed_key = fixed_key.replace("_img_mod_", ".img_mod.")
                fixed_key = fixed_key.replace("_txt_mlp_net_", ".txt_mlp.net.")
                fixed_key = fixed_key.replace("_txt_mod_", ".txt_mod.")
                fixed_key = fixed_key.replace("0_", "0.")
                fixed_key = fixed_key.replace("_0", ".0")
            elif fixed_key.startswith("transformer_blocks."):
                fixed_key = "transformer." + fixed_key
            converted_dict[fixed_key] = value

        base_name = os.path.splitext(os.path.basename(lora_path))[0]
        output_filename = f"{base_name}_diffusers.safetensors"
        output_path = os.path.join("models/lora", output_filename)
        safetensors.torch.save_file(converted_dict, output_path)
        results.append(output_path)
        yield results, f"✅ {output_filename}转换完成"
    pipe = None
    yield results, f"✅ 全部转换完成，请点击刷新模型"


def load_image_info(selected_index, gallery):
    img = Image.open(gallery[selected_index][0])
    # 读取PNG文本信息块
    if img.format == 'PNG' and hasattr(img, 'text'):
        info = "".join([f"{k}: {v}" for k, v in img.text.items()])
    else:
        info = "None"
    return gr.update(value=info) 


def save_openai_config(transformer_dropdown, transformer_dropdown2, res_vram_tb, base_url_tb, api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb):
    global res_vram, base_url, api_key, model_name, temperature, top_p, max_tokens, modelscope_api_key, openai_base_url, openai_api_key
    res_vram, base_url, api_key, model_name, temperature, top_p, max_tokens, modelscope_api_key = res_vram_tb, base_url_tb, api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb
    openai_base_url, openai_api_key = base_url_tb, api_key_tb
    
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
        "TRANSFORMER_DROPDOWN": transformer_dropdown,
        "TRANSFORMER_DROPDOWN2": transformer_dropdown2,
        "RES_VRAM": res_vram_tb,
        "OPENAI_BASE_URL": base_url_tb,
        "OPENAI_API_KEY": api_key_tb,
        "MODEL_NAME": model_name_tb,
        "TEMPERATURE": temperature_tb,
        "TOP_P": top_p_tb,
        "MAX_TOKENS": max_tokens_tb,
        "MODELSCOPE_API_KEY": modelscope_api_key_tb,
    }
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return "✅ 配置已保存到本地文件"

# 解决冲突端口（感谢licyk酱提供的代码~）
def find_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        if s.connect_ex(("localhost", port)) == 0:
            print(f"端口 {port} 已被占用，正在寻找可用端口...")
            return find_port(port=port + 1)
        else:
            return port


def load_gallery():
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
    try:
        file_paths, info = load_gallery()
        return file_paths, info  # 现在返回的是文件路径列表
    except Exception as e:
        return [], f"❌ 加载图库时出错: {str(e)}"
    

def update_selection(selected_state: gr.SelectData):
    return selected_state.index


def load_examples(tab_name):
    if os.path.exists(EXAMPLES_FILE):
        with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
            examples_data = json.load(f)
            # 返回对应TabItem的示例列表，如果不存在则返回空列表
            return examples_data.get(tab_name, [])
    return []


def save_example(prompt, tab_name):
    # 读取现有数据或初始化空字典
    if os.path.exists(EXAMPLES_FILE):
        with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
            examples_data = json.load(f)
    else:
        examples_data = {}
    
    # 确保当前TabItem有列表
    if tab_name not in examples_data:
        examples_data[tab_name] = []
    
    # 添加新示例（如果不存在）
    if prompt and prompt not in examples_data[tab_name]:
        examples_data[tab_name].append(prompt)
    
    # 保存更新后的数据
    with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
        json.dump(examples_data, f, ensure_ascii=False, indent=2)
    
    return gr.update(choices=examples_data[tab_name]), "✅ 提示词已保存"


def scale_resolution_1_5(width, height):
    """
    将宽度和高度都放大1.5倍，并按照16的倍数向下取整
    """
    new_width = int(width * 1.5) // 16 * 16
    new_height = int(height * 1.5) // 16 * 16
    return new_width, new_height, "✅ 分辨率已调整为1.5倍"

css = """
.icon-btn {
    min-width: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
#camera-3d-control { min-height: 450px; }
.slider-row { display: flex; gap: 10px; align-items: center; }
"""


with gr.Blocks() as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">通臂 Tongbi</h2>
            </div>
            <div style="text-align: center;">
                十字鱼
                <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                |Tongbi
                <a href="https://github.com/gluttony-10/Tongbi">🌐github</a> 
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ 该演示仅供学术研究和体验使用。
            </div>
            """)
    with gr.Accordion("模型设置", open=False):
        with gr.Column():
            with gr.Row():
                refresh_button = gr.Button("🔄 刷新模型", scale=0)
            with gr.Row():
                transformer_dropdown = gr.Dropdown(label="QI模型", info="存放基础模型到models/transformer，仅支持mmgp转化版本", choices=transformer_choices, value=transformer_)
                transformer_dropdown2 = gr.Dropdown(label="QIEP模型", info="存放编辑模型到models/transformer，仅支持mmgp转化版本", choices=transformer_choices2, value=transformer_2)
                lora_dropdown = gr.Dropdown(label="LoRA模型", info="存放LoRA模型到models/lora，可多选", choices=lora_choices, multiselect=True)
                lora_weights = gr.Textbox(label="LoRA权重", info="Lora权重，多个权重请用英文逗号隔开。例如：0.8,0.5,0.2", value="")
    with gr.TabItem("文生图"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button = gr.Button("🎬 开始生成", variant='primary', scale=5)
                    enhance_button = gr.Button("提示词增强", scale=1)
                    save_example_button = gr.Button("💾", elem_classes="icon-btn")
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("推荐分辨率：1328x1328、1664x928、1472x1104")
                    with gr.Row():
                        width = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 交换宽高")
                        scale_1_5_button = gr.Button("1.5倍分辨率")
                    batch_images = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                    examples_dropdown = gr.Dropdown(
                        label="提示词库", 
                        choices=load_examples("t2i"),
                        interactive=True,
                        scale=5
                    )
            with gr.Column():
                info = gr.Textbox(label="提示信息", interactive=False)
                image_output = gr.Gallery(label="生成结果", interactive=False)
                stop_button = gr.Button("中止生成", variant="stop")
    with gr.TabItem("图生图"):
        with gr.Row():
            with gr.Column():
                image_i2i = gr.Image(label="输入图片", type="pil", height=400)
                prompt_i2i = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt_i2i = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button_i2i = gr.Button("🎬 开始生成", variant='primary', scale=4)
                    enhance_button_i2i = gr.Button("提示词增强", scale=1)
                    reverse_button_i2i = gr.Button("反推提示词", scale=1)
                    save_example_button_i2i = gr.Button("💾", elem_classes="icon-btn")
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_i2i = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height_i2i = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button_i2i = gr.Button("🔄 交换宽高")
                        scale_1_5_button_i2i = gr.Button("1.5倍分辨率")
                    strength_i2i = gr.Slider(label="strength", minimum=0, maximum=1, step=0.01, value=0.5)
                    batch_images_i2i = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_i2i = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_i2i = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_i2i = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                    examples_dropdown_i2i = gr.Dropdown(
                        label="提示词库", 
                        choices=load_examples("i2i"),
                        interactive=True,
                        scale=5
                    )
            with gr.Column():
                info_i2i = gr.Textbox(label="提示信息", interactive=False)
                image_output_i2i = gr.Gallery(label="生成结果", interactive=False)
                stop_button_i2i = gr.Button("中止生成", variant="stop")
    with gr.TabItem("局部重绘"):
        with gr.Row():
            with gr.Column():
                image_inp = gr.ImageMask(label="输入蒙版", type="pil", height=400)
                prompt_inp = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt_inp = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button_inp = gr.Button("🎬 开始生成", variant='primary', scale=4)
                    enhance_button_inp = gr.Button("提示词增强", scale=1)
                    reverse_button_inp = gr.Button("反推提示词", scale=1)
                    save_example_button_inp = gr.Button("💾", elem_classes="icon-btn")
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_inp = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height_inp = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button_inp = gr.Button("🔄 交换宽高")
                        scale_1_5_button_inp = gr.Button("1.5倍分辨率")
                    strength_inp = gr.Slider(label="strength", minimum=0, maximum=1, step=0.01, value=0.8)
                    batch_images_inp = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_inp = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_inp = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_inp = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                    examples_dropdown_inp = gr.Dropdown(
                        label="提示词库", 
                        choices=load_examples("inp"),
                        interactive=True,
                        scale=5
                    )
            with gr.Column():
                info_inp = gr.Textbox(label="提示信息", interactive=False)
                image_output_inp = gr.Gallery(label="生成结果", interactive=False)
                stop_button_inp = gr.Button("中止生成", variant="stop")
    with gr.TabItem("多图编辑"):
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    image_editplus2 = gr.Image(label="输入图片", type="pil", height=300, image_mode="RGBA")
                    image_editplus3 = gr.Image(label="输入图片", type="pil", height=300, image_mode="RGBA", visible=False)
                    image_editplus4 = gr.Image(label="输入图片", type="pil", height=300, image_mode="RGBA", visible=False)
                    image_editplus5 = gr.Image(label="输入图片", type="pil", height=300, image_mode="RGBA", visible=False)
                reference_count = gr.Slider(
                    label="参考图数量", 
                    minimum=0, 
                    maximum=3, 
                    step=1, 
                    value=0,
                )
                prompt_editplus2 = gr.Textbox(label="提示词", value="给左边的女孩换上右边的衣服")
                negative_prompt_editplus2 = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button_editplus2 = gr.Button("🎬 开始生成", variant='primary', scale=4)
                    enhance_button_editplus2 = gr.Button("提示词增强", scale=1)
                    reverse_button_editplus2 = gr.Button("反推提示词", scale=1)
                    save_example_button_editplus2 = gr.Button("💾", elem_classes="icon-btn")
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_editplus2 = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1024)
                        height_editplus2 = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1024)
                    with gr.Row():
                        exchange_button_editplus2 = gr.Button("🔄 交换宽高")
                        scale_1_5_button_editplus2 = gr.Button("1.5倍分辨率")
                    batch_images_editplus2 = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_editplus2 = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_editplus2 = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_editplus2 = gr.Number(label="种子，请输入自然数，-1为随机", value=0)
                    examples_dropdown_editplus2 = gr.Dropdown(
                        label="提示词库", 
                        choices=load_examples("editplus"),
                        interactive=True,
                        scale=5
                    )
            with gr.Column():
                info_editplus2 = gr.Textbox(label="提示信息", interactive=False)
                image_output_editplus2 = gr.Gallery(label="生成结果", interactive=False)
                stop_button_editplus2 = gr.Button("中止生成", variant="stop")
    with gr.TabItem("3D相机控制"):
        with gr.Row():
            # Left column: Input image and controls
            with gr.Column(scale=1):
                with gr.Row():
                    image_camera = gr.Image(label="输入图片", type="pil", height=500)
                    with gr.Column():
                        camera_3d = CameraControl3D(
                            value={"azimuth": 0, "elevation": 0, "distance": 1.0},
                            elem_id="camera-3d-control"
                        )
                        gr.Markdown("*拖动彩色手柄：🟢 方位角，🩷 仰角，🟡 距离（上远下近）*")
                with gr.Row():
                    run_btn_camera = gr.Button("🚀 生成", variant="primary", scale=4)
                    enhance_button_camera = gr.Button("提示词增强", scale=1)
                    reverse_button_camera = gr.Button("反推提示词", scale=1)
                    save_example_button_camera = gr.Button("💾", elem_classes="icon-btn")
                with gr.Accordion("滑块控制", open=True):
                    azimuth_slider = gr.Slider(
                        label="方位角（水平旋转）",
                        minimum=0,
                        maximum=315,
                        step=45,
                        value=0,
                        info="0°=正面，90°=右侧，180°=背面，270°=左侧"
                    )
                    elevation_slider = gr.Slider(
                        label="仰角（垂直角度）", 
                        minimum=-30,
                        maximum=60,
                        step=30,
                        value=0,
                        info="-30°=低角度，0°=平视，60°=高角度"
                    )
                    distance_slider = gr.Slider(
                        label="距离",
                        minimum=0.6,
                        maximum=1.4,
                        step=0.4,
                        value=1.0,
                        info="0.6=特写，1.0=中景，1.4=全景"
                    )
                    prompt_preview_camera = gr.Textbox(
                        label="生成的提示词",
                        value="<sks> front view eye-level shot medium shot",
                        interactive=False
                    )
                    additional_prompt_camera = gr.Textbox(
                        label="附加提示词",
                        value="",
                        placeholder="可在此添加额外的提示词，将自动合并到生成的提示词中",
                        interactive=True
                    )
                    negative_prompt_camera = gr.Textbox(label="负面提示词", value="")
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_camera = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1024)
                        height_camera = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1024)
                    with gr.Row():
                        exchange_button_camera = gr.Button("🔄 交换宽高")
                        scale_1_5_button_camera = gr.Button("1.5倍分辨率")
                    batch_images_camera = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_camera = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_camera = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_camera = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
                    examples_dropdown_camera = gr.Dropdown(
                        label="提示词库", 
                        choices=load_examples("camera"),
                        interactive=True,
                        scale=5
                    )
            
            # Right column: Output
            with gr.Column(scale=1):
                info_camera = gr.Textbox(label="提示信息", interactive=False)
                result_camera = gr.Gallery(label="生成结果", interactive=False)
                stop_button_camera = gr.Button("中止生成", variant="stop")
    with gr.TabItem("ControlNet预处理"):
        with gr.TabItem("图片预处理"):
            with gr.Row():
                with gr.Column():
                    image_cont = gr.Image(label="输入图片", type="pil", height=400)
                    processor_cont = gr.Dropdown(label="预处理", choices=[
                        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", 
                        "lineart_anime", "lineart_coarse", "lineart_realistic", "mediapipe_face", 
                        "mlsd", "normal_bae", "openpose", "openpose_face", 
                        "openpose_faceonly", "openpose_full", "openpose_hand", "scribble_hed", 
                        "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe", 
                        "softedge_pidinet", "softedge_pidsafe"])
                    generate_button_cont = gr.Button("🎬 开始生成", variant='primary', scale=4)
                with gr.Column():
                    info_cont = gr.Textbox(label="提示信息", interactive=False)
                    image_output_cont = gr.Image(label="生成结果", interactive=False)
                    with gr.Row():
                        send_to_i2i = gr.Button("发送到图生图", scale=1)
                        send_to_inp = gr.Button("发送到局部重绘", scale=1)
                    with gr.Row():
                        send_to_edit2 = gr.Button("发送到多图编辑1", scale=1)
                        send_to_edit3 = gr.Button("发送到多图编辑2", scale=1)
                        send_to_edit4 = gr.Button("发送到多图编辑3", scale=1)
                        send_to_edit5 = gr.Button("发送到多图编辑4", scale=1)
        with gr.TabItem("Open Pose Editor"):
            gr.HTML('<iframe src="https://zhuyu1997.github.io/open-pose-editor/" width="100%" height="800px" frameborder="0" style="border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);"></iframe>')
    with gr.TabItem("转换lora"):
        with gr.Row():
            with gr.Column():
                lora_in = gr.File(label="上传lora文件，可多选", type="filepath", file_count="multiple")
                convert_button = gr.Button("开始转换", variant='primary')
            with gr.Column():
                info_lora = gr.Textbox(label="提示信息", interactive=False)
                lora_out = gr.File(label="输出文件", type="filepath", interactive=False)
                gr.Markdown("可转化lora为diffusers可以使用的lora，比如转化[魔搭](https://modelscope.cn/aigc/modelTraining)训练的lora。")
    with gr.TabItem("图库"):
        with gr.Row():
            with gr.Column(scale=3):
                refresh_gallery_button = gr.Button("🔄 刷新图库")
                gallery = gr.Gallery(label="图库", columns=4, height="auto", object_fit="cover")
                selected_index = gr.Number(value=-1, visible=False)
            with gr.Column(scale=2):
                gallery_info = gr.Textbox(label="提示信息", interactive=False)
                info_info = gr.Textbox(label="图片信息", lines=20, interactive=False)
        with gr.Row():
            send_to_i2i_gallery = gr.Button("发送到图生图")
            send_to_inp_gallery = gr.Button("发送到局部重绘")
            send_to_edit2_gallery = gr.Button("发送到多图编辑1")
            send_to_edit3_gallery = gr.Button("发送到多图编辑2")
            send_to_edit4_gallery = gr.Button("发送到多图编辑3")
            send_to_edit5_gallery = gr.Button("发送到多图编辑4")
            send_to_cont_gallery = gr.Button("发送到ControlNet预处理")
    with gr.TabItem("设置"):
        with gr.Row():
            with gr.Column():
                res_vram_tb = gr.Slider(label="保留显存", info="单位MB，数值越大，显存占用越小，速度越慢", minimum=0, maximum=80000, step=1, value=res_vram)
                with gr.Accordion("多模态API设置", open=True):
                    openai_base_url_tb = gr.Textbox(label="BASE URL", info="请输入BASE URL，例如：https://open.bigmodel.cn/api/paas/v4", value=openai_base_url)
                    openai_api_key_tb = gr.Textbox(label="API KEY", info="请输入API KEY，暗文显示", value=openai_api_key, type="password")
                    with gr.Row():
                        model_name_tb = gr.Textbox(label="MODEL NAME", info="请输入模型名称，需要支持图片输入的多模态模型，例如：GLM-4.5V", value=model_name)
                        temperature_tb = gr.Slider(label="temperature", info="采样温度，控制输出的随机性和创造性", minimum=0, maximum=1, step=0.1, value=temperature)
                    with gr.Row():
                        top_p_tb = gr.Slider(label="top_p", info="核采样（nucleus sampling）参数，是temperature采样的替代方法", minimum=0, maximum=1, step=0.1, value=top_p)
                        max_tokens_tb = gr.Slider(label="max_tokens", info="模型输出的最大令牌（token）数量限制", minimum=1024, maximum=65536, step=1024, value=max_tokens)
                with gr.Accordion("在线生图API设置", open=True):
                    modelscope_api_key_tb = gr.Textbox(label="魔搭的API KEY", info="使用魔搭在线模型时需要，获取地址https://modelscope.cn/my/myaccesstoken", value=modelscope_api_key, type="password")
            with gr.Column():
                info_config = gr.Textbox(label="提示信息", value="修改后请点击保存设置生效。", interactive=False)
                save_button = gr.Button("保存设置", variant='primary')
                gr.Markdown("""多模态API设置支持通用类OPENAI的API，请使用多模态模型，如：GLM-4.5V、GLM-4.1V-Thinking-Flash等（需要支持base64）。
                            可申请[智谱API](https://www.bigmodel.cn/invite?icode=eKq1YoHsX6y4VhGIPJuOPGczbXFgPRGIalpycrEwJ28%3D)。
                            temperature、top_p和max_tokens三个值，默认是GLM-4.5V的推荐值。
                            如果更换模型，请自行修改。
                            保存设置除了保存此页面的设置，还会保存QI基础模型和QI编辑模型的设置。
                            """)
    # 模型设置
    refresh_button.click(
        fn=refresh_model,
        inputs=[],
        outputs=[transformer_dropdown, transformer_dropdown2, lora_dropdown]
    )
    # 文生图
    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit],
        fn = generate_t2i,
        inputs = [
            prompt,
            negative_prompt,
            width,
            height,
            num_inference_steps,
            batch_images,
            true_cfg_scale, 
            seed_param,
            transformer_dropdown,
            lora_dropdown, 
            lora_weights,
            res_vram_tb,
        ],
        outputs = [image_output, info]
    )
    enhance_button.click(
        fn=enhance_prompt, 
        inputs=[prompt], 
        outputs=[prompt, info]
    )
    exchange_button.click(
        fn=exchange_width_height, 
        inputs=[width, height], 
        outputs=[width, height, info]
    )
    scale_1_5_button.click(
        fn=scale_resolution_1_5,
        inputs=[width, height],
        outputs=[width, height, info]
    )
    save_example_button.click(
        fn=lambda prompt: save_example(prompt, "t2i"),
        inputs=[prompt],
        outputs=[examples_dropdown, info]
    )
    examples_dropdown.change(
        fn=lambda selected_example, current_prompt: f"{current_prompt} {selected_example.strip()}",
        inputs=[examples_dropdown, prompt],
        outputs=[prompt]
    )
    stop_button.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info]
    )
    # 图生图
    gr.on(
        triggers=[generate_button_i2i.click, prompt_i2i.submit, negative_prompt_i2i.submit],
        fn = generate_i2i,
        inputs = [
            image_i2i,
            prompt_i2i,
            negative_prompt_i2i,
            width_i2i,
            height_i2i,
            num_inference_steps_i2i,
            strength_i2i,
            batch_images_i2i,
            true_cfg_scale_i2i, 
            seed_param_i2i,
            transformer_dropdown,
            lora_dropdown, 
            lora_weights,
            res_vram_tb,
        ],
        outputs = [image_output_i2i, info_i2i]
    )
    enhance_button_i2i.click(
        fn=enhance_prompt, 
        inputs=[prompt_i2i], 
        outputs=[prompt_i2i, info_i2i]
    )
    reverse_button_i2i.click(
        fn=enhance_prompt, 
        inputs=[prompt_i2i, image_i2i], 
        outputs=[prompt_i2i, info_i2i]
    )
    exchange_button_i2i.click(
        fn=exchange_width_height, 
        inputs=[width_i2i, height_i2i], 
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    scale_1_5_button_i2i.click(
        fn=scale_resolution_1_5,
        inputs=[width_i2i, height_i2i],
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    image_i2i.upload(
        fn=adjust_width_height, 
        inputs=[image_i2i], 
        outputs=[width_i2i, height_i2i, info_i2i]
    )
    save_example_button_i2i.click(
        fn=lambda prompt: save_example(prompt, "i2i"),
        inputs=[prompt_i2i],
        outputs=[examples_dropdown_i2i, info_i2i]
    )
    examples_dropdown_i2i.change(
        fn=lambda selected_example, current_prompt: f"{current_prompt} {selected_example.strip()}",
        inputs=[examples_dropdown_i2i, prompt_i2i],
        outputs=[prompt_i2i]
    )
    stop_button_i2i.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_i2i]
    )
    # 局部重绘
    gr.on(
        triggers=[generate_button_inp.click, prompt_inp.submit, negative_prompt_inp.submit],
        fn = generate_inp,
        inputs = [
            image_inp,
            prompt_inp,
            negative_prompt_inp,
            width_inp,
            height_inp,
            num_inference_steps_inp,
            strength_inp,
            batch_images_inp,
            true_cfg_scale_inp, 
            seed_param_inp,
            transformer_dropdown,
            lora_dropdown, 
            lora_weights,
            res_vram_tb,
        ],
        outputs = [image_output_inp, info_inp]
    )
    enhance_button_inp.click(
        fn=enhance_prompt, 
        inputs=[prompt_inp], 
        outputs=[prompt_inp, info_inp]
    )
    reverse_button_inp.click(
        fn=enhance_prompt, 
        inputs=[prompt_inp, image_inp], 
        outputs=[prompt_inp, info_inp]
    )
    exchange_button_inp.click(
        fn=exchange_width_height, 
        inputs=[width_inp, height_inp], 
        outputs=[width_inp, height_inp, info_inp]
    )
    scale_1_5_button_inp.click(
        fn=scale_resolution_1_5,
        inputs=[width_inp, height_inp],
        outputs=[width_inp, height_inp, info_inp]
    )
    image_inp.upload(
        fn=adjust_width_height, 
        inputs=[image_inp], 
        outputs=[width_inp, height_inp, info_inp]
    )
    save_example_button_inp.click(
        fn=lambda prompt: save_example(prompt, "inp"),
        inputs=[prompt_inp],
        outputs=[examples_dropdown_inp, info_inp]
    )
    examples_dropdown_inp.change(
        fn=lambda selected_example, current_prompt: f"{current_prompt} {selected_example.strip()}",
        inputs=[examples_dropdown_inp, prompt_inp],
        outputs=[prompt_inp]
    )
    stop_button_inp.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_inp]
    )
    # 多图编辑
    reference_count.change(
        fn=change_reference_count,
        inputs=[reference_count],
        outputs=[image_editplus3, image_editplus4, image_editplus5]
    )
    gr.on(
        triggers=[generate_button_editplus2.click, prompt_editplus2.submit, negative_prompt_editplus2.submit],
        fn = generate_editplus2,
        inputs = [
            image_editplus2,
            image_editplus3,
            image_editplus4,
            image_editplus5,
            prompt_editplus2,
            negative_prompt_editplus2,
            width_editplus2,
            height_editplus2,
            num_inference_steps_editplus2,
            batch_images_editplus2,
            true_cfg_scale_editplus2, 
            seed_param_editplus2,
            transformer_dropdown2,
            lora_dropdown, 
            lora_weights,
            res_vram_tb,
        ],
        outputs = [image_output_editplus2, info_editplus2]
    )
    enhance_button_editplus2.click(
        fn=enhance_prompt_edit2, 
        inputs=[prompt_editplus2, image_editplus2, image_editplus3, image_editplus4, image_editplus5], 
        outputs=[prompt_editplus2, info_editplus2]
    )
    reverse_button_editplus2.click(
        fn=enhance_prompt, 
        inputs=[prompt_editplus2, image_editplus2], 
        outputs=[prompt_editplus2, info_editplus2]
    )
    exchange_button_editplus2.click(
        fn=exchange_width_height, 
        inputs=[width_editplus2, height_editplus2], 
        outputs=[width_editplus2, height_editplus2, info_editplus2]
    )
    scale_1_5_button_editplus2.click(
        fn=scale_resolution_1_5,
        inputs=[width_editplus2, height_editplus2],
        outputs=[width_editplus2, height_editplus2, info_editplus2]
    )
    image_editplus2.upload(
        fn=adjust_width_height_editplus2, 
        inputs=[image_editplus2], 
        outputs=[width_editplus2, height_editplus2, info_editplus2]
    )
    save_example_button_editplus2.click(
        fn=lambda prompt: save_example(prompt, "editplus"),
        inputs=[prompt_editplus2],
        outputs=[examples_dropdown_editplus2, info_editplus2]
    )
    examples_dropdown_editplus2.change(
        fn=lambda selected_example, current_prompt: f"{current_prompt} {selected_example.strip()}",
        inputs=[examples_dropdown_editplus2, prompt_editplus2],
        outputs=[prompt_editplus2]
    )
    stop_button_editplus2.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_editplus2]
    )
    # 3D相机控制
    def update_prompt_from_sliders_camera(azimuth, elevation, distance):
        """Update prompt preview when sliders change."""
        prompt = build_camera_prompt(azimuth, elevation, distance)
        return prompt
    
    def sync_3d_to_sliders_camera(camera_value):
        """Sync 3D control changes to sliders."""
        if camera_value and isinstance(camera_value, dict):
            az = camera_value.get('azimuth', 0)
            el = camera_value.get('elevation', 0)
            dist = camera_value.get('distance', 1.0)
            prompt = build_camera_prompt(az, el, dist)
            return az, el, dist, prompt
        return gr.update(), gr.update(), gr.update(), gr.update()
    
    def sync_sliders_to_3d_camera(azimuth, elevation, distance):
        """Sync slider changes to 3D control."""
        return {"azimuth": azimuth, "elevation": elevation, "distance": distance}
    
    def update_3d_image_camera(image):
        """Update the 3D component with the uploaded image."""
        if image is None:
            return gr.update(imageUrl=None)
        # Convert PIL image to base64 data URL
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        data_url = f"data:image/png;base64,{img_str}"
        return gr.update(imageUrl=data_url)
    
    # Slider -> Prompt preview
    for slider in [azimuth_slider, elevation_slider, distance_slider]:
        slider.change(
            fn=update_prompt_from_sliders_camera,
            inputs=[azimuth_slider, elevation_slider, distance_slider],
            outputs=[prompt_preview_camera]
        )
    
    # 3D control -> Sliders + Prompt
    camera_3d.change(
        fn=sync_3d_to_sliders_camera,
        inputs=[camera_3d],
        outputs=[azimuth_slider, elevation_slider, distance_slider, prompt_preview_camera]
    )
    
    # Sliders -> 3D control
    for slider in [azimuth_slider, elevation_slider, distance_slider]:
        slider.release(
            fn=sync_sliders_to_3d_camera,
            inputs=[azimuth_slider, elevation_slider, distance_slider],
            outputs=[camera_3d]
        )
    
    # Prompt enhancement and reverse
    enhance_button_camera.click(
        fn=enhance_prompt, 
        inputs=[additional_prompt_camera, image_camera], 
        outputs=[additional_prompt_camera, info_camera]
    )
    reverse_button_camera.click(
        fn=enhance_prompt, 
        inputs=[additional_prompt_camera, image_camera], 
        outputs=[additional_prompt_camera, info_camera]
    )
    save_example_button_camera.click(
        fn=lambda prompt: save_example(prompt, "camera"),
        inputs=[additional_prompt_camera],
        outputs=[examples_dropdown_camera, info_camera]
    )
    examples_dropdown_camera.change(
        fn=lambda selected_example, current_prompt: f"{current_prompt} {selected_example.strip()}" if current_prompt else selected_example.strip(),
        inputs=[examples_dropdown_camera, additional_prompt_camera],
        outputs=[additional_prompt_camera]
    )
    
    # Generate button
    run_btn_camera.click(
        fn=generate_camera_edit,
        inputs=[image_camera, azimuth_slider, elevation_slider, distance_slider, negative_prompt_camera, 
                width_camera, height_camera, num_inference_steps_camera, 
                batch_images_camera, true_cfg_scale_camera, seed_param_camera, transformer_dropdown2, 
                lora_dropdown, lora_weights, res_vram_tb, additional_prompt_camera],
        outputs=[result_camera, info_camera]
    )
    
    # Exchange width and height
    exchange_button_camera.click(
        fn=exchange_width_height, 
        inputs=[width_camera, height_camera], 
        outputs=[width_camera, height_camera, info_camera]
    )
    
    # Scale resolution 1.5x
    scale_1_5_button_camera.click(
        fn=scale_resolution_1_5,
        inputs=[width_camera, height_camera],
        outputs=[width_camera, height_camera, info_camera]
    )
    
    # Image upload -> update dimensions AND update 3D preview
    image_camera.upload(
        fn=update_dimensions_on_upload_camera,
        inputs=[image_camera],
        outputs=[width_camera, height_camera, info_camera]
    ).then(
        fn=update_3d_image_camera,
        inputs=[image_camera],
        outputs=[camera_3d]
    )
    
    # Also handle image clear
    image_camera.clear(
        fn=lambda: gr.update(imageUrl=None),
        outputs=[camera_3d]
    )
    
    stop_button_camera.click(
        fn=stop_generate, 
        inputs=[], 
        outputs=[info_camera]
    )
    # ControlNet预处理
    generate_button_cont.click(
        fn = generate_cont,
        inputs = [
            image_cont,
            processor_cont,
        ],
        outputs = [image_output_cont, info_cont]
    )
    send_to_i2i.click(
        fn=lambda x: x,
        inputs=[image_output_cont],
        outputs=[image_i2i]
    )
    send_to_inp.click(
        fn=lambda x: {"background": x, "layers": [], "composite": x},
        inputs=[image_output_cont],
        outputs=[image_inp]
    )
    send_to_edit2.click(
        fn=lambda x: x,
        inputs=[image_output_cont],
        outputs=[image_editplus2]
    )
    send_to_edit3.click(
        fn=lambda x: x,
        inputs=[image_output_cont],
        outputs=[image_editplus3]
    )
    send_to_edit4.click(
        fn=lambda x: x,
        inputs=[image_output_cont],
        outputs=[image_editplus4]
    )
    send_to_edit5.click(
        fn=lambda x: x,
        inputs=[image_output_cont],
        outputs=[image_editplus5]
    )
    # 转换lora
    convert_button.click(
        fn=convert_lora,
        inputs = [lora_in],
        outputs = [lora_out, info_lora]
    )
    # 图库
    refresh_gallery_button.click(
        fn=refresh_gallery,
        inputs=[],
        outputs=[gallery, gallery_info]
    )
    demo.load(
        fn=refresh_gallery,
        inputs=[],
        outputs=[gallery, gallery_info]
    )
    gallery.select(
        fn=update_selection,
        outputs=selected_index
    ).then(
        fn=load_image_info,
        inputs=[selected_index, gallery],
        outputs=[info_info]
    )
    send_to_i2i_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_i2i]
    )
    send_to_inp_gallery.click(
        fn=lambda idx, gallery: {"background": Image.open(gallery[idx][0]), "layers": [], "composite": Image.open(gallery[idx][0])} if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_inp]
    )
    send_to_edit2_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_editplus2]
    )
    send_to_edit3_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_editplus3]
    )
    send_to_edit4_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_editplus4]
    )
    send_to_edit5_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_editplus5]
    )
    send_to_cont_gallery.click(
        fn=lambda idx, gallery: Image.open(gallery[idx][0]) if idx >= 0 and idx < len(gallery) else None,
        inputs=[selected_index, gallery],
        outputs=[image_cont]
    )
    # 设置
    save_button.click(
        fn=save_openai_config,
        inputs=[transformer_dropdown, transformer_dropdown2, res_vram_tb, openai_base_url_tb, openai_api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb],
        outputs=[info_config],
    )


if __name__ == "__main__": 
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
        theme=gr.themes.Soft(font=[gr.themes.GoogleFont("IBM Plex Sans")]), 
        css=css,
        head=head,
    )