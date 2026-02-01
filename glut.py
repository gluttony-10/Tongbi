import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.registry")
warnings.filterwarnings("ignore", category=UserWarning, module="controlnet_aux.segment_anything.modeling.tiny_vit_sam")
import io
import os
import json
import time
import base64
import torch
import numpy as np
import psutil
import gradio as gr
from openai import OpenAI
import requests
import argparse
from PIL import Image
from diffusers.utils import load_image
from utils.camera_control import CameraControl3D
from utils.prompt_enhancer import enhance_prompt, enhance_prompt_edit2
from utils.image_utils import exchange_width_height, adjust_width_height, adjust_width_height_editplus2
from utils.config_utils import initialize_examples_file, save_tab_model, load_tab_models, load_examples, save_example
from utils.gallery_utils import load_gallery, refresh_gallery, load_image_info
from utils.model_loader import load_model
from utils.camera_utils import build_camera_prompt, update_dimensions_on_upload_camera
from utils.ui_utils import stop_generate, change_reference_count, scale_resolution_1_5, find_port, update_selection, load_image_info_wrapper, generate_cont, save_openai_config
from utils.generator import generate_t2i, generate_i2i, generate_inp, generate_editplus2, generate_camera_edit
import utils.state as state

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

# 启用 CUDA 加速优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # 自动寻找最优卷积算法
    torch.backends.cuda.matmul.allow_tf32 = True  # 允许 TF32 矩阵乘法
    torch.backends.cudnn.allow_tf32 = True  # 允许 TF32 加速

#初始化（使用全局状态模块）
state.config = {}
state.transformer_choices = []
state.transformer_choices2 = []
state.t2i_choices = []
state.transformer_loaded = None
state.lora_choices = []
state.lora_loaded = None
state.lora_loaded_weights = None
state.image_loaded = None
state.mode = None
state.mode_loaded = None
state.pipe = None
state.prompt_cache = None
state.negative_prompt_cache = None
state.model_id = "models/Qwen-Image"
state.stop_generation = False
state.mmgp = None
state.device = device
state.dtype = dtype
state.args = args
state.mem = mem

# 为了兼容性，创建局部变量引用
config = state.config
transformer_choices = state.transformer_choices
transformer_choices2 = state.transformer_choices2
t2i_choices = state.t2i_choices
transformer_loaded = state.transformer_loaded
lora_choices = state.lora_choices
lora_loaded = state.lora_loaded
lora_loaded_weights = state.lora_loaded_weights
image_loaded = state.image_loaded
mode = state.mode
mode_loaded = state.mode_loaded
pipe = state.pipe
prompt_cache = state.prompt_cache
negative_prompt_cache = state.negative_prompt_cache
model_id = state.model_id
stop_generation = state.stop_generation
mmgp = state.mmgp
prompt_embeds_cache = state.prompt_embeds_cache

EXAMPLES_FILE = "json/prompts.json"

#确保输出文件夹存在
os.makedirs("outputs", exist_ok=True)
#确保json文件夹存在
os.makedirs("json", exist_ok=True)
#读取设置
CONFIG_FILE = "json/config.json"

# 初始化模型列表（在UI创建前）
def init_model_choices():
    """初始化模型选择列表（使用固定模型，不扫描目录）"""
    global transformer_choices, transformer_choices2, t2i_choices, controlnet_processor_choices
    # 同步到全局状态
    state.transformer_choices = transformer_choices
    state.transformer_choices2 = transformer_choices2
    state.t2i_choices = t2i_choices
    
    # 固定基础模型（用于图生图、局部重绘）
    base_model_full = "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp.safetensors"
    base_model_display = base_model_full.replace(".safetensors", "")  # 显示时去掉后缀
    transformer_choices = [base_model_display]
    
    # 文生图模型（包含本地模型和MS-Qwen-Image、MS-Qwen-Image-2512和MS-Z-Image-Turbo云端选项）
    t2i_choices = [base_model_display, "MS-Qwen-Image", "MS-Qwen-Image-2512", "MS-Z-Image-Turbo"]
    
    # 编辑模型（用于多图编辑、3D相机控制）- 包含本地模型和MS选项
    edit_model_full = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-mmgp.safetensors"
    edit_model_display = edit_model_full.replace(".safetensors", "")  # 显示时去掉后缀
    transformer_choices2 = [edit_model_display, "MS-Qwen-Image-Edit-2509", "MS-Qwen-Image-Edit-2511"]
    
    # ControlNet预处理选项列表
    controlnet_processor_choices = [
        "canny", "depth_leres", "depth_leres++", "depth_midas", "depth_zoe", 
        "lineart_anime", "lineart_coarse", "lineart_realistic", "mediapipe_face", 
        "mlsd", "normal_bae", "openpose", "openpose_face", 
        "openpose_faceonly", "openpose_full", "openpose_hand", "scribble_hed", 
        "scribble_pidinet", "shuffle", "softedge_hed", "softedge_hedsafe", 
        "softedge_pidinet", "softedge_pidsafe"
    ]

# 初始化模型选择
init_model_choices()
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {}

# 每个TabItem的模型选择（从独立的JSON文件加载）
tab_models = load_tab_models()
default_t2i_model = "Qwen-Image-2512-Lightning-4steps-V1.0-mmgp"
transformer_t2i = tab_models.get("t2i", default_t2i_model)
# 如果保存的是旧名称，自动迁移到新名称
if transformer_t2i == "ModelScope-QI.safetensors" or transformer_t2i == "ModelScope-QI":
    transformer_t2i = "MS-Qwen-Image"

transformer_i2i = tab_models.get("i2i", default_t2i_model)

transformer_inp = tab_models.get("inp", default_t2i_model)

default_edit_model = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-mmgp"
transformer_editplus = tab_models.get("editplus", default_edit_model)

transformer_camera = tab_models.get("camera", default_edit_model)
res_vram = float(config.get("RES_VRAM", "1500"))
state.res_vram = res_vram  # 生成时只使用已保存的设置
openai_base_url = config.get("OPENAI_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
openai_api_key = config.get("OPENAI_API_KEY", "")
model_name = config.get("MODEL_NAME", "GLM-4.6V-Flash")
temperature = float(config.get("TEMPERATURE", "0.8"))
top_p = float(config.get("TOP_P", "0.6"))
max_tokens = float(config.get("MAX_TOKENS", "16384"))
modelscope_api_key = config.get("MODELSCOPE_API_KEY", "")
image_format = config.get("IMAGE_FORMAT", "png").lower()  # 图片保存格式，默认png
state.image_format = image_format  # 同步到全局状态，使设置里的 webp 等格式生效


def refresh_model():
    """只刷新 LoRA 模型列表"""
    global lora_choices
    
    lora_dir = "models/lora/Qwen-Image"  # 只读取Qwen-Image文件夹
    
    if os.path.exists(lora_dir):
        lora_files = [f for f in os.listdir(lora_dir) if f.endswith(".safetensors")]
        lora_choices = sorted(lora_files)
    else:
        lora_choices = []
        if not os.path.exists("models/lora"):
            print("models/lora文件夹不存在")
        elif not os.path.exists(lora_dir):
            print(f"models/lora/Qwen-Image文件夹不存在")
    
    # 只更新 LoRA 下拉框，其他组件不更新
    return (
        gr.Dropdown(),  # transformer_dropdown (不更新)
        gr.Dropdown(choices=lora_choices, multiselect=True),  # lora_dropdown (只更新LoRA)
        gr.Dropdown(),  # transformer_t2i (不更新)
        gr.Dropdown(),  # transformer_i2i (不更新)
        gr.Dropdown(),  # transformer_inp (不更新)
        gr.Dropdown(),  # transformer_editplus (不更新)
        gr.Dropdown(),  # transformer_camera (不更新)
    )

initialize_examples_file()
refresh_model()


# load_model 已移至 utils.model_loader，这里保留包装函数以同步全局状态
def load_model_wrapper(mode, transformer_dropdown, lora_dropdown, lora_weights, res_vram):
    """包装 load_model 以同步全局状态"""
    from utils.model_loader import load_model
    load_model(mode, transformer_dropdown, lora_dropdown, lora_weights, res_vram)
    # 同步状态
    global pipe, mode_loaded, transformer_loaded, lora_loaded, lora_loaded_weights, mmgp
    pipe = state.pipe
    mode_loaded = state.mode_loaded
    transformer_loaded = state.transformer_loaded
    lora_loaded = state.lora_loaded
    lora_loaded_weights = state.lora_loaded_weights
    mmgp = state.mmgp

# 为了兼容性，创建别名
load_model = load_model_wrapper

css = """
.icon-btn {
    min-width: unset !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.refresh-btn {
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    max-width: 36px !important;
    max-height: 36px !important;
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    font-size: 16px !important;
    line-height: 1 !important;
    border-radius: 4px !important;
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
            """)
    with gr.Row():
        # 页面选择下拉框
        page_choices = ["文生图", "图生图", "局部重绘", "多图编辑", "3D相机控制", "ControlNet预处理", "图库", "设置"]
        page_dropdown = gr.Dropdown(label="功能选择", info="第一次打开，请先选择设置", choices=page_choices, value="文生图", scale=1)
        transformer_dropdown = gr.Dropdown(label="模型选择", info="MS开头为云端模型，调用API", choices=t2i_choices, value=transformer_t2i if transformer_t2i in t2i_choices else (t2i_choices[0] if t2i_choices else None), scale=2)
        lora_dropdown = gr.Dropdown(label="LoRA模型", info="下载LoRA到models/lora/对应目录，可多选", choices=lora_choices, multiselect=True, scale=2)
        lora_weights = gr.Textbox(label="LoRA权重", info="多个权重请用英文逗号隔开。例如：0.8,0.5,0.2", value="", scale=2)
        refresh_button = gr.Button("🔄", scale=0, min_width=36, elem_classes="refresh-btn")
    
    # 用于跟踪当前选中的模型值（初始化为文生图模型的第一个选项）
    initial_model_value = transformer_t2i if transformer_t2i in t2i_choices else (t2i_choices[0] if t2i_choices else None)
    current_model_value = gr.State(value=initial_model_value)
    
    # 文生图页面
    with gr.Column(visible=True) as page_t2i:
        with gr.Row():
            with gr.Column():
                # 文生图使用t2i_choices（包含MS-Qwen-Image、MS-Qwen-Image-2512和MS-Z-Image-Turbo）
                # 注意：t2i_choices 已经包含所有选项，直接使用
                transformer_t2i = gr.Dropdown(label="基础模型", choices=t2i_choices, value=transformer_t2i if transformer_t2i in t2i_choices else (t2i_choices[0] if t2i_choices else None), interactive=True, visible=False)
                prompt = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button = gr.Button("🖼️ 开始生成", variant='primary', scale=5)
                    enhance_button = gr.Button("✨ 提示词增强", scale=2)
                    save_example_button = gr.Button("💾 保存提示词", scale=2)
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("推荐分辨率：1328x1328、1664x928、1472x1104")
                    with gr.Row():
                        width = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button = gr.Button("🔄 交换宽高")
                        scale_1_5_button = gr.Button("📐 1.5倍分辨率")
                    batch_images = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
            with gr.Column():
                info = gr.Textbox(label="提示信息", interactive=False)
                image_output = gr.Gallery(label="生成结果", interactive=False)
                stop_button = gr.Button("⏹️ 中止生成", variant="stop")
                examples_dropdown = gr.Dropdown(
                    label="提示词库", 
                    choices=load_examples("t2i"),
                    interactive=True,
                    scale=5
                )
    # 图生图页面
    with gr.Column(visible=False) as page_i2i:
        with gr.Row():
            with gr.Column():
                transformer_i2i = gr.Dropdown(label="基础模型", choices=transformer_choices, value=transformer_i2i if transformer_i2i in transformer_choices else (transformer_choices[0] if transformer_choices else None), interactive=True, visible=False)
                image_i2i = gr.Image(label="输入图片", type="pil", height=400)
                prompt_i2i = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt_i2i = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button_i2i = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    enhance_button_i2i = gr.Button("✨ 提示词增强", scale=2)
                    reverse_button_i2i = gr.Button("🔍 反推提示词", scale=2)
                    save_example_button_i2i = gr.Button("💾 保存提示词", scale=2)
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_i2i = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height_i2i = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button_i2i = gr.Button("🔄 交换宽高")
                        scale_1_5_button_i2i = gr.Button("📐 1.5倍分辨率")
                    strength_i2i = gr.Slider(label="strength", minimum=0, maximum=1, step=0.01, value=0.5)
                    batch_images_i2i = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_i2i = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_i2i = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_i2i = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
            with gr.Column():
                info_i2i = gr.Textbox(label="提示信息", interactive=False)
                image_output_i2i = gr.Gallery(label="生成结果", interactive=False)
                stop_button_i2i = gr.Button("⏹️ 中止生成", variant="stop")
                examples_dropdown_i2i = gr.Dropdown(
                    label="提示词库", 
                    choices=load_examples("i2i"),
                    interactive=True,
                    scale=5
                )
    # 局部重绘页面
    with gr.Column(visible=False) as page_inp:
        with gr.Row():
            with gr.Column():
                transformer_inp = gr.Dropdown(label="基础模型", choices=transformer_choices, value=transformer_inp if transformer_inp in transformer_choices else (transformer_choices[0] if transformer_choices else None), interactive=True, visible=False)
                image_inp = gr.ImageMask(label="输入蒙版", type="pil", height=400)
                prompt_inp = gr.Textbox(label="提示词", value="超清，4K，电影级构图，")
                negative_prompt_inp = gr.Textbox(label="负面提示词", value="")
                with gr.Row():
                    generate_button_inp = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    enhance_button_inp = gr.Button("✨ 提示词增强", scale=2)
                    reverse_button_inp = gr.Button("🔍 反推提示词", scale=2)
                    save_example_button_inp = gr.Button("💾 保存提示词", scale=2)
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_inp = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1328)
                        height_inp = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1328)
                    with gr.Row():
                        exchange_button_inp = gr.Button("🔄 交换宽高")
                        scale_1_5_button_inp = gr.Button("📐 1.5倍分辨率")
                    strength_inp = gr.Slider(label="strength", minimum=0, maximum=1, step=0.01, value=0.8)
                    batch_images_inp = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_inp = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_inp = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_inp = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
            with gr.Column():
                info_inp = gr.Textbox(label="提示信息", interactive=False)
                image_output_inp = gr.Gallery(label="生成结果", interactive=False)
                stop_button_inp = gr.Button("⏹️ 中止生成", variant="stop")
                examples_dropdown_inp = gr.Dropdown(
                    label="提示词库", 
                    choices=load_examples("inp"),
                    interactive=True,
                    scale=5
                )
    # 多图编辑页面
    with gr.Column(visible=False) as page_editplus:
        with gr.Row():
            with gr.Column():
                transformer_editplus = gr.Dropdown(label="编辑模型", choices=transformer_choices2, value=transformer_editplus if transformer_editplus in transformer_choices2 else (transformer_choices2[0] if transformer_choices2 else None), interactive=True, visible=False)
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
                    generate_button_editplus2 = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                    enhance_button_editplus2 = gr.Button("✨ 提示词增强", scale=2)
                    reverse_button_editplus2 = gr.Button("🔍 反推提示词", scale=2)
                    save_example_button_editplus2 = gr.Button("💾 保存提示词", scale=2)
                with gr.Accordion("参数设置", open=True):
                    gr.Markdown("上传图像后分辨率自动计算")
                    with gr.Row():
                        width_editplus2 = gr.Slider(label="宽度", minimum=256, maximum=3072, step=16, value=1024)
                        height_editplus2 = gr.Slider(label="高度", minimum=256, maximum=3072, step=16, value=1024)
                    with gr.Row():
                        exchange_button_editplus2 = gr.Button("🔄 交换宽高")
                        scale_1_5_button_editplus2 = gr.Button("📐 1.5倍分辨率")
                    batch_images_editplus2 = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_editplus2 = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_editplus2 = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_editplus2 = gr.Number(label="种子，请输入自然数，-1为随机", value=0)
            with gr.Column():
                info_editplus2 = gr.Textbox(label="提示信息", interactive=False)
                image_output_editplus2 = gr.Gallery(label="生成结果", interactive=False)
                stop_button_editplus2 = gr.Button("⏹️ 中止生成", variant="stop")
                examples_dropdown_editplus2 = gr.Dropdown(
                    label="提示词库", 
                    choices=load_examples("editplus"),
                    interactive=True,
                    scale=5
                )
    # 3D相机控制页面
    with gr.Column(visible=False) as page_camera:
        with gr.Row():
            # Left column: Input image and controls
            with gr.Column(scale=1):
                transformer_camera = gr.Dropdown(label="编辑模型", choices=transformer_choices2, value=transformer_camera if transformer_camera in transformer_choices2 else (transformer_choices2[0] if transformer_choices2 else None), interactive=True, visible=False)
                with gr.Row():
                    image_camera = gr.Image(label="输入图片", type="pil", height=500)
                    with gr.Column():
                        camera_3d = CameraControl3D(
                            value={"azimuth": 0, "elevation": 0, "distance": 1.0},
                            elem_id="camera-3d-control"
                        )
                        gr.Markdown("*拖动彩色手柄：🟢 方位角，🩷 仰角，🟡 距离（上远下近）*")
                with gr.Row():
                    run_btn_camera = gr.Button("🖼️ 生成", variant="primary", scale=4)
                    enhance_button_camera = gr.Button("✨ 提示词增强", scale=2)
                    reverse_button_camera = gr.Button("🔍 反推提示词", scale=2)
                    save_example_button_camera = gr.Button("💾 保存提示词", scale=2)
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
                        scale_1_5_button_camera = gr.Button("📐 1.5倍分辨率")
                    batch_images_camera = gr.Slider(label="批量生成", minimum=1, maximum=100, step=1, value=1)
                    num_inference_steps_camera = gr.Slider(label="采样步数（推荐4步）", minimum=1, maximum=100, step=1, value=4)
                    true_cfg_scale_camera = gr.Slider(label="true cfg scale", minimum=1, maximum=10, step=0.1, value=1.0)
                    seed_param_camera = gr.Number(label="种子，请输入自然数，-1为随机", value=-1)
            
            # Right column: Output
            with gr.Column(scale=1):
                info_camera = gr.Textbox(label="提示信息", interactive=False)
                result_camera = gr.Gallery(label="生成结果", interactive=False)
                stop_button_camera = gr.Button("⏹️ 中止生成", variant="stop")
                examples_dropdown_camera = gr.Dropdown(
                    label="提示词库", 
                    choices=load_examples("camera"),
                    interactive=True,
                    scale=5
                )
    # ControlNet预处理页面
    with gr.Column(visible=False) as page_controlnet:
        with gr.TabItem("图片预处理"):
            with gr.Row():
                with gr.Column():
                    image_cont = gr.Image(label="输入图片", type="pil", height=400)
                    # 预处理下拉框已移到顶部的模型选择区域
                    generate_button_cont = gr.Button("🖼️ 开始生成", variant='primary', scale=4)
                with gr.Column():
                    info_cont = gr.Textbox(label="提示信息", interactive=False)
                    image_output_cont = gr.Image(label="生成结果", interactive=False)
                    with gr.Row():
                        send_to_i2i = gr.Button("📤 发送到图生图", scale=1)
                        send_to_inp = gr.Button("📤 发送到局部重绘", scale=1)
                    with gr.Row():
                        send_to_edit2 = gr.Button("📤 发送到多图编辑1", scale=1)
                        send_to_edit3 = gr.Button("📤 发送到多图编辑2", scale=1)
                        send_to_edit4 = gr.Button("📤 发送到多图编辑3", scale=1)
                        send_to_edit5 = gr.Button("📤 发送到多图编辑4", scale=1)
        with gr.TabItem("Open Pose Editor"):
            gr.HTML('<iframe src="https://zhuyu1997.github.io/open-pose-editor/" width="100%" height="800px" frameborder="0" style="border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);"></iframe>')
    # 图库页面
    with gr.Column(visible=False) as page_gallery:
        with gr.Row():
            with gr.Column(scale=3):
                refresh_gallery_button = gr.Button("🔄 刷新图库")
                gallery = gr.Gallery(label="图库", columns=4, height="auto", object_fit="cover")
                selected_index = gr.Number(value=-1, visible=False)
            with gr.Column(scale=2):
                gallery_info = gr.Textbox(label="提示信息", interactive=False)
                info_info = gr.Textbox(label="图片信息", lines=20, interactive=False)
        with gr.Row():
            send_to_i2i_gallery = gr.Button("📤 发送到图生图")
            send_to_inp_gallery = gr.Button("📤 发送到局部重绘")
            send_to_edit2_gallery = gr.Button("📤 发送到多图编辑1")
            send_to_edit3_gallery = gr.Button("📤 发送到多图编辑2")
            send_to_edit4_gallery = gr.Button("📤 发送到多图编辑3")
            send_to_edit5_gallery = gr.Button("📤 发送到多图编辑4")
            send_to_cont_gallery = gr.Button("📤 发送到ControlNet预处理")
    # 设置页面
    with gr.Column(visible=False) as page_settings:
        with gr.Row():
            with gr.Column():
                res_vram_tb = gr.Slider(label="保留显存", info="单位MB，数值越大，显存占用越小，速度越慢", minimum=0, maximum=80000, step=1, value=res_vram)
                with gr.Accordion("多模态API设置", open=True):
                    openai_base_url_tb = gr.Textbox(label="BASE URL", info="请输入BASE URL，例如：https://open.bigmodel.cn/api/paas/v4", value=openai_base_url)
                    openai_api_key_tb = gr.Textbox(label="API KEY", info="请输入API KEY，暗文显示", value=openai_api_key, type="password")
                    with gr.Row():
                        model_name_tb = gr.Textbox(label="MODEL NAME", info="请输入模型名称，需要支持图片输入的多模态模型，例如：GLM-4.6V", value=model_name)
                        temperature_tb = gr.Slider(label="temperature", info="采样温度，控制输出的随机性和创造性", minimum=0, maximum=1, step=0.1, value=temperature)
                    with gr.Row():
                        top_p_tb = gr.Slider(label="top_p", info="核采样（nucleus sampling）参数，是temperature采样的替代方法", minimum=0, maximum=1, step=0.1, value=top_p)
                        max_tokens_tb = gr.Slider(label="max_tokens", info="模型输出的最大令牌（token）数量限制", minimum=1024, maximum=65536, step=1024, value=max_tokens)
                with gr.Accordion("在线生图API设置", open=True):
                    modelscope_api_key_tb = gr.Textbox(label="魔搭的API KEY", info="使用魔搭在线模型时需要，获取地址https://modelscope.cn/my/myaccesstoken", value=modelscope_api_key, type="password")
                with gr.Accordion("图片保存设置", open=True):
                    image_format_tb = gr.Dropdown(label="图片保存格式", info="选择生成图片的保存格式", choices=["png", "jpg", "webp"], value=image_format)
            with gr.Column():
                info_config = gr.Textbox(label="提示信息", value="修改后请点击保存设置生效；生成时仅使用已保存的设置，不会使用未保存的更改。", interactive=False)
                save_button = gr.Button("💾 保存设置", variant='primary')
                gr.Markdown("""多模态API设置支持通用类OPENAI的API，请使用多模态模型，如：GLM-4.6V、GLM-4.6V-Flash等（需要支持base64）。
                            可申请[智谱API](https://www.bigmodel.cn/invite?icode=eKq1YoHsX6y4VhGIPJuOPGczbXFgPRGIalpycrEwJ28%3D)。
                            temperature、top_p和max_tokens三个值，默认是GLM-4.6V的推荐值。
                            如果更换模型，请自行修改。
                            保存设置除了保存此页面的设置，还会保存QI基础模型和QI编辑模型的设置。
                            """)
    # 页面切换时更新页面可见性和基础模型下拉框的选项
    def on_page_change(selected_page, current_model):
        """根据选择的页面更新页面可见性和基础模型下拉框的选项"""
        # 初始化所有页面的可见性
        page_visibility = {
            "文生图": False,
            "图生图": False,
            "局部重绘": False,
            "多图编辑": False,
            "3D相机控制": False,
            "ControlNet预处理": False,
            "图库": False,
            "设置": False
        }
        
        # 设置当前页面为可见
        if selected_page in page_visibility:
            page_visibility[selected_page] = True
        
        # 根据页面判断使用哪个模型列表
        choices = None
        new_value = None
        if selected_page == "文生图":
            # 文生图使用完整模型列表（包含MS-Z-Image-Turbo）
            choices = t2i_choices
            # 检查当前值是否在新列表中，如果在就保持，否则使用第一个值
            if current_model and current_model in choices:
                new_value = current_model
            else:
                new_value = choices[0] if choices else None
            # 确保 new_value 在 choices 中
            if new_value not in choices:
                new_value = choices[0] if choices else None
            model_update = gr.update(choices=choices, value=new_value)
        elif selected_page in ["图生图", "局部重绘"]:
            # 图生图、局部重绘只使用本地模型
            choices = transformer_choices
            if current_model and current_model in choices:
                new_value = current_model
            else:
                new_value = choices[0] if choices else None
            # 确保 new_value 在 choices 中
            if new_value not in choices:
                new_value = choices[0] if choices else None
            model_update = gr.update(choices=choices, value=new_value)
        elif selected_page in ["多图编辑", "3D相机控制"]:
            # 多图编辑、3D相机控制使用编辑模型列表
            choices = transformer_choices2
            if current_model and current_model in choices:
                new_value = current_model
            else:
                new_value = choices[0] if choices else None
            # 确保 new_value 在 choices 中
            if new_value not in choices:
                new_value = choices[0] if choices else None
            model_update = gr.update(choices=choices, value=new_value)
        elif selected_page == "ControlNet预处理":
            # ControlNet预处理使用预处理选项列表
            choices = controlnet_processor_choices
            # 如果当前值是预处理选项，使用它；否则使用第一个选项
            if current_model and isinstance(current_model, str) and current_model in choices:
                new_value = current_model
            else:
                new_value = choices[0] if choices else None
            # 最终确保 new_value 在 choices 中且不为 None
            if not choices or new_value not in choices:
                new_value = choices[0] if choices else None
            model_update = gr.update(choices=choices, value=new_value)
        elif selected_page in ["图库", "设置"]:
            # 图库和设置页面，模型选择为空
            model_update = gr.update(choices=[], value=None)
            new_value = None
        else:
            # 其他页面保持当前选项，不更新模型列表
            model_update = gr.update()
            new_value = current_model
        
        return (
            gr.update(visible=page_visibility["文生图"]),  # page_t2i
            gr.update(visible=page_visibility["图生图"]),  # page_i2i
            gr.update(visible=page_visibility["局部重绘"]),  # page_inp
            gr.update(visible=page_visibility["多图编辑"]),  # page_editplus
            gr.update(visible=page_visibility["3D相机控制"]),  # page_camera
            gr.update(visible=page_visibility["ControlNet预处理"]),  # page_controlnet
            gr.update(visible=page_visibility["图库"]),  # page_gallery
            gr.update(visible=page_visibility["设置"]),  # page_settings
            model_update,  # transformer_dropdown
            new_value  # current_model_value
        )
    
    # 修改页面切换函数，在切换时保存模型选择
    def on_page_change_with_save(selected_page, current_model):
        """页面切换时，保存当前模型选择到对应标签"""
        result = on_page_change(selected_page, current_model)
        # 如果当前有模型选择，保存到对应标签
        if current_model:
            if selected_page == "文生图":
                save_tab_model("t2i", current_model)
            elif selected_page == "图生图":
                save_tab_model("i2i", current_model)
            elif selected_page == "局部重绘":
                save_tab_model("inp", current_model)
            elif selected_page == "多图编辑":
                save_tab_model("editplus", current_model)
            elif selected_page == "3D相机控制":
                save_tab_model("camera", current_model)
        return result
    
    page_dropdown.change(
        fn=on_page_change_with_save,
        inputs=[page_dropdown, current_model_value],
        outputs=[page_t2i, page_i2i, page_inp, page_editplus, page_camera, page_controlnet, page_gallery, page_settings, transformer_dropdown, current_model_value]
    )
    
    # 当模型下拉框值改变时，更新 State
    def update_model_state(selected_model):
        return selected_model
    
    transformer_dropdown.change(
        fn=update_model_state,
        inputs=[transformer_dropdown],
        outputs=[current_model_value]
    )
    
    # 模型设置
    refresh_button.click(
        fn=refresh_model,
        inputs=[],
        outputs=[transformer_dropdown, lora_dropdown, transformer_t2i, transformer_i2i, transformer_inp, transformer_editplus, transformer_camera]
    )
    # 当基础模型选择改变时，同步更新所有TabItem的模型选择器（仅限基础模型TabItem，排除MS选项）
    def sync_model_to_tabs(selected_model, current_page):
        """同步基础模型选择到所有基础模型TabItem，并保存配置（MS选项只同步到文生图）"""
        if selected_model:
            # 根据当前页面保存到对应标签
            if current_page == "文生图":
                save_tab_model("t2i", selected_model)
            elif current_page == "图生图":
                save_tab_model("i2i", selected_model)
            elif current_page == "局部重绘":
                save_tab_model("inp", selected_model)
            elif current_page == "多图编辑":
                save_tab_model("editplus", selected_model)
            elif current_page == "3D相机控制":
                save_tab_model("camera", selected_model)
            
            # 保存到配置（基础模型标签）
            save_tab_model("t2i", selected_model)
            # MS选项只用于文生图，不同步到其他TabItem
            if selected_model not in ["MS-Qwen-Image", "MS-Qwen-Image-2512", "MS-Z-Image-Turbo"]:
                save_tab_model("i2i", selected_model)
                save_tab_model("inp", selected_model)
                return (
                    gr.Dropdown(value=selected_model),  # transformer_t2i
                    gr.Dropdown(value=selected_model),  # transformer_i2i
                    gr.Dropdown(value=selected_model),  # transformer_inp
                )
            else:
                # MS选项只更新文生图
                return (
                    gr.Dropdown(value=selected_model),  # transformer_t2i
                    gr.Dropdown(),  # transformer_i2i (不更新)
                    gr.Dropdown(),  # transformer_inp (不更新)
                )
        return (
            gr.Dropdown(),  # transformer_t2i
            gr.Dropdown(),  # transformer_i2i
            gr.Dropdown(),  # transformer_inp
        )
    
    transformer_dropdown.change(
        fn=sync_model_to_tabs,
        inputs=[transformer_dropdown, page_dropdown],
        outputs=[transformer_t2i, transformer_i2i, transformer_inp]
    )
    # 文生图
    gr.on(
        triggers=[generate_button.click, prompt.submit, negative_prompt.submit, seed_param.submit],
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
            transformer_t2i,
            lora_dropdown, 
            lora_weights,
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
        triggers=[generate_button_i2i.click, prompt_i2i.submit, negative_prompt_i2i.submit, seed_param_i2i.submit],
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
            transformer_i2i,
            lora_dropdown, 
            lora_weights,
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
        triggers=[generate_button_inp.click, prompt_inp.submit, negative_prompt_inp.submit, seed_param_inp.submit],
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
            transformer_inp,
            lora_dropdown, 
            lora_weights,
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
        triggers=[generate_button_editplus2.click, prompt_editplus2.submit, negative_prompt_editplus2.submit, seed_param_editplus2.submit],
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
            transformer_dropdown,  # 使用顶部的模型选择下拉框，而不是隐藏的 transformer_editplus
            lora_dropdown, 
            lora_weights,
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
    
    # Generate button - 支持回车触发
    gr.on(
        triggers=[run_btn_camera.click, negative_prompt_camera.submit, seed_param_camera.submit],
        fn=generate_camera_edit,
        inputs=[image_camera, azimuth_slider, elevation_slider, distance_slider, negative_prompt_camera, 
                width_camera, height_camera, num_inference_steps_camera, 
                batch_images_camera, true_cfg_scale_camera, seed_param_camera, transformer_camera, 
                lora_dropdown, lora_weights, additional_prompt_camera],
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
    # ControlNet预处理（使用顶部的 transformer_dropdown 作为预处理选择）
    generate_button_cont.click(
        fn = generate_cont,
        inputs = [
            image_cont,
            transformer_dropdown,  # 使用顶部的模型选择下拉框作为预处理选择
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
        fn=load_image_info_wrapper,
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
        inputs=[transformer_dropdown, res_vram_tb, openai_base_url_tb, openai_api_key_tb, model_name_tb, temperature_tb, top_p_tb, max_tokens_tb, modelscope_api_key_tb, image_format_tb],
        outputs=[info_config],
    )


# 日间模式 + 护眼配色：暖灰背景、柔和主色、降低对比度
_theme = (
    gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="stone",
        font=[gr.themes.GoogleFont("IBM Plex Sans")],
    ).set(
        body_background_fill="#fafafa",
        block_background_fill="#ffffff",
        block_border_color="#e8e6e1",
        body_text_color="#000000",
        block_label_text_color="#000000",
        block_title_text_color="#000000",
        input_background_fill="#f0f0f0",
        input_border_color="#b8d0e8",
        button_secondary_background_fill="#e8f0f8",
        button_secondary_background_fill_hover="#d5e6f5",
        button_secondary_text_color="#000000",
        button_secondary_border_color="#b8d0e8",
    )
)

if __name__ == "__main__": 
    head = '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    demo.launch(
        server_name=args.server_name, 
        server_port=find_port(args.server_port),
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
        theme=_theme,
        css=css,
        head=head,
    )