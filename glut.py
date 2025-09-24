import gradio as gr
import numpy as np
import torch
import random

from diffusers import FluxKontextPipeline, GGUFQuantizationConfig, FluxTransformer2DModel, FluxPipeline
from transformers import T5EncoderModel
import psutil
import argparse
from openai import OpenAI
import os

from nunchaku import NunchakuFluxTransformer2dModel
from nunchaku.caching.diffusers_adapters import apply_cache_on_pipe
from nunchaku.utils import get_precision

import datetime
from PIL import Image

from image_gen_aux import DepthPreprocessor

parser = argparse.ArgumentParser() 
parser.add_argument("--flux_text_encoder_2", type=str, default="t5-v1_1-xxl-encoder-Q8_0.gguf", help="flux的text_encoder_2模型对应的gguf文件")
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址，局域网访问改为0.0.0.0")
parser.add_argument("--server_port", type=int, default=7890, help="使用端口")
parser.add_argument("--share", action="store_true", help="是否启用gradio共享")
parser.add_argument("--mcp_server", action="store_true", help="是否启用mcp服务")
parser.add_argument("--afbc", type=float, default=0, help="第一块缓存，用来加速生成，0为关闭，0.12为倍速")
parser.add_argument("--vram", type=str, default="low", choices=['low', 'mid', 'high'], help="调整显存占用，high占用22G，mid占用10G，low占用3.5G")
args = parser.parse_args()

MODEL_NAME = os.environ.get("MODEL_NAME", "glm-4-flash-250414")

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
        print(f'\033[32m不支持BF16，仅支持FP16\033[0m')
        dtype = torch.float16
else:
    print(f'\033[32mCUDA不可用，请检查\033[0m')
    device = "cpu"
precision = get_precision()
print(f'\033[32m使用{precision}模型\033[0m')

MAX_SEED = np.iinfo(np.int32).max

transformer = NunchakuFluxTransformer2dModel.from_pretrained(
    f"mit-han-lab/nunchaku-flux.1-dev/svdq-{precision}_r32-flux.1-dev.safetensors",
    offload=True if args.vram == "low" else False,
    torch_dtype=dtype
)
transformer.set_attention_impl("nunchaku-fp16")
pipe = FluxPipeline.from_pretrained(
    "models/FLUX.1-dev", 
    transformer=transformer, 
    torch_dtype=dtype
)
if args.afbc!=0:
    apply_cache_on_pipe(
        pipe,
        residual_diff_threshold=args.afbc,
    )
if args.vram == "high":
    pipe.to(device)
else:
    pipe.enable_sequential_cpu_offload()

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

# 添加本地 LoRA 扫描功能
LORA_DIR = "models/lora"
os.makedirs(LORA_DIR, exist_ok=True)
flux_loras_raw = []
current_lora = None 

if os.path.exists(LORA_DIR) and os.path.isdir(LORA_DIR):
    for file in os.listdir(LORA_DIR):
        if file.endswith(".safetensors"):
            file_path = os.path.join(LORA_DIR, file)
            base_name = os.path.splitext(file)[0]
            
            # 查找同名图片文件
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                test_path = os.path.join(LORA_DIR, base_name + ext)
                if os.path.isfile(test_path):
                    image_path = test_path
                    break
            
            # 查找并读取同名txt文件作为trigger_word
            trigger_word = ""
            txt_path = os.path.join(LORA_DIR, base_name + ".txt")
            if os.path.isfile(txt_path):
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        trigger_word = f.read().strip()
                except Exception as e:
                    print(f"读取 trigger_word 文件失败: {e}")
            
            flux_loras_raw.append({
                "image": image_path or "models/lora/Gluttony10.png",  # 设置默认图片路径
                "title": base_name,
                "path": file_path,  # 存储本地文件路径
                "trigger_word": trigger_word,  # 使用从txt读取的内容
            })
    print(f"从本地目录加载了 {len(flux_loras_raw)} 个 LoRA")
else:
    print(f"警告：LoRA 目录 {LORA_DIR} 不存在")


def load_lora_weights(lora_path):
    """直接加载本地 LoRA 文件"""
    if os.path.isfile(lora_path):
        return lora_path
    print(f"LoRA 文件不存在: {lora_path}")
    return None


def update_selection(selected_state: gr.SelectData, flux_loras):
    if selected_state.index >= len(flux_loras):
        return "### 未选择 LoRA", gr.update(), None
    
    lora_item = flux_loras[selected_state.index]
    updated_text = f"### 已选择: {lora_item['title']}"
    
    return updated_text, lora_item["trigger_word"], selected_state.index, gr.update(visible=True)


def remove_custom_lora():
    """Remove custom LoRA"""
    return gr.update(visible=False), None, None, gr.Gallery(selected_index=None), "### Click on a LoRA in the gallery to select it"


def classify_gallery(flux_loras):
    """Sort gallery by likes"""
    sorted_gallery = sorted(flux_loras, key=lambda x: x.get("likes", 0), reverse=True)
    return [(item["image"], item["title"]) for item in sorted_gallery], sorted_gallery


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt, "未设置 OPENAI_API_KEY"
    client = OpenAI()
    text = prompt.strip()

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[{"role": "system", "content": """
                翻译成英文
            """
            },
            {
                "role": "user",
                "content": text.strip()
            },
            ],
            model=MODEL_NAME,
            temperature=0.95,
            top_p=0.7,
            stream=False,
            max_tokens=1024,
        )
        if response.choices:
            return response.choices[0].message.content.replace('"', ''), "提示词增强完毕"
    return prompt.replace('"', ''), "提示词增强完毕"


def infer_with_lora(input_images, prompt, selected_index, custom_lora, seed=42, randomize_seed=False, steps=20, guidance_scale=2.5, lora_scale=1.0, flux_loras=None):
    global current_lora, pipe
    print(f"Received {len(input_images) if input_images else 0} images")

    # 创建输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定基础种子
    if randomize_seed:
        base_seed = random.randint(0, MAX_SEED)
    else:
        base_seed = seed

    all_results = []  # 存储所有生成结果
    
    lora_to_use = custom_lora or (
        flux_loras[selected_index] 
        if selected_index is not None and flux_loras and selected_index < len(flux_loras) 
        else None
    )
    
    if lora_to_use and lora_to_use != current_lora:
        try:
            # 直接使用本地文件路径
            lora_path = lora_to_use["path"]
            if os.path.isfile(lora_path):
                pipe.transformer.update_lora_params(lora_path)
                pipe.transformer.set_lora_strength(lora_scale)
                print(f"Loaded LoRA: {lora_path} with scale {lora_scale}")
                current_lora = lora_to_use
        except Exception as e:
            print(f"Error loading LoRA: {e}")
    elif not lora_to_use:
        pipe.transformer.set_lora_strength(0)
        
    # 逐步生成并输出每张图片
    if input_images:
        # 处理每张输入图像
        for index, img_path in enumerate(input_images):
            # 从文件路径加载图像
            try:
                # 提取元组中的实际文件路径
                actual_path = img_path[0] if isinstance(img_path, tuple) else img_path
                img = Image.open(actual_path).convert("RGB")
            except Exception as e:
                print(f"无法加载图像 {img_path}: {e}")
                continue
                
            # 为每张图确定独立种子
            if randomize_seed:
                current_seed = base_seed + index
            else:
                current_seed = seed + index
                
            width, height = img.size
            aspect_ratio = width / height
            
            # 选择最接近的分辨率
            _, target_width, target_height = min(
                (abs(aspect_ratio - w / h), w, h) 
                for w, h in PREFERED_KONTEXT_RESOLUTIONS
            )
            
            # 生成单张图像
            generator = torch.Generator().manual_seed(current_seed)
            result_img = pipe(
                image=img,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                height=target_height,
                width=target_width,
                generator=generator,
            ).images[0]
            
            # 保存图片到输出目录
            output_path = os.path.join(output_dir, f"result_{index+1}.png")
            result_img.save(output_path)
            
            # 添加到结果列表
            all_results.append(result_img)
            
            # 实时输出当前结果
            size_info = f"图{index+1}: {target_width}×{target_height} (已保存到 {output_path})"
            yield all_results, current_seed, f"正在生成... ({index+1}/{len(input_images)}) {size_info}"
        
        # 所有图片生成完成后输出最终结果
        if not all_results:
            size_info = "错误: 没有生成任何有效图片"
            yield [], base_seed, size_info
        else:
            size_info = f"完成! 共生成 {len(all_results)} 张图片，已保存到 {output_dir}"
            yield all_results, base_seed, size_info
    else:
        # 没有上传图像时生成单张图
        generator = torch.Generator().manual_seed(base_seed)
        result_img = pipe(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        
        # 保存单张图片
        output_path = os.path.join(output_dir, "result.png")
        result_img.save(output_path)
        
        yield [result_img], base_seed, f"生成尺寸: 1024×1024 (已保存到 {output_path})"


with gr.Blocks(theme=gr.themes.Base()) as demo:
    with gr.Column():
        gr.Markdown("""
                <div>
                    <h2 style="font-size: 30px;text-align: center;">通臂 Tongbi</h2>
                </div>
                <div style="text-align: center;">
                    十字鱼
                    <a href="https://space.bilibili.com/893892">🌐bilibili</a> 
                    |FLUX
                    <a href="https://github.com/black-forest-labs/flux">🌐github</a> 
                </div>
                <div style="text-align: center; font-weight: bold; color: red;">
                    ⚠️ 该演示仅供学术研究和体验使用。
                </div>
                """)
        with gr.Tabs():
            with gr.TabItem("FLUX.1-dev"):
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="选择LoRA",
                            allow_preview=False,
                            columns=3,
                            show_share_button=False,
                        )
                        lora_prompt = gr.Textbox(
                            label="LoRA提示词示例",
                            value="",
                            interactive=False, 
                        )
                        lora_scale = gr.Slider(
                            label="LoRA强度",
                            minimum=0,
                            maximum=2,
                            step=0.1,
                            value=1.0,
                        )
                        custom_model_button = gr.Button("取消选择的LoRA", visible=False)
                    with gr.Column():
                        prompt = gr.Text(
                            label='提示词（例如："Change the car color to red"，"Transform to Bauhaus art style"）',
                            placeholder="输入提示词，自然语言，英文描述",
                        )
                        with gr.Row():
                            randomize_seed = gr.Checkbox(label="随机种子", value=True, scale=1)
                            seed = gr.Slider(
                                label="种子",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0,
                                scale=2,
                            )
                        guidance_scale = gr.Slider(
                            label="指导量表",
                            minimum=1,
                            maximum=10,
                            step=0.1,
                            value=2.5,
                        )       
                        steps = gr.Slider(
                            label="推理步数",
                            minimum=1,
                            maximum=50,
                            value=20,
                            step=1
                        )

                        with gr.Row():
                            run_button = gr.Button("开始生成", variant="primary")
                            enhance_button = gr.Button("提示词增强", variant="secondary")
                        input_image = gr.Gallery(label="上传图像进行编辑（可多选）", type="filepath", file_types=["image"])
                    with gr.Column():
                        state = gr.Textbox(value="请修改参数、填写提示词并上传图像", interactive=False, show_label=False)
                        result = gr.Gallery(label="生成结果", interactive=False)
                        prompt_title = gr.Markdown(
                            value="### Click on a LoRA in the gallery to select it",
                            visible=True,
                        )
                        gr_flux_loras = gr.State(value=flux_loras_raw)
                        selected_state = gr.State(value=None)
                        custom_loaded_lora = gr.State(value=None)
            with gr.TabItem("新标签页"):
                gr.Markdown("## 新标签页内容")
                
    custom_model_button.click(
        fn=remove_custom_lora,
        outputs=[custom_model_button, custom_loaded_lora, selected_state, gallery, prompt_title]
    )
    gallery.select(
        fn=update_selection,
        inputs=[gr_flux_loras],
        outputs=[prompt_title, lora_prompt, selected_state, custom_model_button],
        show_progress=False
    )
    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer_with_lora,
        inputs = [input_image, prompt, selected_state, custom_loaded_lora, seed, randomize_seed, steps, guidance_scale, lora_scale, gr_flux_loras],
        outputs = [result, seed, state]
    )
    enhance_button.click(
        fn = convert_prompt,
        inputs = [prompt],
        outputs = [prompt, state]
    )
    demo.load(
        fn=classify_gallery, 
        inputs=[gr_flux_loras], 
        outputs=[gallery, gr_flux_loras]
    )

if __name__ == "__main__": 
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        mcp_server=args.mcp_server,
        inbrowser=True,
    )