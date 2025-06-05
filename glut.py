import gradio as gr
import numpy as np
import random
import torch
from diffusers import  FluxPipeline, AutoencoderTiny, AutoencoderKL, FluxTransformer2DModel 
from transformers import T5EncoderModel, BitsAndBytesConfig

import argparse
from typing import Any, Dict, List, Optional, Union
import os
from glob import glob
import psutil
import time
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe


parser = argparse.ArgumentParser() 
parser.add_argument("--server_name", type=str, default="127.0.0.1", help="IP地址")
parser.add_argument("--server_port", type=int, default=7860, help="端口号")
parser.add_argument("--share", action="store_true", help="是否公开分享")
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
        print(f'\033[32m不支持BF16，使用FP16\033[0m')
        dtype = torch.float16
else:
    print(f'\033[32mCUDA不可用，启用CPU模式\033[0m')
    device = "cpu"

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 2048


def infer(prompt, seed=42, randomize_seed=False, width=1024, height=1024, guidance_scale=3.5, num_inference_steps=28):
    start_time = time.time()
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator().manual_seed(seed)
    for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator,
            output_type="pil",
            good_vae=good_vae,
        ):
            yield img, seed
    total_time = time.time() - start_time
    print(f"生成完毕，种子数: {seed}，生成时间{total_time:.2f}秒")


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


# FLUX pipeline function
@torch.inference_mode()
def flux_pipe_call_that_returns_an_iterable_of_images(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    timesteps: List[int] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    good_vae: Optional[Any] = None,
):
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._interrupt = False

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = self._execution_device

    # 3. Encode prompt
    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
    prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.base_image_seq_len,
        self.scheduler.config.max_image_seq_len,
        self.scheduler.config.base_shift,
        self.scheduler.config.max_shift,
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    # Handle guidance
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0]) if self.transformer.config.guidance_embeds else None

    # 6. Denoising loop
    for i, t in enumerate(timesteps):
        if self.interrupt:
            continue

        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        noise_pred = self.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=self.joint_attention_kwargs,
            return_dict=False,
        )[0]
        # Yield intermediate result
        latents_for_image = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_for_image = (latents_for_image / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents_for_image, return_dict=False)[0]
        yield self.image_processor.postprocess(image, output_type=output_type)[0]
        
        latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        torch.cuda.empty_cache()

    # Final image using good_vae
    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = (latents / good_vae.config.scaling_factor) + good_vae.config.shift_factor
    image = good_vae.decode(latents, return_dict=False)[0]
    self.maybe_free_model_hooks()
    torch.cuda.empty_cache()
    yield self.image_processor.postprocess(image, output_type=output_type)[0]


examples = [
    "a tiny astronaut hatching from an egg on the moon",
    "a cat holding a sign that says hello world",
    "an anime illustration of a wiener schnitzel",
]

model_dirs = [os.path.basename(d) for d in glob('models/diffusers/*') if os.path.isdir(d)]
model_dirs.sort()
transformer_files = [os.path.basename(f) for f in glob('models/transformers/*') if os.path.isfile(f)]
transformer_files.sort()
transformer_files = ["默认"] + model_dirs + transformer_files

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
    with gr.Column():
        with gr.Row():
            model_dir=gr.Dropdown(label="模型", choices=model_dirs, value=model_dirs[0] if model_dirs else None)
            transformer_file=gr.Dropdown(label="transformer", choices=transformer_files, value=transformer_files[0] if transformer_files else None)
        with gr.Tab("文生图"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(
                            label="提示词",
                            placeholder="输入你的提示词",
                        )
                        run_button = gr.Button("开始生成", variant="primary")
                        with gr.Accordion("高级设置"):
                            seed = gr.Slider(
                                label="Seed",
                                minimum=0,
                                maximum=MAX_SEED,
                                step=1,
                                value=0,
                            )
                            randomize_seed = gr.Checkbox(label="随机种子", value=True)
                            with gr.Row(): 
                                width = gr.Slider(
                                    label="宽度",
                                    minimum=256,
                                    maximum=MAX_IMAGE_SIZE,
                                    step=64,
                                    value=1024,
                                )
                                height = gr.Slider(
                                    label="高度",
                                    minimum=256,
                                    maximum=MAX_IMAGE_SIZE,
                                    step=64,
                                    value=1024,
                                )
                            with gr.Row():
                                guidance_scale = gr.Slider(
                                    label="引导强度",
                                    minimum=1,
                                    maximum=15,
                                    step=0.1,
                                    value=3.5,
                                )
                                num_inference_steps = gr.Slider(
                                    label="推理步数",
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    value=28,
                                )
                    result = gr.Image(label="生成图像", show_label=False)
                gr.Examples(
                    examples = examples,
                    fn = infer,
                    inputs = [prompt],
                    outputs = [result, seed],
                )

    gr.on(
        triggers=[run_button.click, prompt.submit],
        fn = infer,
        inputs = [prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps],
        outputs = [result, seed]
    )

def load_model():
    global pipe, good_vae
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, torch_dtype=torch.float32)
    taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        f"models/diffusers/{model_dir.value}",
        subfolder="transformer",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    text_encoder_2 = T5EncoderModel.from_pretrained(
        f"models/diffusers/{model_dir.value}",
        subfolder="text_encoder_2",
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    good_vae = AutoencoderKL.from_pretrained(
        f"models/diffusers/{model_dir.value}", 
        subfolder="vae", 
        quantization_config=quantization_config,
        torch_dtype=dtype,
    )
    pipe = FluxPipeline.from_pretrained(
        f"models/diffusers/{model_dir.value}",
        transformer=transformer,
        text_encoder_2=text_encoder_2,
        vae=taef1,
        torch_dtype=dtype, 
    ).to(device)
    pipe.enable_model_cpu_offload()
    pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)


if __name__ == "__main__": 
    load_model()
    demo.launch(
        server_name=args.server_name, 
        server_port=args.server_port,
        share=args.share, 
        inbrowser=True,
    )