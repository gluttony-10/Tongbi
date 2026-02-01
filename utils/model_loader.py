"""
模型加载工具模块

包含模型加载、LoRA加载等功能
"""

import os
import gc
import time
import math
import torch
import safetensors.torch
from mmgp import offload
from diffusers import QwenImageTransformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers import QwenImageImg2ImgPipeline, QwenImagePipeline, QwenImageInpaintPipeline, QwenImageEditPlusPipeline
from transformers import Qwen2_5_VLForConditionalGeneration
import utils.state as state
from utils.model_downloader import check_and_download_model
from utils.lora_utils import build_lora_names, load_and_merge_lora_weight, load_and_merge_lora_weight_from_safetensors


def load_model(mode, transformer_dropdown, lora_dropdown, lora_weights, res_vram):
    """
    加载和配置模型
    
    参数:
        mode: 生成模式 (t2i, i2i, inp, editplus)
        transformer_dropdown: transformer 模型名称
        lora_dropdown: LoRA 模型列表
        lora_weights: LoRA 权重字符串
        res_vram: 保留显存大小 (MB)
    """
    # 导入全局变量
    import utils.state as state
    
    # 清理内存和显存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 计算显存预算
    res_vram = float(res_vram)
    if torch.cuda.is_available():
        free_memory, _ = torch.cuda.mem_get_info(0)
        budgets = int(free_memory / 1048576 - res_vram)
        print(f"💾 可用显存: {free_memory / 1073741824:.2f}GB, 分配预算: {budgets / 1024:.2f}GB")
    else:
        budgets = 0
    
    # Scheduler 配置
    scheduler_config = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    
    # Pipeline 类映射
    PIPELINE_CLASSES = {
        "t2i": QwenImagePipeline,
        "i2i": QwenImageImg2ImgPipeline,
        "inp": QwenImageInpaintPipeline,
        "editplus": QwenImageEditPlusPipeline,
    }
    
    # 判断是否需要重新加载模型
    need_reload = (
        state.pipe is None or 
        state.mode_loaded != mode or 
        state.transformer_loaded != transformer_dropdown or 
        state.lora_loaded != lora_dropdown or 
        state.lora_loaded_weights != lora_weights
    )
    
    if not need_reload:
        print("✅ 模型已加载，无需重新加载")
        return
    
    print(f"🔄 开始加载模型 [模式: {mode}, 模型: {transformer_dropdown}]")
    load_start_time = time.time()
    
    try:
        # 检查并下载模型（如果是本地模型）
        if not transformer_dropdown.startswith("MS-"):
            print("🔍 检查模型是否存在...")
            success, msg = check_and_download_model(transformer_dropdown)
            if not success:
                raise ValueError(msg)
            print(msg)
        
        # 卸载旧模型
        if state.pipe is not None:
            print("🗑️ 卸载旧模型...")
            state.pipe.unload_lora_weights()
            state.mmgp.release()
        
        # 更新全局状态
        state.mode_loaded, state.transformer_loaded, state.lora_loaded, state.lora_loaded_weights = (
            mode, transformer_dropdown, lora_dropdown, lora_weights
        )
        
        # 1. 加载 Text Encoder
        print("📝 加载 Text Encoder...")
        text_encoder = offload.fast_load_transformers_model(
            f"{state.model_id}/text_encoder/text_encoder-mmgp.safetensors",
            do_quantize=False,
            modelClass=Qwen2_5_VLForConditionalGeneration,
            forcedConfigPath=f"{state.model_id}/text_encoder/config.json",
        )
        
        # 2. 加载 Transformer
        print(f"🎨 加载 Transformer: {transformer_dropdown}")
        # 如果模型名没有 .safetensors 后缀，自动添加（用于本地模型）
        if not transformer_dropdown.startswith("MS-") and not transformer_dropdown.endswith(".safetensors"):
            transformer_dropdown_full = f"{transformer_dropdown}.safetensors"
        else:
            transformer_dropdown_full = transformer_dropdown
        
        if "mmgp" not in transformer_dropdown_full:
            raise ValueError("❌ 请使用 mmgp 转换后保存的模型")
        
        transformer = offload.fast_load_transformers_model(
            f"models/transformer/{transformer_dropdown_full}",
            do_quantize=False,
            modelClass=QwenImageTransformer2DModel,
            forcedConfigPath=f"{state.model_id}/transformer/config.json",
        )
        
        # 3. 加载 Scheduler
        if "Lightning" in transformer_dropdown:
            print("⚡ 使用 Lightning Scheduler（加速版）")
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
        else:
            print("🌊 使用标准 Scheduler")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(state.model_id, subfolder="scheduler")
        
        # 4. 初始化 Pipeline
        pipeline_class = PIPELINE_CLASSES.get(mode)
        if pipeline_class is None:
            raise ValueError(f"❌ 不支持的模式: {mode}")
        
        print(f"🔧 初始化 Pipeline: {pipeline_class.__name__}")
        
        state.pipe = pipeline_class.from_pretrained(
            state.model_id,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            torch_dtype=state.dtype,
        )
        
        # 5. 配置进度条
        if mode in ["editplus"]:
            state.pipe.set_progress_bar_config(disable=None)
        
        # 6. 加载 LoRA
        if lora_dropdown:
            print(f"🎯 加载 {len(lora_dropdown)} 个 LoRA 模型...")
        load_lora(lora_dropdown, lora_weights)
        
        # 7. 配置 MMGP（显存管理和量化）
        print("⚙️ 配置 MMGP（显存管理）...")
        import psutil
        mem = psutil.virtual_memory()
        pinned_models = ["text_encoder", "transformer"] if mem.total/1073741824 > 60 else "transformer"
        state.mmgp = offload.all(
            state.pipe,
            pinnedMemory=pinned_models,
            budgets={'*': budgets},
            extraModelsToQuantize=["text_encoder"],
            compile=True if state.args.compile else False,
        )
        
        # 8. 设置注意力后端
        if state.device == "cuda":
            if torch.cuda.get_device_capability()[0] >= 8:
                state.pipe.transformer.set_attention_backend("flash")
                print("⚡ 使用 Flash Attention 加速")
            else:
                state.pipe.transformer.set_attention_backend("native")
                print("🔧 使用标准 Attention")
        
        # 9. 启用 Channels Last 内存格式
        if state.device == "cuda" and hasattr(state.pipe, 'transformer'):
            try:
                state.pipe.transformer = state.pipe.transformer.to(memory_format=torch.channels_last)
                print("✅ Channels Last 内存格式已启用")
            except Exception as e:
                print(f"⚠️ Channels Last 启用失败: {e}")
        
        # 加载完成
        load_time = time.time() - load_start_time
        print(f"✅ 模型加载完成！耗时 {load_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def load_lora(lora_dropdown, lora_weights):
    """
    加载和配置 LoRA 模型（加载前自动转换）
    
    参数:
        lora_dropdown: LoRA 模型列表
        lora_weights: LoRA 权重字符串（逗号分隔）
    """
    if not lora_dropdown:
        return
    
    import utils.state as state
    
    adapter_names = []
    adapter_weights = []
    
    # 解析权重字符串
    weights = [float(w) for w in lora_weights.split(',')] if lora_weights else []
    
    # 加载每个 LoRA 模型
    for idx, lora_name in enumerate(lora_dropdown):
        try:
            lora_path = f"models/lora/Qwen-Image/{lora_name}"
            
            # 加载前自动转换（如果需要）
            converted_path = _convert_lora_file(lora_path)
            
            # 获取适配器名称（使用转换后的文件名，但去掉 _diffusers 后缀）
            base_name = os.path.splitext(os.path.basename(converted_path))[0]
            if "_diffusers" in base_name:
                adapter_name = base_name.replace("_diffusers", "")
            else:
                adapter_name = base_name
            adapter_names.append(adapter_name)
            
            weight = weights[idx] if idx < len(weights) else 1.0
            adapter_weights.append(weight)
            
            state.pipe.load_lora_weights(converted_path, adapter_name=adapter_name)
            print(f"  ✅ {lora_name} (权重: {weight})")
            
        except Exception as e:
            print(f"  ❌ {lora_name} 加载失败: {str(e)}")
    
    # 设置适配器
    if adapter_names:
        state.pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
        print(f"✅ LoRA 加载完成，共 {len(adapter_names)} 个模型")


def _convert_lora_file(lora_path):
    """
    自动转换单个 LoRA 文件（如果需要）
    
    参数:
        lora_path: LoRA 文件路径（相对于 models/lora/Qwen-Image/ 或绝对路径）
    
    返回:
        转换后的文件路径（如果已转换或不需要转换，返回原路径或已存在的转换文件路径）
    """
    # 确保路径是绝对路径
    if not os.path.isabs(lora_path):
        lora_path = os.path.abspath(lora_path)
    
    # 检查是否已经是转换后的文件（包含 _diffusers 后缀）
    base_name = os.path.splitext(os.path.basename(lora_path))[0]
    if "_diffusers" in base_name:
        # 已经是转换后的文件，直接返回
        return lora_path
    
    # 检查转换后的文件是否已存在
    output_dir = os.path.dirname(lora_path)
    output_filename = f"{base_name}_diffusers.safetensors"
    output_path = os.path.join(output_dir, output_filename)
    if os.path.exists(output_path):
        # 转换后的文件已存在，返回转换后的路径
        return output_path
    
    # 读取LoRA文件并检查是否需要转换
    try:
        lora_data = safetensors.torch.load_file(lora_path, device="cpu")
        
        # 检查是否需要转换（查找需要转换的key格式）
        needs_conversion = False
        for key in lora_data.keys():
            if ('.lora_A.' in key or '.lora_B.' in key or 
                'diffusion_model.transformer_blocks.' in key or
                'lora_unet_transformer_blocks_' in key or
                (key.startswith('transformer_blocks.') and not key.startswith('transformer.transformer_blocks.'))):
                needs_conversion = True
                break
        
        if not needs_conversion:
            # 不需要转换，直接返回原路径
            return lora_path
        
        # 需要转换，执行转换
        print(f"🔄 检测到需要转换的 LoRA 文件: {os.path.basename(lora_path)}，开始自动转换...")
        converted_dict = {}
        for key, value in lora_data.items():
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

        # 保存转换后的文件
        os.makedirs(output_dir, exist_ok=True)
        safetensors.torch.save_file(converted_dict, output_path)
        print(f"✅ {output_filename} 转换完成")
        return output_path
        
    except Exception as e:
        print(f"⚠️ 转换 LoRA 文件时出错: {str(e)}，尝试使用原文件")
        return lora_path
