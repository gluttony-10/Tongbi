"""
生成器模块

包含所有图像生成相关的函数
"""

import gc
import io
import json
import time
import random
import base64
import datetime
import numpy as np
import torch
import requests
from io import BytesIO
from PIL import Image
from diffusers.utils import load_image
from diffusers.image_processor import VaeImageProcessor
import gradio as gr

import utils.state as state
from utils.model_loader import load_model
from utils.cache_utils import get_cached_prompt_embeds
from utils.image_utils import create_pnginfo, calculate_dimensions, upload_image_to_smms
from utils.config_utils import save_tab_model
from utils.camera_utils import build_camera_prompt


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
    image_urls=None,  # 新增：支持直接传入图片URL列表
):
    """ModelScope API 云端生成"""
    num_inference_steps = 50  
    true_cfg_scale = 4.0 
    results = []
    inference_times = []  # 记录每张图的生成时间
    total_start_time = time.time()  # 记录总开始时间
    
    # 显示ModelScope API生成提示信息
    mode_name_map = {
        "t2i_ms": "文生图",
        "edit_ms": "多图编辑" if image_urls else "图生图/编辑"
    }
    mode_name = mode_name_map.get(mode, "生成")
    # 根据选择的模型显示对应的API模型名称
    if mode == "t2i_ms":
        if "MS-Z-Image-Turbo" in transformer_dropdown:
            api_model_name = "Tongyi-MAI/Z-Image-Turbo"
        elif "MS-Qwen-Image-2512" in transformer_dropdown:
            api_model_name = "Qwen/Qwen-Image-2512"
        else:
            api_model_name = "Qwen/Qwen-Image"
    else:
        # 根据选择的模型显示对应的API模型名称
        if "MS-Qwen-Image-Edit-2511" in transformer_dropdown:
            api_model_name = "Qwen/Qwen-Image-Edit-2511"
        else:
            api_model_name = "Qwen/Qwen-Image-Edit-2509"  # 默认使用2509
    
    if image_urls:
        msg = f"🌐 使用ModelScope API云端生成 ({mode_name}) | 🤖 模型: {api_model_name} | 📝 提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''} | 🖼️ 图片数量: {len(image_urls)}张, 批量: {batch_images}张"
    else:
        msg = f"🌐 使用ModelScope API云端生成 ({mode_name}) | 🤖 模型: {api_model_name} | 📝 提示词: {prompt[:50]}{'...' if len(prompt) > 50 else ''} | 📊 分辨率: {width}x{height}, 批量: {batch_images}张"
    print(msg)
    yield results, msg
    
    resolutions = [
        (928, 1664),
        (1104, 1472),
        (1328, 1328),
        (1472, 1104),
        (1664, 928)
    ]
    # 如果使用image_urls（多图编辑），不需要处理base64编码
    if image_urls:
        # 多图编辑模式，使用URL，不需要分辨率处理
        pass
    elif image:
        pil_img = image.convert("RGB")
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
        "Authorization": f"Bearer {state.modelscope_api_key}",
        "Content-Type": "application/json",
    }
    # 禁用代理，避免代理连接问题
    proxies = {'http': None, 'https': None}
    
    # 第一步：一次性提交所有任务
    task_ids = []
    task_info = []  # 存储每个任务的信息（索引、种子、开始时间等）
    
    # 确定API模型ID（只计算一次）
    if mode == "t2i_ms":
        if "MS-Z-Image-Turbo" in transformer_dropdown:
            api_model_id = "Tongyi-MAI/Z-Image-Turbo"
        elif "MS-Qwen-Image-2512" in transformer_dropdown:
            api_model_id = "Qwen/Qwen-Image-2512"
        else:
            api_model_id = "Qwen/Qwen-Image"  # 默认使用MS-Qwen-Image
    else:
        # 根据选择的模型确定API模型ID
        if "MS-Qwen-Image-Edit-2511" in transformer_dropdown:
            api_model_id = "Qwen/Qwen-Image-Edit-2511"
        else:
            api_model_id = "Qwen/Qwen-Image-Edit-2509"  # 默认使用2509
    
    # 提交所有任务
    for i in range(batch_images):
        if state.stop_generation:
            state.stop_generation = False
            msg = f"✅ 生成已中止，已提交{len(task_ids)}个任务"
            print(msg)
            yield results, msg
            break
        
        img_start_time = time.time()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}.{state.image_format}"
        
        try:
            if mode == "t2i_ms":
                response = requests.post(
                    f"{base_url}v1/images/generations",
                    headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                    proxies=proxies,
                    data=json.dumps({
                        "model": api_model_id,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": num_inference_steps,
                        "true_cfg_scale": true_cfg_scale,
                        "size": f"{width}x{height}",
                        "seed": seed + i,
                    }, ensure_ascii=False).encode('utf-8')
                )
            elif mode == "edit_ms":
                if image_urls and len(image_urls) > 0:
                    response = requests.post(
                        f"{base_url}v1/images/generations",
                        headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                        proxies=proxies,
                        data=json.dumps({
                            "model": api_model_id,
                            "prompt": prompt,
                            "image_url": image_urls,
                        }, ensure_ascii=False).encode('utf-8')
                    )
                else:
                    response = requests.post(
                        f"{base_url}v1/images/generations",
                        headers={**common_headers, "X-ModelScope-Async-Mode": "true"},
                        data=json.dumps({
                            "model": api_model_id,
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
            task_ids.append(task_id)
            task_info.append({
                "index": i,
                "seed": seed + i,
                "start_time": img_start_time,
                "filename": filename,
                "task_id": task_id
            })
            
            msg = f"✅ 第{i+1}/{batch_images}张任务已提交 (任务ID: {task_id[:8]}...)"
            print(msg)
            yield results, msg
        except Exception as e:
            error_msg = f"❌ 第{i+1}/{batch_images}张任务提交失败: {str(e)}"
            print(error_msg)
            yield results, error_msg
    
    if not task_ids:
        msg = "❌ 没有成功提交任何任务"
        print(msg)
        yield results, msg
        return
    
    # 显示所有任务已提交
    msg = f"✅ 所有{len(task_ids)}个任务已提交完成 | ⏳ 正在等待云端生成..."
    print(msg)
    yield results, msg
    
    # 第二步：并行轮询所有任务状态
    completed_tasks = {}  # {task_id: image}
    task_status_map = {}  # {task_id: status}
    task_start_time = {}  # {task_id: start_time} 记录每个任务的开始时间
    last_status_update = time.time()
    
    while len(completed_tasks) < len(task_ids):
        if state.stop_generation:
            state.stop_generation = False
            msg = f"✅ 生成已中止，已完成{len(completed_tasks)}/{len(task_ids)}个任务"
            print(msg)
            yield results, msg
            break
        
        # 轮询所有未完成的任务
        for task_info_item in task_info:
            task_id = task_info_item["task_id"]
            if task_id in completed_tasks:
                continue
            
            try:
                result = requests.get(
                    f"{base_url}v1/tasks/{task_id}",
                    headers={**common_headers, "X-ModelScope-Task-Type": "image_generation"},
                    proxies=proxies,
                )
                result.raise_for_status()
                data = result.json()
                task_status = data.get("task_status", "UNKNOWN")
                task_status_map[task_id] = task_status
                
                # 记录任务开始时间（第一次轮询时）
                if task_id not in task_start_time:
                    task_start_time[task_id] = time.time()
                
                if task_status == "SUCCEED":
                    image_response = requests.get(data["output_images"][0], proxies=proxies)
                    image = Image.open(BytesIO(image_response.content))
                    completed_tasks[task_id] = image
                    
                    # 处理完成的图片
                    i = task_info_item["index"]
                    img_time = time.time() - task_info_item["start_time"]
                    inference_times.append(img_time)
                    
                    # 创建 PNG 元数据
                    pnginfo = create_pnginfo(
                        mode=mode,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        seed=task_info_item["seed"],
                        transformer_dropdown=transformer_dropdown,
                        num_inference_steps=num_inference_steps,
                        true_cfg_scale=true_cfg_scale,
                        width=width,
                        height=height,
                        image=image if mode == "edit_ms" else None,
                        generation_time=img_time
                    )
                    
                    # 根据格式选择保存方式
                    if state.image_format == "png":
                        image.save(task_info_item["filename"], pnginfo=pnginfo)
                    else:
                        # JPG和WEBP不支持pnginfo，需要单独保存元数据
                        image.save(task_info_item["filename"], format=state.image_format.upper())
                    results.append((task_info_item["filename"], None))
                    
                    msg = f"✅ 第{i+1}/{batch_images}张完成 (ModelScope API) | 🌱 种子: {task_info_item['seed']}, ⏱️ 耗时: {img_time:.2f}秒 | 💾 保存至: {task_info_item['filename']}"
                    print(msg)
                    yield results, msg
                    
                elif task_status == "FAILED":
                    i = task_info_item["index"]
                    error_msg = f"❌ 第{i+1}/{batch_images}张生成失败"
                    print(error_msg)
                    yield results, error_msg
                    completed_tasks[task_id] = None  # 标记为已完成（失败）
            except Exception as e:
                i = task_info_item["index"]
                error_msg = f"❌ 第{i+1}/{batch_images}张轮询失败: {str(e)}"
                print(error_msg)
                yield results, error_msg
        
        # 每20秒更新一次状态显示
        current_time = time.time()
        if current_time - last_status_update >= 20:
            pending_count = len(task_ids) - len(completed_tasks)
            status_summary = []
            for task_info_item in task_info:
                task_id = task_info_item["task_id"]
                if task_id not in completed_tasks:
                    status = task_status_map.get(task_id, "UNKNOWN")
                    elapsed_time = int(current_time - task_start_time.get(task_id, current_time))
                    status_summary.append(f"第{task_info_item['index']+1}张: {status}({elapsed_time}秒)")
            
            if status_summary:
                msg = f"⏳ 等待中: {pending_count}个任务未完成 | " + " | ".join(status_summary[:5])  # 最多显示5个
                if len(status_summary) > 5:
                    msg += f" | ...还有{len(status_summary)-5}个任务"
                print(msg)
                yield results, msg
            last_status_update = current_time
        
        time.sleep(5)  # 5秒间隔轮询
    
    # 生成完成后显示总结信息
    if results:
        total_time = time.time() - total_start_time
        avg_time = total_time / len(results) if results else 0
        msg = f"🎉 ModelScope API生成全部完成！ | 📊 共{len(results)}张，总耗时{total_time:.2f}秒，平均{avg_time:.2f}秒/张"
        print(msg)
        yield results, msg


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
    image=None, 
    mask_image=None, 
    strength=None,
    size_edit2=None, 
    reserve_edit2=None,
):
    """通用生成函数（本地模型）。显存等设置仅使用已保存的 state，不使用未保存的 UI 值。"""
    results = []
    inference_times = []  # 记录每张图的生成时间
    total_start_time = time.time()  # 记录总开始时间
    
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
    if (state.mode_loaded != mode or state.prompt_cache != prompt or state.negative_prompt_cache != negative_prompt or 
        state.transformer_loaded != transformer_dropdown or state.lora_loaded != lora_dropdown or
          state.lora_loaded_weights != lora_weights or state.image_loaded!=image):
        load_model(mode, transformer_dropdown, lora_dropdown, lora_weights, state.res_vram)
        state.prompt_cache, state.negative_prompt_cache, state.image_loaded = prompt, negative_prompt, image
    # 始终编码提示词（缓存命中时也需有 prompt_embeds 供 pipe 使用）
    if mode in ["editplus"]:
        prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = get_cached_prompt_embeds(
            mode, prompt, negative_prompt, true_cfg_scale, condition_images=condition_images
        )
    else:
        prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask = get_cached_prompt_embeds(
            mode, prompt, negative_prompt, true_cfg_scale, image=image
        )
    for i in range(batch_images):
        if state.stop_generation:
            state.stop_generation = False
            msg = f"✅ 生成已中止，最后种子数{seed+i-1}"
            print(msg)
            yield results, msg
            break
        
        # 记录单张图生成开始时间
        img_start_time = time.time()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"outputs/{timestamp}.{state.image_format}"
        
        with torch.no_grad():
            if mode == "t2i":
                output = state.pipe(
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
                output = state.pipe(
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
                output = state.pipe(
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
            elif mode == "editplus":
                output = state.pipe(
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
        
        # 生成完成，保存图像
        image = output.images[0]
        
        # 计算单张图生成时间
        img_time = time.time() - img_start_time
        inference_times.append(img_time)
        
        # 创建 PNG 元数据
        pnginfo = create_pnginfo(
            mode=mode,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed + i,
            transformer_dropdown=transformer_dropdown,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            width=width,
            height=height,
            strength=strength,
            lora_dropdown=lora_dropdown,
            lora_weights=lora_weights,
            image=image if mode in ["i2i", "inp", "editplus"] else None,
            generation_time=img_time
        )
        
        # 根据格式选择保存方式
        if state.image_format == "png":
            image.save(filename, pnginfo=pnginfo)
        else:
            # JPG和WEBP不支持pnginfo，需要单独保存元数据
            image.save(filename, format=state.image_format.upper())
        results.append((filename, None))
        
        # 显示进度信息：种子、保存路径、耗时
        msg = f"✅ 第{i+1}/{batch_images}张完成，种子{seed+i}，耗时{img_time:.2f}秒 | 保存至: {filename}"
        print(msg)
        yield results, msg
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    # 生成完成后显示总结信息
    if results:
        total_time = time.time() - total_start_time
        avg_time = total_time / len(results) if results else 0
        msg = f"🎉 全部完成！共{len(results)}张，总耗时{total_time:.2f}秒，平均{avg_time:.2f}秒/张"
        print(msg)
        yield results, msg


def generate_t2i(prompt, negative_prompt, width, height, num_inference_steps, 
                 batch_images, true_cfg_scale, seed_param, transformer_dropdown, 
                 lora_dropdown, lora_weights):
    """文生图生成函数。显存/图片格式等仅使用已保存的设置。"""
    # 保存当前TabItem的模型选择
    save_tab_model("t2i", transformer_dropdown)
    if "MS-Qwen-Image" in transformer_dropdown or "MS-Qwen-Image-2512" in transformer_dropdown or "MS-Z-Image-Turbo" in transformer_dropdown or "ModelScope" in transformer_dropdown:
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
        )


def generate_i2i(image, prompt, negative_prompt, width, height, num_inference_steps,
                 strength, batch_images, true_cfg_scale, seed_param, transformer_dropdown, 
                 lora_dropdown, lora_weights):
    """图生图生成函数。显存/图片格式等仅使用已保存的设置。"""
    # 保存当前TabItem的模型选择
    save_tab_model("i2i", transformer_dropdown)
    # MS-Qwen-Image只在文生图中可用，图生图不支持
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
    )


def generate_inp(image, prompt, negative_prompt, width, height, num_inference_steps,
                 strength, batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                 lora_dropdown, lora_weights):
    """局部重绘生成函数。显存/图片格式等仅使用已保存的设置。"""
    # 保存当前TabItem的模型选择
    save_tab_model("inp", transformer_dropdown)
    # MS-Qwen-Image只在文生图中可用，局部重绘不支持
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
    )


def generate_editplus2(image_editplus2, image_editplus3, image_editplus4, image_editplus5, prompt, negative_prompt, width, height, num_inference_steps,
                  batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                  lora_dropdown, lora_weights):
    """多图编辑生成函数。显存/图片格式等仅使用已保存的设置。"""
    # 保存当前TabItem的模型选择
    save_tab_model("editplus", transformer_dropdown)
    
    # 检查是否使用ModelScope API（多图编辑支持ModelScope）
    if "MS-Qwen-Image-Edit" in transformer_dropdown or "MS-Qwen-Image" in transformer_dropdown or "ModelScope" in transformer_dropdown:
        # 收集所有非空的图片
        image_list = [image_editplus2, image_editplus3, image_editplus4, image_editplus5]
        image_list = [img for img in image_list if img is not None]
        
        if len(image_list) == 0:
            yield [], "❌ 请至少上传一张图片"
            return
        
        # 上传图片到SM.MS图床
        msg = f"📤 正在上传{len(image_list)}张图片到图床..."
        print(msg)
        yield [], msg
        
        image_urls = []
        for idx, img in enumerate(image_list):
            # 转换为RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            # 上传到SM.MS
            url = upload_image_to_smms(img)
            if url:
                image_urls.append(url)
                print(f"  ✅ 第{idx+1}张图片上传成功: {url[:50]}...")
            else:
                yield [], f"❌ 第{idx+1}张图片上传失败，请重试"
                return
        
        if len(image_urls) == 0:
            yield [], "❌ 所有图片上传失败，请重试"
            return
        
        # 使用ModelScope API生成
        yield from modelscope_generate(
            mode="edit_ms",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            batch_images=batch_images, 
            seed_param=seed_param,
            transformer_dropdown=transformer_dropdown,
            image_urls=image_urls,  # 传递图片URL列表
        )
    else:
        # 使用本地模型生成
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
        )


def generate_camera_edit(image_camera, azimuth, elevation, distance, negative_prompt, width, height, num_inference_steps,
                  batch_images, true_cfg_scale, seed_param, transformer_dropdown,
                  lora_dropdown, lora_weights, additional_prompt=""):
    """
    3D相机控制生成函数。显存/图片格式等仅使用已保存的设置。
    Edit the camera angle of an image using Qwen Image Edit Plus with multi-angles LoRA.
    """
    # 保存当前TabItem的模型选择
    save_tab_model("camera", transformer_dropdown)
    if image_camera is None:
        raise gr.Error("请先上传图片")
    
    # Build camera prompt
    camera_prompt = build_camera_prompt(azimuth, elevation, distance)
    
    # Merge additional prompt if provided
    if additional_prompt and additional_prompt.strip():
        camera_prompt = f"{camera_prompt}, {additional_prompt.strip()}"
    
    # Convert image to RGB
    pil_image = image_camera.convert("RGB") if isinstance(image_camera, Image.Image) else Image.open(image_camera).convert("RGB")
    
    # MS-Qwen-Image只在文生图中可用，3D相机控制不支持
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
    )
