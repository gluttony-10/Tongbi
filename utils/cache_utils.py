"""
缓存工具模块

包含提示词缓存、图像哈希等功能
"""

import utils.state as state


def get_image_hash(img):
    """获取图像的简单哈希值用于缓存"""
    if img is None:
        return None
    # 对于 dict 类型的输入（如 ImageEditor 组件）
    if isinstance(img, dict):
        img = img.get("background", img)
    # 使用图像尺寸和部分像素数据生成简单哈希
    if hasattr(img, 'size') and hasattr(img, 'mode') and hasattr(img, 'tobytes'):
        return hash((img.size, img.mode, img.tobytes()[:1000]))
    return None


def get_cached_prompt_embeds(mode, prompt, negative_prompt, true_cfg_scale, image=None, condition_images=None):
    """获取缓存的 prompt_embeds，避免重复编码"""
    # 生成缓存键
    image_hash = get_image_hash(image) if image is not None else get_image_hash(condition_images)
    cache_key = (mode, prompt, negative_prompt if true_cfg_scale > 1 else None, image_hash)
    
    # 检查缓存
    if state.prompt_embeds_cache["key"] == cache_key and state.prompt_embeds_cache["prompt_embeds"] is not None:
        print("📦 使用缓存的 prompt_embeds")
        return (
            state.prompt_embeds_cache["prompt_embeds"],
            state.prompt_embeds_cache["prompt_embeds_mask"],
            state.prompt_embeds_cache["negative_prompt_embeds"],
            state.prompt_embeds_cache["negative_prompt_embeds_mask"],
        )
    
    # 编码新的提示词
    print("🔄 编码提示词...")
    import torch
    with torch.inference_mode():
        if mode in ["editplus"]:
            prompt_embeds, prompt_embeds_mask = state.pipe.encode_prompt(image=condition_images, prompt=prompt)
            if true_cfg_scale > 1:
                negative_prompt_embeds, negative_prompt_embeds_mask = state.pipe.encode_prompt(image=condition_images, prompt=negative_prompt)
            else:
                negative_prompt_embeds, negative_prompt_embeds_mask = None, None
        else:
            prompt_embeds, prompt_embeds_mask = state.pipe.encode_prompt(prompt)
            if true_cfg_scale > 1:
                negative_prompt_embeds, negative_prompt_embeds_mask = state.pipe.encode_prompt(negative_prompt)
            else:
                negative_prompt_embeds, negative_prompt_embeds_mask = None, None
    
    # 更新缓存
    state.prompt_embeds_cache["key"] = cache_key
    state.prompt_embeds_cache["prompt_embeds"] = prompt_embeds
    state.prompt_embeds_cache["prompt_embeds_mask"] = prompt_embeds_mask
    state.prompt_embeds_cache["negative_prompt_embeds"] = negative_prompt_embeds
    state.prompt_embeds_cache["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
    
    return prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask
