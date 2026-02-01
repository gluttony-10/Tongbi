# Utils 工具模块

这个文件夹包含 Tongbi 项目的辅助工具和组件。

## 文件结构

```
utils/
├── __init__.py           # 包初始化文件，导出所有公共接口
├── camera_control.py     # 3D 相机控制组件
├── prompt_enhancer.py    # 提示词增强功能
└── lora_utils.py         # LoRA 权重加载和合并工具
```

## 模块说明

### camera_control.py

**CameraControl3D** - 基于 Three.js 的交互式 3D 相机视角控制组件

- 提供方位角（Azimuth）、仰角（Elevation）和距离（Distance）的可视化调节
- 支持鼠标和触摸操作
- 自动生成多视角提示词（如 "front view", "low-angle shot", "close-up" 等）
- 可加载用户图像作为参考平面
- 适用于 Gradio 界面

**使用示例：**
```python
from utils.camera_control import CameraControl3D

# 创建 3D 相机控制组件
camera_control = CameraControl3D(
    value={"azimuth": 0, "elevation": 0, "distance": 1.0},
    imageUrl=None  # 可选：图像 URL
)
```

### prompt_enhancer.py

**提示词增强功能** - 使用 AI 模型优化和扩展用户输入的提示词

包含以下函数：
- `enhance_prompt()` - 增强普通提示词
- `enhance_prompt_edit2()` - 增强编辑模式的提示词
- `update_config()` - 更新增强器配置

**使用示例：**
```python
from utils.prompt_enhancer import enhance_prompt

# 增强提示词
enhanced = enhance_prompt(
    original_prompt="a cat",
    api_key="your_api_key",
    base_url="https://api.example.com"
)
```

### lora_utils.py

**LoRA 工具函数** - 用于 Qwen-Image 模型的 LoRA 权重加载和合并

包含以下函数：
- `build_lora_names()` - 构建 LoRA 权重的名称
- `load_and_merge_lora_weight()` - 加载并合并 LoRA 权重到模型中
- `load_and_merge_lora_weight_from_safetensors()` - 从 safetensors 文件加载并合并 LoRA 权重

**使用示例：**
```python
from utils.lora_utils import load_and_merge_lora_weight_from_safetensors

# 从文件加载 LoRA 权重
model = load_and_merge_lora_weight_from_safetensors(
    model=transformer,
    lora_weight_path="models/lora/my_lora.safetensors"
)
```

**技术说明：**
- 支持原生权重格式和标准格式
- 自动计算 LoRA 缩放因子（scaling_factor = alpha / rank）
- 使用 safetensors 格式确保安全加载
- 验证权重数据类型（必须为 float32）

## 导入方式

### 从包导入（推荐）
```python
from utils import (
    CameraControl3D,
    enhance_prompt,
    enhance_prompt_edit2,
    update_config,
    build_lora_names,
    load_and_merge_lora_weight,
    load_and_merge_lora_weight_from_safetensors
)
```

### 从子模块导入
```python
from utils.camera_control import CameraControl3D
from utils.prompt_enhancer import enhance_prompt, enhance_prompt_edit2, update_config
from utils.lora_utils import load_and_merge_lora_weight_from_safetensors
```

## 维护说明

- 所有新的工具函数和组件应该添加到这个文件夹中
- 确保在 `__init__.py` 中导出公共接口
- 保持模块的独立性，减少相互依赖
- 为新功能添加文档说明
- 为函数添加详细的 docstring，包括参数、返回值和使用示例
