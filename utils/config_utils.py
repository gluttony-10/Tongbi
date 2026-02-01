"""
配置管理工具函数

包含配置保存、加载、示例管理等功能
"""

import os
import json


# 配置文件路径（从 state 模块导入）
import utils.state as state
CONFIG_FILE = state.CONFIG_FILE
EXAMPLES_FILE = state.EXAMPLES_FILE
TAB_MODELS_FILE = state.TAB_MODELS_FILE


def initialize_examples_file():
    """
    初始化 EXAMPLES_FILE 文件，确保每个 TabItem 都有默认提示词
    """
    default_examples = {
        "t2i": ["选择保存过的提示词"],
        "i2i": ["选择保存过的提示词"],
        "inp": ["选择保存过的提示词"],
        "editplus": ["选择保存过的提示词"],
        "camera": ["选择保存过的提示词"]
    }
    
    if not os.path.exists(EXAMPLES_FILE):
        try:
            dir_path = os.path.dirname(EXAMPLES_FILE)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
                json.dump(default_examples, f, ensure_ascii=False, indent=2)
            print(f"✅ 已创建 {EXAMPLES_FILE} 并写入默认示例")
        except IOError as e:
            print(f"⚠️ 创建 {EXAMPLES_FILE} 失败: {e}")
    else:
        # 检查现有文件是否包含所有必需的标签
        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # 如果文件损坏或为空，重新创建
            print(f"⚠️ 读取 {EXAMPLES_FILE} 失败: {e}，将重新创建")
            existing_data = {}
        
        # 确保所有标签都存在
        updated = False
        for tab in default_examples.keys():
            if tab not in existing_data:
                existing_data[tab] = default_examples[tab]
                updated = True
                
        if updated:
            try:
                with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)
                print(f"✅ 已更新 {EXAMPLES_FILE}，添加缺失的标签")
            except IOError as e:
                print(f"⚠️ 更新 {EXAMPLES_FILE} 失败: {e}")


def load_tab_models():
    """加载所有标签的模型配置"""
    # 如果新文件存在，直接读取
    if os.path.exists(TAB_MODELS_FILE):
        try:
            with open(TAB_MODELS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 读取 {TAB_MODELS_FILE} 失败: {e}")
            return {}
    
    # 如果新文件不存在，返回空字典
    return {}


def save_tab_model(tab_name, model_name):
    """保存指定TabItem的模型选择到独立的JSON文件"""
    # 读取现有配置
    tab_models = {}
    if os.path.exists(TAB_MODELS_FILE):
        try:
            with open(TAB_MODELS_FILE, "r", encoding="utf-8") as f:
                tab_models = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 读取 {TAB_MODELS_FILE} 失败: {e}，将创建新文件")
            tab_models = {}
    
    # 更新对应标签的模型
    tab_models[tab_name] = model_name
    
    # 确保目录存在
    dir_path = os.path.dirname(TAB_MODELS_FILE)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    
    # 保存到新文件
    try:
        with open(TAB_MODELS_FILE, "w", encoding="utf-8") as f:
            json.dump(tab_models, f, ensure_ascii=False, indent=2)
    except IOError as e:
        print(f"⚠️ 保存 {TAB_MODELS_FILE} 失败: {e}")


def load_examples(tab_name):
    """加载指定TabItem的提示词示例列表"""
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                examples_data = json.load(f)
                # 返回对应TabItem的示例列表，如果不存在则返回空列表
                return examples_data.get(tab_name, [])
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 读取 {EXAMPLES_FILE} 失败: {e}")
            return []
    return []


def save_example(prompt, tab_name):
    """保存提示词示例到指定TabItem"""
    # 读取现有数据或初始化空字典
    examples_data = {}
    if os.path.exists(EXAMPLES_FILE):
        try:
            with open(EXAMPLES_FILE, "r", encoding="utf-8") as f:
                examples_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ 读取 {EXAMPLES_FILE} 失败: {e}，将创建新文件")
            examples_data = {}
    
    # 确保当前TabItem有列表
    if tab_name not in examples_data:
        examples_data[tab_name] = []
    
    # 如果提示词不在列表中，则添加
    if prompt and prompt not in examples_data[tab_name]:
        examples_data[tab_name].append(prompt)
        try:
            dir_path = os.path.dirname(EXAMPLES_FILE)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(EXAMPLES_FILE, "w", encoding="utf-8") as f:
                json.dump(examples_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"⚠️ 保存 {EXAMPLES_FILE} 失败: {e}")
    
    # 返回更新后的列表
    return examples_data.get(tab_name, [])
