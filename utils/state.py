"""
全局状态管理模块

管理应用程序的全局状态变量
"""

# 全局状态变量
config = {}
transformer_choices = []
transformer_choices2 = []
t2i_choices = []
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
prompt_embeds_cache = {
    "key": None,
    "prompt_embeds": None,
    "prompt_embeds_mask": None,
    "negative_prompt_embeds": None,
    "negative_prompt_embeds_mask": None,
}

# 配置变量（需要在初始化时设置）
res_vram = 1000.0
openai_base_url = ""
openai_api_key = ""
base_url = ""
api_key = ""
model_name = ""
temperature = 0.8
top_p = 0.6
max_tokens = 16384
modelscope_api_key = ""
image_format = "png"

# 运行时变量（需要在初始化时设置）
device = "cuda"
dtype = None
args = None
mem = None

# 常量（文件路径）
CONFIG_FILE = "json/config.json"
EXAMPLES_FILE = "json/prompts.json"
TAB_MODELS_FILE = "json/tab_models.json"
