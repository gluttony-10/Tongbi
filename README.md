# 通臂 Tongbi
拿日月、缩千山、辨休咎、乾坤摩弄。

目前支持的功能有：
文生图
图生图
局部重绘
多图编辑
ControlNet预处理
转换lora
图库

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 使用需求
1.显存大于4G（4G显存需要配合核显使用）

2.显卡最好支持BF16。如果不支持将使用FP32精度，显存占用增大。
## 安装依赖
```
git clone https://github.com/gluttony-10/Tongbi
cd Tongbi
conda create -n Tongbi python=3.12
conda activate Tongbi
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
#pip install flash_attn --no-build-isolation
```
## 下载模型
```
modelscope download --model Gluttony10/Tongbi --local_dir ./models
```
## 开始运行
```
python glut.py
```
## 参考项目
https://github.com/QwenLM/Qwen-Image

https://github.com/deepbeepmeep/mmgp
