# 通臂 Tongbi
拿日月、缩千山、辨休咎、乾坤摩弄。

目前支持的功能有：
1.文生图
2.图生图
3.局部重绘
4.ControlNet
5.图像编辑
6.图像编辑（双图）
7.局部编辑
8.转换lora
9.图片信息

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 更新内容
250606 更新一版基于flux官方库的代码，然后放弃此代码，转向nunchaku

250708 更新一版基于nunchaku的代码

250924 放弃flux和nunchaku，改为使用Qwen-Image和mmgp

## 使用需求
1.显卡支持BF16

2.显存大于4G

## 安装依赖
```
git clone https://github.com/gluttony-10/Tongbi
cd Tongbi
conda create -n Tongbi python=3.10
conda activate Tongbi
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```
## 下载模型
```
python download.py
```
## 开始运行
```
python glut.py
```
## 参考项目
https://github.com/QwenLM/Qwen-Image

https://github.com/deepbeepmeep/mmgp
