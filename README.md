# 通臂 Tongbi
拿日月、缩千山、辨休咎、乾坤摩弄。

目前支持的功能有：
1.图像生成
2.图生图
3.批量生成

一键包详见 [bilibili@十字鱼](https://space.bilibili.com/893892)

## 更新内容
250606 更新一版基于flux官方库的代码，然后放弃此代码，转向nunchaku
250708 更新一版基于nunchaku的代码
## 安装依赖
```
git clone https://github.com/gluttony-10/Tongbi
cd Tongbi
conda create -n Tongbi python=3.10
conda activate Tongbi
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/mit-han-lab/nunchaku
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
https://github.com/black-forest-labs/flux

https://github.com/mit-han-lab/nunchaku

https://huggingface.co/spaces/black-forest-labs/FLUX.1-Kontext-Dev

https://huggingface.co/spaces/AlekseyCalvin/flux-kontext-SilverAgePoets-LoRAs

