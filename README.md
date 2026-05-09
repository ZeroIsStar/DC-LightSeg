# DC-Light
[Bijie数据地址](https://gpcv.whu.edu.cn/data/Bijie_pages.html)

[Nepal数据集地址](https://zenodo.org/records/3675410)

数据划分采取：基于标签聚类分层抽取的方式划分数据集

数据集结构：
```bash
Bijie/
├── image/
├── mask/
├── train.txt
├── val.txt
└── test.txt

Nepal_landslide_dataset/
├── image/
├── label/
├── train.txt
├── val.txt
└── test.txt
```
## 📦 环境依赖

### 核心依赖
```txt
torch==2.0.0+cu118
torchvision==0.15.1+cu118
timm==1.0.24
Pillow==9.4.0
tifffile==2023.7.10
mamba-ssm==2.2.2
numpy==1.24.2
scipy==1.10.1
matplotlib==3.7.1
pandas==2.0.0
h5py==3.11.0

# 模型评估与工具
tensorboard==2.12.0
thop==0.1.1.post2209072238
torchsummary==1.5.1
einops==0.8.1
```
# Training Process Monitoring
```bash
tensorboard --logdir=tf-logs
```
