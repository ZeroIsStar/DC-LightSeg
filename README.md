# DC-Light
[Bijie数据地址](https://gpcv.whu.edu.cn/data/Bijie_pages.html)
[Nepal数据集地址](https://zenodo.org/records/3675410)
数据划分采取：基于标签聚类分层抽取的方式划分数据集
数据集结构：


├── Bijie/ # 毕节滑坡数据集
│ ├── image/ # 原始遥感影像（）
│ ├── mask/ # 像素级滑坡标注（二值掩膜）
│ ├── train.txt # 训练集样本
│ ├── val.txt # 验证集样本
│ └── test.txt # 测试集样本
│
└── Nepal_landslide_dataset/ # 尼泊尔滑坡数据集
├── image/ # 原始遥感影像
├── label/ # 像素级滑坡标注
├── train.txt # 训练集样本
├── val.txt # 验证集样本
└── test.txt # 测试集样本
