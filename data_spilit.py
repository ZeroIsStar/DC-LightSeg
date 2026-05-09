import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from PIL import Image
import h5py
import matplotlib.pyplot as plt

# # 数据路径
image_path = r'E:\\dataset\\new_data\\img'
label_path = r"E:\\dataset\\new_data\\mask"
text_path = r'E:\\dataset\\new_data'

# 获取所有图像和标签文件名
image_files = os.listdir(image_path)
label_files = os.listdir(label_path)

# 确保图像和标签文件名一致
assert len(image_files) == len(label_files)

# 提取类别分布信息
class_distribution = []
for label_file in label_files:
    with h5py.File(os.path.join(label_path, label_file), 'r') as hf:
        label = hf['mask'][:]
    unique, counts = np.unique(label, return_counts=True)
    class_dist = dict(zip(unique, counts / label.size))
    class_distribution.append(class_dist)
#
# # 将类别分布转换为DataFrame
class_df = pd.DataFrame(class_distribution)
class_df['image_name'] = image_files

# 填充缺失值（某些类别可能在某些图像中不存在）
class_df = class_df.fillna(0)

# 使用K-means聚类将图像分为不同的类别分布组
kmeans = KMeans(n_clusters=5, random_state=42)
class_df['cluster'] = kmeans.fit_predict(class_df.drop(columns=['image_name']))

# 使用分层抽样划分数据集
train_data, val_test_data = train_test_split(
    class_df,
    test_size=0.2,
    stratify=class_df['cluster'],
    random_state=42
)

val_data, test_data = train_test_split(
    val_test_data,
    test_size=0.5,
    stratify=val_test_data['cluster'],
    random_state=42
)

# 提取训练集、验证集和测试集的图像文件名
train_image_files = train_data['image_name'].tolist()
val_image_files = val_data['image_name'].tolist()
test_image_files = test_data['image_name'].tolist()

train_txt_path = os.path.join(text_path, 'train.txt')
val_txt_path = os.path.join(text_path, 'val.txt')
test_txt_path = os.path.join(text_path, 'test.txt')
with open(train_txt_path, 'w') as f:
    pass  # 不写入任何内容，只是创建一个空文件

with open(val_txt_path, 'w') as f:
    pass  # 不写入任何内容，只是创建一个空文件

with open(test_txt_path, 'w') as f:
    pass  # 不写入任何内容，只是创建一个空文件

f_train = open(train_txt_path, 'w')
f_val = open(val_txt_path, 'w')
f_test = open(test_txt_path, 'w')

for i in range(len(train_image_files)):
    f_train.write(train_image_files[i])
    f_train.write('\n')
for i in range(len(val_image_files)):
    f_val.write(val_image_files[i])
    f_val.write('\n')
for i in range(len(test_image_files)):
    f_test.write(test_image_files[i])
    f_test.write('\n')


f_train.close()
f_val.close()
f_test.close()

# 验证类别分布是否一致
print("train：")
print(train_data.drop(columns=['image_name', 'cluster']).mean())

print("val：")
print(val_data.drop(columns=['image_name', 'cluster']).mean())

print("test：")
print(test_data.drop(columns=['image_name', 'cluster']).mean())

# 可视化类别分布
for column in class_df.columns:
    if column not in ['image_name', 'cluster']:
        plt.figure(figsize=(10, 6))
        plt.hist(train_data[column], bins=30, alpha=0.5, label='trian')
        plt.hist(val_data[column], bins=30, alpha=0.5, label='val')
        plt.hist(test_data[column], bins=30, alpha=0.5, label='test')
        plt.title(f'{column} ')
        plt.xlabel(column)
        plt.ylabel('')
        plt.legend()
        plt.show()
