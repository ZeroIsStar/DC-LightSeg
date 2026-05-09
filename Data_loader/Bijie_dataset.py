import os
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from Data_loader.Data_augmentation import DataTransform
import torch.nn.functional as F

class Bijie_Dataset(data.dataloader.Dataset):
    def __init__(self, dir = None, set = None, h_flip=True, v_flip=True, scale_random_crop=True, noise=True, rotate = True, erase = True):
        """
        :param dir: dataset directory
        :param set:['train', 'val', 'test']
        """
        self.files = []
        self.set = set
        self.transform = DataTransform(h_flip=h_flip, v_flip=v_flip, scale=scale_random_crop, noise=noise, rotate = rotate, erase = erase)
        self.img_dir = os.path.join(dir, "image")
        self.mask_dir = os.path.join(dir, "mask")
        with open(dir + '/' + set + '.txt', 'r') as file:
            # 读取文件中的所有内容
            data_name = [line.strip() for line in file]
        self.files = []
        for name in data_name:
            img_dir = os.path.join(self.img_dir, name)
            mask_dir = os.path.join(self.mask_dir, name)
            self.files.append({
                "img": img_dir,
                "mask": mask_dir,
                "name": name
            })

    def __len__(self):
        # 返回数据集的长度
        return len(self.files)

    def normalize(self, data):
        min = data.min()
        max = data.max()
        x = (data - min) / (max - min)
        return x

    def __getitem__(self, index):
        datas = self.files[index]
        name = datas['name']
        img = Image.open(datas['img']).resize((224, 224), Image.NEAREST)
        label = np.array(Image.open(datas['mask']).resize((224, 224), Image.NEAREST))
        img = torch.as_tensor(np.array(img, np.float32).transpose((-1, 0, 1)))
        label = np.where(label == 255, 1, 0)
        label = torch.as_tensor(label, dtype=torch.uint8)
        # 数据加强
        if self.set == 'train':
            label = label.unsqueeze(0)
            img, label = self.transform(img, label)
            label = label.squeeze(0)
        img = self.normalize(img)
        label = label.long()
        return img, label, name

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from scipy import ndimage

    folder_path = r"E:\dataset\Bijie_landslide_dataset\landslide\mask"  # 替换为你的文件夹路径


    # 获取所有文件名
    all_files = os.listdir(folder_path)

    # 筛选出所有 .png 文件
    mask_files = [file for file in all_files if file.endswith("png")]
    file_path = []
    # 初始化滑坡和非滑坡像素计数
    landslide_counts = []
    non_landslide_counts = []

    # 遍历并处理每个 .png 文件
    for mask_file in mask_files:
        mask_path = os.path.join(folder_path, mask_file)
        label_image = Image.open(mask_path).resize((224, 224), Image.NEAREST)
        # 将图像转换为数组
        mask_array = np.array(label_image)
        # 统计滑坡和非滑坡像素
        landslide_count = np.sum(mask_array == 255)  # 假设滑坡区域的像素值为 255
        non_landslide_count = np.sum(mask_array == 0)  # 假设非滑坡区域的像素值为 0
        landslide_counts.append(landslide_count)
        non_landslide_counts.append(non_landslide_count)

    # 找到最小和最大滑坡像素数量
    min_landslide_count = min(landslide_counts)
    max_landslide_count = max(landslide_counts)

    print(f"最小滑坡像素数量: {min_landslide_count}")
    print(f"最大滑坡像素数量: {max_landslide_count}")
    # 初始化滑坡和非滑坡像元计数
    landslide_count = 0
    non_landslide_count = 0

    # 遍历并处理每个 .png 文件
    for mask_file in mask_files:
        mask_path = os.path.join(folder_path, mask_file)
        label_image = Image.open(mask_path).resize((224, 224), Image.NEAREST)
        # 将图像转换为数组
        mask_array = np.array(label_image)
        # 统计滑坡和非滑坡像元
        landslide_count += np.sum(mask_array == 255)  # 假设滑坡区域的像素值为 255
        non_landslide_count += np.sum(mask_array == 0)  # 假设非滑坡区域的像素值为 0

    # 计算滑坡与非滑坡的占比
    total_count = landslide_count + non_landslide_count
    landslide_ratio = landslide_count / total_count
    non_landslide_ratio = non_landslide_count / total_count
    # 遍历并处理每个 .h5 文件
    landslide_area = []
    for mask in mask_files:
        full_path = os.path.join(folder_path, mask)
        file_path.append(full_path)
    # 初始化长宽列表
    widths = []
    heights = []
    for i, file in enumerate(file_path):
        label_image = Image.open(file).resize((224, 224), Image.NEAREST)
        # 使用连通区域标记
        labeled_array, num_features = ndimage.label(label_image)
        # 遍历每个连通区域
        for i in range(1, num_features + 1):
            # 提取当前区域的坐标
            slip_y, slip_x = np.where(labeled_array == i)
            if len(slip_y) > 0 and len(slip_x) > 0:
                # 计算长宽
                min_x, max_x = np.min(slip_x), np.max(slip_x)
                min_y, max_y = np.min(slip_y), np.max(slip_y)
                slip_height = max_y - min_y + 1
                slip_width = max_x - min_x + 1
                widths.append(slip_width)
                heights.append(slip_height)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘制长宽的散点图
    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, _, img = ax.hist2d(widths, heights, bins=48, cmap='Oranges', range=[[0, 224], [0, 224]])
    # 添加坐标轴标签和标题
    ax.set_xlabel('滑坡宽')
    ax.set_ylabel('滑坡高')
    ax.set_title('毕节数据集滑坡尺寸及数量分布')
    # 设置坐标轴范围
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 224)

    # 添加颜色条
    cb = fig.colorbar(img, ax=ax)
    cb.set_label('数量热力图')
    plt.tight_layout()
    plt.savefig(r'bijie_landslide.png', dpi=450, bbox_inches='tight')
    plt.show()

# a = Bijie_Dataset(dir= r'E:\dataset\Bijie_landslide_dataset\landslide', set='train')
# img, label, name = a.__getitem__(0)
# print(img.shape, label.shape)
# label = label.numpy().astype(np.uint8)
# img = img.numpy().transpose(1,2,0).astype(np.float32)



