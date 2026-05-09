import os
import torch
from torch.utils import data
from Data_loader.Data_augmentation import DataTransform
from PIL import Image
import numpy as np

# Nepal landslide dataset for semantic segmentation
class Nepal_Dataset(data.dataloader.Dataset):
    def __init__(self, dir = None, set = None, h_flip=True, v_flip=True, scale_random_crop=True, noise=True, rotate = True, erase = True):
        """
        :param dir: dataset directory
        :param set:['train', 'val', 'test']
        """
        self.files = []
        self.set = set
        self.transform = DataTransform(h_flip=h_flip, v_flip=v_flip, scale=scale_random_crop, noise=noise, rotate = rotate, erase = erase)
        self.img_dir = os.path.join(dir, "image")
        self.mask_dir = os.path.join(dir, "label")
        with open(dir + '/' + set + '.txt', 'r') as file:
            # 读取文件中的所有内容
            data_name = [line.strip() for line in file]
        self.files = []
        for name in data_name:
            img_dir = os.path.join(self.img_dir, name)
            mask_dir = os.path.join(self.mask_dir, os.path.splitext(name)[0] + '.png')
            self.files.append({
                "img": img_dir,
                "mask": mask_dir,
                "name": os.path.splitext(name)[0]
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
    from scipy import ndimage

    folder_path = r"E:\dataset\Nepal_landslide_dataset\label"  # 替换为你的文件夹路径

    # 获取所有文件名
    all_files = os.listdir(folder_path)

    # 筛选出所有 .png 文件
    mask_files = [file for file in all_files if file.endswith("png")]
    file_path = []

    # 遍历并处理每个 .h5 文件
    landslide_area = []
    for mask in mask_files:
        full_path = os.path.join(folder_path, mask)
        file_path.append(full_path)
    # 初始化长宽列表
    widths = []
    heights = []
    # 初始化滑坡像素数量列表
    landslide_pixel_counts = []

    # 遍历并处理每个 .png 文件
    for mask_file in mask_files:
        mask_path = os.path.join(folder_path, mask_file)
        label_image = Image.open(mask_path).convert('L')  # 转换为灰度图像
        # 将图像转换为数组
        mask_array = np.array(label_image.resize((224, 224), Image.NEAREST))
        # 使用连通区域标记
        labeled_array, num_features = ndimage.label(mask_array)
        # 遍历每个连通区域
        for i in range(1, num_features + 1):
            # 提取当前区域的坐标
            slip_y, slip_x = np.where(labeled_array == i)
            if len(slip_y) > 0 and len(slip_x) > 0:
                # 计算像素数量
                pixel_count = len(slip_y)
                landslide_pixel_counts.append(pixel_count)

    # 找到最小和最大滑坡像素数量
    min_pixel_count = min(landslide_pixel_counts) if landslide_pixel_counts else 0
    max_pixel_count = max(landslide_pixel_counts) if landslide_pixel_counts else 0

    print(f"最小滑坡像素数量: {min_pixel_count}")
    print(f"最大滑坡像素数量: {max_pixel_count}")

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
    ax.set_title('尼泊尔数据集滑坡尺寸及数量分布')
    # 设置坐标轴范围
    ax.set_xlim(0, 224)
    ax.set_ylim(0, 224)
    # 添加颜色条
    cb = fig.colorbar(img, ax=ax)
    cb.set_label('数量热力图')

    plt.tight_layout()
    plt.savefig(r'G:\周报\研二上\中文paper图\文中数据图\Nepal_landslide.png', dpi=450, bbox_inches='tight')
    plt.show()

# a = Nepal_Dataset(dir= r'E:\dataset\Nepal_landslide_dataset', set='train')
# img, label, name = a.__getitem__(0)
# print(img.shape, label.max(), name)
# label = label.numpy().astype(np.uint8)
# img = img.numpy().transpose(1,2,0).astype(np.float32)


