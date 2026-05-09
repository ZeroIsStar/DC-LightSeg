import torch
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import equalize  # 导入直方图均衡化函数




class DataTransform:
    def __init__(self,
                 h_flip=True,
                 v_flip=True,
                 rotate=True,
                 erase=True,
                 scale=True,
                 noise=True,
                 probability=0.5,
                 erase_params=(0.05, 0.2, 0.3),
                 scale_range=(0.8, 1.2),
                 noise_std=0.05):
        """
        改进的数据增强类，包含多项增强操作和边界检查

        参数说明:
            h_flip (bool): 水平翻转开关
            v_flip (bool): 垂直翻转开关
            rotate (bool): 旋转增强开关
            erase (bool): 随机擦除开关
            scale (bool): 缩放裁剪开关
            noise (bool): 高斯噪声开关
            probability (float): 基础应用概率
            erase_params (tuple): 擦除参数 (面积下限, 面积上限, 长宽比范围)
            scale_range (tuple): 缩放范围 (最小比例, 最大比例)
            noise_std (float): 噪声标准差上限
        """
        # 初始化增强开关
        self.h_flip = h_flip
        self.v_flip = v_flip
        self.rotate = rotate
        self.erase = erase
        self.scale = scale
        self.noise = noise
        self.probability = probability

        # 擦除参数校验
        if erase_params[0] > erase_params[1]:
            raise ValueError("擦除面积下限不能大于上限")
        self.sl, self.sh, self.r1 = erase_params

        # 缩放参数校验
        if scale_range[0] > scale_range[1]:
            raise ValueError("最小缩放比例不能大于最大比例")
        self.scale_range = scale_range

        # 噪声参数校验
        if noise_std < 0:
            raise ValueError("噪声标准差不能为负数")
        self.noise_std = noise_std

    def _apply_to_all(self, args, func, indices=None):
        """通用增强应用方法"""
        if indices is None:
            indices = range(len(args))
        return [func(arg) if i in indices else arg for i, arg in enumerate(args)]

    def __call__(self, *args):
        args = list(args)
        if not args:
            return tuple(args)

        # 获取基准尺寸
        c, h, w = args[0].shape

        # 水平翻转
        if self.h_flip and random.random() < self.probability:
            args = self._apply_to_all(args, TF.hflip)

        # 垂直翻转
        if self.v_flip and random.random() < self.probability:
            args = self._apply_to_all(args, TF.vflip)

        # 随机旋转
        if self.rotate and random.random() < self.probability:
            angle = random.choice([90, 180, 270])
            rotate_fn = lambda x: TF.rotate(x, angle)
            args = self._apply_to_all(args, rotate_fn)

        # 随机擦除（仅对图像）
        if self.erase and len(args) > 1 and random.random() < self.probability:
            args = self._apply_erase(args, h, w)

        # 缩放裁剪
        if self.scale and random.random() < self.probability:
            args = self._apply_scale_crop(args, h, w)

        # 高斯噪声（仅对图像）
        if self.noise and len(args) > 1 and random.random() < self.probability:
            args = self._apply_noise(args)

        return tuple(args)

    def _apply_erase(self, args, h, w):
        """改进的随机擦除实现"""
        # 只对非标签数据应用擦除
        image_args = args[:-1]
        label = args[-1]

        # 计算有效擦除区域
        target_area = random.uniform(self.sl, self.sh) * h * w
        aspect_ratio = random.uniform(1 / self.r1, self.r1)

        he = int(round((target_area * aspect_ratio) ** 0.5))
        we = int(round((target_area / aspect_ratio) ** 0.5))

        # 尺寸边界检查
        he = max(1, min(he, h - 1))
        we = max(1, min(we, w - 1))

        if he > 0 and we > 0:
            i = random.randint(0, h - he)
            j = random.randint(0, w - we)
            erase_mask = torch.zeros(1, h, w, device=args[0].device)
            erase_mask[:, i:i + he, j:j + we] = 1

            # 应用擦除到所有图像数据
            return [img * (1 - erase_mask) for img in image_args] + [label]

    def _apply_scale_crop(self, args, orig_h, orig_w):
        """改进的缩放裁剪实现"""
        scale = random.uniform(*self.scale_range)
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)

        # 确保缩放后尺寸不小于原始尺寸
        new_h = max(new_h, orig_h)
        new_w = max(new_w, orig_w)

        scaled = []
        for idx, tensor in enumerate(args):
            mode = 'bicubic' if idx < len(args) - 1 else 'nearest'
            # 添加批次维度进行插值
            scaled_tensor = F.interpolate(
                tensor.unsqueeze(0),
                size=(new_h, new_w),
                mode=mode,
                align_corners=False if mode != 'nearest' else None
            ).squeeze(0)
            scaled.append(scaled_tensor)

        # 随机裁剪
        h_start = random.randint(0, new_h - orig_h)
        w_start = random.randint(0, new_w - orig_w)

        return [
            t[:, h_start:h_start + orig_h, w_start:w_start + orig_w]
            for t in scaled
        ]

    def _apply_noise(self, args):
        """改进的高斯噪声实现"""
        image_args = args[:-1]
        label = args[-1]

        std = random.uniform(0, self.noise_std)
        return [img + torch.randn_like(img) * std for img in image_args] + [label]


# 测试用例
if __name__ == "__main__":
    # 初始化增强器
    dataau = DataTransform(
        scale_range=(0.8, 2.0),  # 包含缩小和放大
        erase_params=(0.02, 0.3, 0.2)
    )

    # 生成测试数据（包含多模态图像和标签）
    y = torch.rand(3, 256, 256).cuda()  # RGB图像
    label = torch.randint(0, 2, (1, 256, 256), dtype=torch.uint8).cuda()

    # 应用增强
    enhanced_y, enhanced_label = dataau(y, label)

    # 验证输出
    print(f"增强后图像尺寸: {enhanced_y.shape}")

    print(f"增强后标签尺寸: {enhanced_label.shape}")
    print(f"图像数据类型: {enhanced_y.dtype}")
    print(f"标签数据类型: {enhanced_label.dtype}")
