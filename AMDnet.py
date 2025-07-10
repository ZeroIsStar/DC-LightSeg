import torch
import torch.nn as nn
import torch.nn.functional as F


class AMDConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scales=[1, 2, 3], reduction=2):
        super().__init__()
        self.scales = scales
        self.out_channels = out_channels

        # 多尺度卷积分支
        self.conv_branches = nn.ModuleList()
        for s in scales:
            self.conv_branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=s * (kernel_size - 1) // 2, dilation=s),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # 动态权重生成器
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, len(scales)),
            nn.Softmax(dim=1)
        )

        # 特征融合
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(scales), out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        # 生成动态权重
        w = self.gap(x).view(b, c)
        weights = self.fc(w)

        # 多尺度特征加权融合
        features = []
        for i, conv in enumerate(self.conv_branches):
            scale_feat = conv(x) * weights[:, i].view(b, 1, 1, 1)
            features.append(scale_feat)

        fused = torch.cat(features, dim=1)
        return self.fusion_conv(fused)


class DownBlock(nn.Module):
    """下采样块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            AMDConv(in_channels, out_channels),
            AMDConv(out_channels, out_channels)
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    """上采样块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            AMDConv(in_channels, out_channels),
            AMDConv(out_channels, out_channels)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_channels=64):
        super().__init__()

        # 编码器
        self.down1 = DownBlock(in_channels, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)

        # 瓶颈层
        self.bottleneck = nn.Sequential(
            AMDConv(base_channels * 8, base_channels * 16),
            AMDConv(base_channels * 16, base_channels * 16)
        )

        # 解码器
        self.up1 = UpBlock(base_channels * 16, base_channels * 8)
        self.up2 = UpBlock(base_channels * 8, base_channels * 4)
        self.up3 = UpBlock(base_channels * 4, base_channels * 2)
        self.up4 = UpBlock(base_channels * 2, base_channels)

        # 输出层
        self.out = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 编码路径
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)

        # 瓶颈层
        x = self.bottleneck(x)

        # 解码路径
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out(x)


if __name__ == "__main__":
    # 验证网络结构
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=2).to(device)

    # 测试输入
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    output = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")