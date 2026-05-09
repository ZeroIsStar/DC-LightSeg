""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True,padding_mode='replicate'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True,padding_mode = 'replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class D_DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True, padding_mode = 'replicate'),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True,padding_mode = 'replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.dw_conv = D_DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.dw_conv(self.maxpool(x))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.conv_list = 64

        self.inc = DoubleConv(n_channels, self.conv_list)
        self.down1 = Down(self.conv_list, self.conv_list*2)
        self.down2 = Down(self.conv_list*2, self.conv_list*4)
        self.down3 = Down(self.conv_list*4, self.conv_list*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.conv_list*8, self.conv_list*16 // factor)
        self.up1 = Up(self.conv_list*16, self.conv_list*8 // factor, bilinear)
        self.up2 = Up(self.conv_list*8, self.conv_list*4 // factor, bilinear)
        self.up3 = Up(self.conv_list*4, self.conv_list*2 // factor, bilinear)
        self.up4 = Up(self.conv_list*2, self.conv_list, bilinear)
        self.outc = nn.Sequential(
            DoubleConv(self.conv_list, self.conv_list),
            nn.Conv2d(self.conv_list, self.n_classes, kernel_size=1)
        )

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == "__main__":
    from thop import profile
    import time
    x = torch.randn(2, 3, 256, 256).cuda()
    model = UNet(3, 2, bilinear=True).cuda()
    num_runs = 10
    total_time = 0
    flops, params = profile(model, inputs=(x,))
    # 多次推理，计算平均推理时间
    for _ in range(num_runs):
        start_time = time.time()
        results = model(x)
        end_time = time.time()
        total_time += (end_time - start_time)
    # 计算平均推理时间
    avg_inference_time = total_time / num_runs
    # 计算FPS
    fps = 1 / avg_inference_time

    print(f"FPS: {fps:.2f} frames per second")
    print(f'FLOPs: {flops / 1e9}G')
    print(f'Params: {params / 1e6}M')
    print(model(x).shape)



