from torch import nn
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
import torch
import math


class Deform(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     padding=kernel_size // 2)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.GroupNorm(out_channels, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        off_set = self.offset_conv(x)
        x = self.deform_conv(x, off_set)
        x = self.act(self.bn(x))
        return x


class C1C3BR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.c3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.c3(self.c1(x))))


class SoftPool2D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(SoftPool2D, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size, stride)

    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = self.avgpool(x_exp)
        x = self.avgpool(x_exp * x)
        return x / x_exp_pool


class DRFM(nn.Module):
    def __init__(self, in_channels, out_channels, variant='A6'):
        """
        variant: 指定消融配置，可选 A0~A9
        """
        super(DRFM, self).__init__()
        self.variant = variant
        self.out_channels = out_channels
        self.act = nn.ReLU(inplace=True)  # 统一使用 ReLU 以便对比

        self.conv1x1_in = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 根据 variant 决定中间的卷积模块
        mid_channels = out_channels

        # ----- 定义中间层 -----
        if variant == 'A0':
            # 无中间卷积，直接 1x1 输出 + 残差
            self.mid = nn.Identity()
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A1':  # 3x3 普通卷积
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A2':  # 5x5 普通卷积
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A3':  # 7x7 普通卷积
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=7, padding=3),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A4':  # 只有 Deform 3x3
            self.mid = Deform(mid_channels, mid_channels, kernel_size=3)
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A5':  # 3x3 普通 + Deform
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True),
                Deform(mid_channels, mid_channels, kernel_size=3)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A6':  # 5x5 DW + Deform
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.SiLU(inplace=True),
                Deform(mid_channels, mid_channels, kernel_size=3)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A7':  # A6 但去掉残差
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.SiLU(inplace=True),
                Deform(mid_channels, mid_channels, kernel_size=3)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        elif variant == 'A8':  # A6 但将 Deform 替换为空洞卷积 3x3 dilation=2
            self.mid = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=5, padding=2, groups=mid_channels),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2),
                nn.BatchNorm2d(mid_channels, momentum=0.1),
                nn.ReLU(inplace=True)
            )
            self.conv1x1_out = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        else:
            raise ValueError(f'Unknown variant {variant}')

        # 是否使用残差（A7 不使用）
        self.use_residual = (variant != 'A7')

    def forward(self, x):
        identity = self.conv1x1_in(x)  # 所有 variant 都有这个投影
        out = self.mid(identity)
        out = self.conv1x1_out(out)
        if self.use_residual:
            out = out + identity
        return self.act(out)


class FPN_add(nn.Module):
    """FPN for feature fusion."""

    def __init__(self):
        super(FPN_add, self).__init__()

    def forward(self, P4, P3, P2, P1):
        out4 = P4
        out3 = F.interpolate(P4, scale_factor=2, mode='bilinear') + P3
        out2 = F.interpolate(out3, scale_factor=2, mode='bilinear') + P2
        out1 = F.interpolate(out2, scale_factor=2, mode='bilinear') + P1
        return out4, out3, out2, out1


class FPN_cat(nn.Module):
    """FPN for feature fusion."""

    def __init__(self):
        super(FPN_cat, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, 1)
        self.conv2 = nn.Conv2d(128, 64, 1)
        self.conv3 = nn.Conv2d(128, 64, 1)

    def forward(self, P4, P3, P2, P1):
        out4 = P4
        out3 = self.conv1(torch.cat([F.interpolate(P4, scale_factor=2, mode='bilinear'), P3], dim=1))
        out2 = self.conv2(torch.cat([F.interpolate(out3, scale_factor=2, mode='bilinear'), P2], dim=1))
        out1 = self.conv3(torch.cat([F.interpolate(out2, scale_factor=2, mode='bilinear'), P1], dim=1))
        return out4, out3, out2, out1


class BiFPN_add(nn.Module):
    """BiFPN for feature fusion."""

    def __init__(self, channels, domn=None):
        super(BiFPN_add, self).__init__()
        # Bottom-up convolutions
        self.P4_P3_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.P3_P2_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.P2_P1_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.P1_P2_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.P2_P3_w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.P3_P4_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.act = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        if domn is None:
            self.down = SoftPool2D(2, 2)
        elif domn == 'maxpool':
            self.down = nn.MaxPool2d(2, 2)
        elif domn == 'avgpool':
            self.down = nn.AvgPool2d(2, 2)
        self.epsilon = 1e-4

        self.conv4_td = C1C3BR(channels, channels, kernel_size=3)
        self.conv3_td = C1C3BR(channels, channels, kernel_size=3)
        self.conv2_td = C1C3BR(channels, channels, kernel_size=3)
        self.conv2_dt = C1C3BR(channels, channels, kernel_size=3)
        self.conv3_dt = C1C3BR(channels, channels, kernel_size=3)
        self.conv4_dt = C1C3BR(channels, channels, kernel_size=3)

    def forward(self, P4, P3, P2, P1):
        # Top-down pathway
        P31_W = self.act(self.P4_P3_w)
        weight = P31_W / (torch.sum(self.P4_P3_w, dim=0) + self.epsilon)
        P31 = self.conv4_td(self.up(P4) * weight[0] + P3 * weight[1])

        P21_W = self.act(self.P3_P2_w)
        weight = P21_W / (torch.sum(self.P3_P2_w, dim=0) + self.epsilon)
        P21 = self.conv3_td(self.up(P31) * weight[0] + P2 * weight[1])

        P11_W = self.act(self.P2_P1_w)
        weight = P11_W / (torch.sum(self.P2_P1_w, dim=0) + self.epsilon)
        P11 = self.conv2_td(self.up(P21) * weight[0] + P1 * weight[1])

        # Bottom-up pathway
        P22_W = self.act(self.P1_P2_w)
        weight = P22_W / (torch.sum(self.P1_P2_w, dim=0) + self.epsilon)
        P22 = self.conv2_dt(self.down(P11) * weight[0] + P21 * weight[1] + P2 * weight[2])

        P32_W = self.act(self.P2_P3_w)
        weight = P32_W / (torch.sum(self.P2_P3_w, dim=0) + self.epsilon)
        P32 = self.conv3_dt(self.down(P22) * weight[0] + P31 * weight[1] + P3 * weight[2])

        P41_W = self.act(self.P3_P4_w)
        weight = P41_W / (torch.sum(self.P3_P4_w, dim=0) + self.epsilon)
        P41 = self.conv4_dt(self.down(P32) * weight[0] + P4 * weight[1])
        return P41, P32, P22, P11


class BiFPN_cat(nn.Module):
    """BiFPN for feature fusion."""

    def __init__(self, channels, domn=None):
        super(BiFPN_cat, self).__init__()
        # Top-down convolutions
        self.conv1_td = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv4_td = C1C3BR(channels * 2, channels, kernel_size=3)
        self.conv3_td = C1C3BR(channels * 2, channels, kernel_size=3)
        self.conv2_td = C1C3BR(channels * 2, channels, kernel_size=3)
        self.conv2_dt = C1C3BR(channels * 3, channels, kernel_size=3)
        self.conv3_dt = C1C3BR(channels * 3, channels, kernel_size=3)
        self.conv4_dt = C1C3BR(channels * 2, channels, kernel_size=3)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        if domn is None:
            self.down = SoftPool2D(2, 2)
        elif domn == 'maxpool':
            self.down = nn.MaxPool2d(2, 2)
        elif domn == 'avgpool':
            self.down = nn.AvgPool2d(2, 2)

    def forward(self, P4, P3, P2, P1):
        # Top-down pathway
        P4_td = self.conv1_td(P4)
        P3_td = self.conv4_td(torch.cat([P3, self.up(P4_td)], dim=1))
        P2_td = self.conv3_td(torch.cat([P2, self.up(P3_td)], dim=1))
        P1_td = self.conv2_td(torch.cat([P1, self.up(P2_td)], dim=1))

        # Bottom-up pathway
        P2_bu = self.conv2_dt(torch.cat([P2, P2_td, self.down(P1_td)], dim=1))
        P3_bu = self.conv3_dt(torch.cat([P3, P3_td, self.down(P2_bu)], dim=1))
        P4_bu = self.conv4_dt(torch.cat([P4_td, self.down(P3_bu)], dim=1))
        return P4_bu, P3_bu, P2_bu, P1_td


class DC_light(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, variant='A6', decoder='bifpn_cat', domn=None):
        super(DC_light, self).__init__()
        self.in_channels = in_channels
        if domn is None:
            self.down = SoftPool2D(2, 2)
        elif domn == 'maxpool':
            self.down = nn.MaxPool2d(2, 2)
        elif domn == 'avgpool':
            self.down = nn.AvgPool2d(2, 2)

        self.filters = [64, 128, 256, 512]
        self.conv00 = DRFM(in_channels=in_channels, out_channels=self.filters[0], variant=variant)
        self.conv01 = DRFM(in_channels=self.filters[0], out_channels=self.filters[1], variant=variant)
        self.conv02 = DRFM(in_channels=self.filters[1], out_channels=self.filters[2], variant=variant)
        self.conv03 = DRFM(in_channels=self.filters[2], out_channels=self.filters[3], variant=variant)
        self.c1 = nn.Conv2d(self.filters[0], self.filters[0], 1)
        self.c2 = nn.Conv2d(self.filters[1], self.filters[0], 1)
        self.c3 = nn.Conv2d(self.filters[2], self.filters[0], 1)
        self.c4 = nn.Conv2d(self.filters[3], self.filters[0], 1)
        if decoder == 'fpn_add':
            self.decoder = FPN_add()
        elif decoder == 'fpn_cat':
            self.decoder = FPN_cat()
        elif decoder == 'bifpn_add':
            self.decoder = BiFPN_add(self.filters[0], domn=domn)
        else:
            self.decoder = BiFPN_cat(self.filters[0], domn=domn)
        self.Classifier = nn.Sequential(
            nn.Conv2d(self.filters[0] * 4, self.filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(self.filters[0], momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(self.filters[0], n_classes, kernel_size=1),
        )

    def forward(self, inputs):
        conv00 = self.conv00(inputs)
        conv01 = self.conv01(self.down(conv00))
        conv02 = self.conv02(self.down(conv01))
        conv03 = self.conv03(self.down(conv02))
        p4 = self.c4(conv03)
        p3 = self.c3(conv02)
        p2 = self.c2(conv01)
        p1 = self.c1(conv00)
        out4, out3, out2, out1 = self.decoder(p4, p3, p2, p1)
        out4 = F.interpolate(out4, scale_factor=8, mode='bilinear')
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear')
        out2 = F.interpolate(out2, scale_factor=2, mode='bilinear')
        totoal = torch.cat([out4, out3, out2, out1], dim=1)
        out = self.Classifier(totoal)
        return out


if __name__ == '__main__':
    print('#### Test Case ###')
    from thop import profile
    import time

    x = torch.randn(1, 3, 224, 224).cuda()
    # print(a(x).shape)
    model = DC_light().cuda()
    flops, params = profile(model, inputs=(x,))
    num_runs = 10
    total_time = 0
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
    print(f'params: {params / 1e6}M')
    print(model(x).shape)







