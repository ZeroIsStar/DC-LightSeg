import torch
from torch.nn import init
from torch import nn, einsum
import numpy as np
import torch.nn.functional as F
from einops import rearrange

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        self.fc = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3),
            nn.Sigmoid()
        )
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)


class UNet_Nested(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=1, is_deconv=False, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        self.conv = nn.Conv2d(filters[4]*2,filters[4],kernel_size=1)

        # down_sampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)

        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        self.classfier = nn.Conv2d(filters[0]*4, n_classes, 1)

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32

        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return final
        else:
            return final_4


if __name__ == '__main__':
    print('#### Test Case ###')
    from torch.autograd import Variable
    from thop import profile
    import time
    x = torch.randn(1, 3, 256, 256).cuda()
    model = UNet_Nested().cuda()
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
