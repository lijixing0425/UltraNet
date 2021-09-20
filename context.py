# -*- coding: utf-8 -*-
# @Time    : 21-6-10 下午12:35
# @Author  : lijixing&&gongyuxin
# @FileName: context.py
# @Software: PyCharm
# @Github £º https://github.com/lijixing0425/Ultranet


import torch
import torch.nn as nn
import torch.nn.functional as F


class paired_conv(nn.Module):
    def __init__(self, in_ch, kernel_size):
        super(paired_conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_ch, 1, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_ch, 1, kernel_size=2, padding=0)
        self.conv3 = nn.Conv2d(in_ch, 1, kernel_size=3, padding=1)

    def forward(self, x):
        N, C, H, W = x.shape
        x = F.unfold(x, self.kernel_size, stride=1, padding=(self.kernel_size - 1)//2)
        x = x.permute(0, 2, 1).reshape(-1, C, self.kernel_size, self.kernel_size)
        x_mid = x[..., self.kernel_size//2, self.kernel_size//2].unsqueeze(-1).unsqueeze(-1)
        x = x - x_mid
        x = (self.conv1(x) + self.conv2(nn.ZeroPad2d((1,0,1,0))(x)) + self.conv3(x)) / 3.
        out = torch.exp(x / -9.)
        out = out.reshape(N, H*W, self.kernel_size*self.kernel_size)
        return out


class DCK(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(DCK, self).__init__()

        self.kernel_size = kernel_size

        self.conv1x1_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

        self.pool = nn.AdaptiveMaxPool2d(3)

        self.weights0 = nn.Parameter(nn.init.normal_(torch.randn(out_ch, 1, 3, 3, requires_grad=True)))
        self.weights1 = nn.Parameter(nn.init.normal_(torch.randn(out_ch, 1, 3, 3, requires_grad=True)))
        self.weights2 = nn.Parameter(nn.init.normal_(torch.randn(out_ch, 1, 3, 3, requires_grad=True)))

        self.paired_conv = paired_conv(out_ch, kernel_size)
        self.norm0 = nn.BatchNorm2d(out_ch)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.conv1x1_4 = nn.Conv2d(out_ch * 4, out_ch, kernel_size=1, padding=0)


    def forward(self, x, coarse):
        N, C, H, W = x.shape
        _, C1, _, _ = coarse.shape

        out1 = self.conv1x1_1(x).reshape(N, C, -1)
        out2 = self.conv1x1_2(x).reshape(N, C, -1)
        out3 = self.conv1x1_3(x).reshape(N, C, -1)
        out4 = self.paired_conv(x)
        coarse = coarse.reshape(N, C1, -1)

        mul1 = torch.matmul(out1, coarse.permute(0, 2, 1))
        mul2 = torch.matmul(out2.permute(0, 2, 1), out3)
        mul2 = F.softmax(mul2.reshape(N, -1), -1).reshape(N, H * W, H * W)
        mul3 = torch.matmul(mul2, out4)
        mul4 = torch.matmul(coarse, mul3)
        mul5 = torch.matmul(mul1, mul4)
        mul5 = mul5.reshape(N, C, self.kernel_size, self.kernel_size)
        mul5 = self.pool(mul5).permute(1, 0, 2, 3)

        output0 = []
        for i in range(N):
            temp = F.conv2d(x[i, :, :, :].unsqueeze(0), mul5[:, i, :, :].unsqueeze(1)*self.weights0, dilation=1,
                            padding=(mul5.shape[-1] - 1) // 2,
                            groups=mul5.shape[0], stride=1)
            output0.append(temp)
        output0 = torch.cat(output0, 0)
        output0 = self.norm0(output0)
        output0 = self.relu(output0)

        output1 = []
        for i in range(N):
            temp = F.conv2d(x[i, :, :, :].unsqueeze(0), mul5[:, i, :, :].unsqueeze(1)*self.weights1, dilation=2,
                            padding=2 * (mul5.shape[-1] - 1) // 2,
                            groups=mul5.shape[0], stride=1)
            output1.append(temp)
        output1 = torch.cat(output1, 0)
        output1 = self.norm1(output1)
        output1 = self.relu(output1)

        output2 = []
        for i in range(N):
            temp = F.conv2d(x[i, :, :, :].unsqueeze(0), mul5[:, i, :, :].unsqueeze(1)*self.weights2, dilation=3,
                            padding=3 * (mul5.shape[-1] - 1) // 2,
                            groups=mul5.shape[0], stride=1)
            output2.append(temp)
        output2 = torch.cat(output2, 0)
        output2 = self.norm2(output2)
        output2 = self.relu(output2)

        output = torch.cat([x, output0, output1, output2], 1)
        output = self.conv1x1_4(output)

        return output


class context(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(context, self).__init__()

        self.dck1 = DCK(in_ch, out_ch, kernel_size=5)
        self.dck2 = DCK(in_ch, out_ch, kernel_size=7)
        self.dck3 = DCK(in_ch, out_ch, kernel_size=9)
        self.conv = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch*6, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_ch*6),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(out_ch * 6, out_ch * 3, kernel_size=1, padding=0)
                                  )


    def forward(self, x, label):

        out1 = self.dck1(x, label)
        out2 = self.dck2(x, label)
        out3 = self.dck3(x, label)
        output = self.conv(torch.cat([out1, out2, out3], 1))

        return output

