# -*- coding: utf-8 -*-
# @Time    : 21-6-10 下午1:22
# @Author  : lijixing&&gongyuxin
# @FileName: resunet.py
# @Software: PyCharm
# @Github £º https://github.com/lijixing0425/ultranet


import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BPB(nn.Module):
    def __init__(self, in_ch):
        super(BPB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=1, padding=0, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=6, dilation=6),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(64 * 5, 1, kernel_size=1, padding=0, dilation=1),
            nn.Sigmoid())

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        board_map = self.conv6(x6)
        return board_map


class res_unet(nn.Module):
    def __init__(self, num_classes=15, num_channels=1):
        super(res_unet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = torch.nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.bpb1 = BPB(64)
        self.encoder1 = resnet.layer1
        self.bpb2 = BPB(64)
        self.encoder2 = resnet.layer2
        self.bpb3 = BPB(128)
        self.encoder3 = resnet.layer3
        self.bpb4 = BPB(256)
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.bpb5 = BPB(256)
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.bpb6 = BPB(128)
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.bpb7 = BPB(64)
        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.bpb8 = BPB(64)

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        initialize_weights(self.decoder4, self.decoder3, self.decoder2, self.decoder1,
                           self.finaldeconv1, self.finalconv2, self.finalconv3)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        bpb1 = self.bpb1(x)
        x = x + x * bpb1
        e1 = self.encoder1(x)
        bpb2 = self.bpb2(e1)
        e1 = e1 + e1 * bpb2
        e2 = self.encoder2(e1)
        bpb3 = self.bpb3(e2)
        e2 = e2 + e2 * bpb3
        e3 = self.encoder3(e2)
        bpb4 = self.bpb4(e3)
        e3 = e3 + e3 * bpb4
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        bpb5 = self.bpb5(d4)
        d4 = d4 + d4 * bpb5
        d3 = self.decoder3(d4) + e2
        bpb6 = self.bpb6(d3)
        d3 = d3 + d3 * bpb6
        d2 = self.decoder2(d3) + e1
        bpb7 = self.bpb7(d2)
        d2 = d2 + d2 * bpb7
        d1 = self.decoder1(d2)
        bpb8 = self.bpb8(d1)
        d1 = d1 + d1 * bpb8

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return e1, e2, e3, bpb1, bpb2, bpb3, bpb4, bpb5, bpb6, bpb7, bpb8, out
