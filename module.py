# -*- coding: utf-8 -*-
# @Time    : 21-11-24 上午12:10
# @Author  : lijixing
# @FileName: resunet.py
# @Software: PyCharm
# @Github £º


from torchvision import models
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class Regularization(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)  # 获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)
        print("---------------------------------------------------")


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

class Resunet_1(nn.Module):
    def __init__(self, num_classes=4, num_channels=1):
        super(Resunet_1, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = torch.nn.BatchNorm2d(64)
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

        initialize_weights(self.decoder4, self.decoder3, self.decoder2, self.decoder1,self.bpb1,self.bpb2,
                           self.bpb3,self.bpb4,self.bpb5,self.bpb6,self.bpb7,self.bpb8,
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

        self.dck1 = DCK(in_ch, out_ch, kernel_size=7)
        self.dck2 = DCK(in_ch, out_ch, kernel_size=9)
        self.dck3 = DCK(in_ch, out_ch, kernel_size=11)

    def forward(self, x, label):

        out1 = self.dck1(x, label)
        out2 = self.dck2(x, label)
        out3 = self.dck3(x, label)

        return torch.cat([out1, out2, out3], 1)
        # return out2

import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class downsample(nn.Module):

    def __init__(self, inchannels=4):
        self.inplanes = 16
        super(downsample, self).__init__()

        self.conv1 = nn.Conv2d(inchannels, 16, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 16, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, 24, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 16, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, inchannels, 2, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x





class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, set_board=False):
        super(down, self).__init__()
        self.set_board = set_board
        self.bpb = BPB(in_ch)
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        if self.set_board:
            board_map = self.bpb(x)
            x = (1 + board_map) * x
            x = self.max_pool_conv(x)
            return board_map, x
        else:
            x = self.max_pool_conv(x)
            return None, x



class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, set_board=False):
        super(up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.set_board = set_board
        self.conv = double_conv(in_ch, out_ch)
        self.bpb = BPB(out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2), diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        if self.set_board:
            board_map = self.bpb(x)
            x = (1 + board_map) * x
            return board_map, x
        else:
            return None, x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=4, board_layer=[True, True, True, True, True, True, True, True]):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128, set_board=board_layer[0])
        self.down2 = down(128, 256, set_board=board_layer[1])
        self.down3 = down(256, 512, set_board=board_layer[2])
        self.down4 = down(512, 512, set_board=board_layer[3])
        self.up1 = up(1024, 256, set_board=board_layer[4])
        self.up2 = up(512, 128, set_board=board_layer[5])
        self.up3 = up(256, 64, set_board=board_layer[6])
        self.up4 = up(128, 64, set_board=board_layer[7])
        self.outc = outconv(64, n_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2[-1])
        x4 = self.down3(x3[-1])
        x5 = self.down4(x4[-1])
        x6 = self.up1(x5[-1], x4[-1])
        x7 = self.up2(x6[-1], x3[-1])
        x8 = self.up3(x7[-1], x2[-1])
        x9 = self.up4(x8[-1], x1)
        x = self.outc(x9[-1])

        return x2[0], x3[0], x4[0], x5[0], x6[0], x7[0], x8[0], x9[0], x


class ultranet(nn.Module):
    def __init__(self, num_classes=4, num_channels=1):
        super(ultranet, self).__init__()

        self.onestage_net = Resunet_1(num_classes=num_classes, num_channels=num_channels)
        checkpoint = torch.load('./record_unet/val5_best_model_3_0.9244.pth')
        self.onestage_net.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

        # self.onestage_net.load_state_dict(torch.load('./record/best_model_0_0.6983.pth'))

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = torch.nn.Conv2d(num_classes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.downsample = downsample()
        self.context = context(512, 512)

        self.decoder4 = DecoderBlock(1536, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 1, padding=0)


    def forward(self, x):

        with torch.no_grad():
            e11, e12, e13, down1, down2, down3, down4, up1, up2, up3, up4, onestage_out = self.onestage_net(x)
            predict_mask = torch.sigmoid(onestage_out)

        x = self.firstconv(predict_mask * x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e4 = self.context(e4, self.downsample(predict_mask))

        # Decoder
        d4 = self.decoder4(e4) + e3 + e13
        d3 = self.decoder3(d4) + e2 + e12
        d2 = self.decoder2(d3) + e1 + e11
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out


