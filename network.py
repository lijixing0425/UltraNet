# -*- coding: utf-8 -*-
# @Time    : 21-6-10 下午1:19
# @Author  : lijixing&&gongyuxin
# @FileName: network.py
# @Software: PyCharm
# @Github £º https://github.com/lijixing0425/Ultranet

from resunet import *
from context import context
from coarse_net import downsample

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

class ultranet(nn.Module):
    def __init__(self, n_channels=1, n_classes=15):
        super(ultranet, self).__init__()

        self.onestage_net = res_unet(num_classes=n_classes, num_channels=n_channels)
        self.onestage_net.load_state_dict(torch.load('best_model_8_0.1058.pth'))

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)

        self.firstconv = torch.nn.Conv2d(n_classes, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu

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


        self.finalconv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, n_classes, 1, padding=0)

        initialize_weights(self.decoder4, self.decoder3, self.decoder2, self.decoder1,
                           self.finalconv2, self.finalconv3)


    def forward(self, x):

        with torch.no_grad():
            k2, k3, k4, down1, down2, down3, down4, up1, up2, up3, up4, onestage_out = self.onestage_net(x)
            predict_mask = torch.sigmoid(onestage_out)

        x = self.firstconv(predict_mask * x)
        x = self.firstbn(x)
        x = self.firstrelu(x)


        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.context(e4, self.downsample(predict_mask))

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finalconv2(d1)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out
