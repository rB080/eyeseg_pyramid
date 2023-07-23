import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils import *


class PYNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(PYNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.convx = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self.act = nn.Sigmoid()

        self.inc = DoubleConv(3, 64)
        self.conv = DoubleConv(2048, 2048)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.down5 = Down(1024, 2048 // factor)

        self.up5 = Up(2048, 1024 // factor, bilinear)
        self.up4 = Up(1024, 512 // factor, bilinear)
        self.up3 = Up(512, 256 // factor, bilinear)
        self.up2 = Up(256, 128, bilinear)
        self.up1 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.pyup1 = Up(512, 256, bilinear)
        self.pyup2 = Up(256, 128, bilinear)
        self.pyup3 = Up(128, 64, bilinear)
        #self.pyup4 = Up(128, 64, bilinear)
        #self.py = nn.Conv2d(in_channels = 64 , out_channels = 32, kernel_size = 1)

        self.rfb_d1 = RFB_modified(64, 64)
        self.rfb_d2 = RFB_modified(128, 128)
        self.rfb_d3 = RFB_modified(256, 256)
        self.rfb_d4 = RFB_modified(512, 512)
        self.rfb_d5 = RFB_modified(1024, 1024)

        self.rfb_b = RFB_modified(512, 512)

        self.rfb_u5 = RFB_modified(1024, 1024)
        self.rfb_u4 = RFB_modified(512, 512)
        self.rfb_u3 = RFB_modified(256, 256)
        self.rfb_u2 = RFB_modified(128, 128)
        self.rfb_u1 = RFB_modified(64, 64)

        r = 16
        self.se_d1 = SE_Block(64, r)
        self.se_d2 = SE_Block(128, r)
        self.se_d3 = SE_Block(256, r)
        self.se_d4 = SE_Block(512, r)
        self.se_d5 = SE_Block(1024, r)
        self.se_u5 = SE_Block(2048, r)
        self.se_u4 = SE_Block(1024, r)
        self.se_u3 = SE_Block(512, r)
        self.se_u2 = SE_Block(256, r)
        self.se_u1 = SE_Block(128, r)

        self.puth1 = SpatialAttention(7, 64)
        self.puth2 = SpatialAttention(7, 128)
        self.puth3 = SpatialAttention(7, 256)
        self.puth4 = SpatialAttention(7, 512)
        self.puth5 = SpatialAttention(7, 1024)

    def forward(self, x):
        # down half
        #c = 3
        # print(x.shape)
        x1 = self.inc(x)
        x1 = self.rfb_d1(x1)
        x1_s = self.se_d1(x1)
        #c = 64
        x2 = self.down1(x1_s)
        x2 = self.rfb_d2(x2)
        x2_s = self.se_d2(x2)
        #c = 128
        x3 = self.down2(x2_s)
        x3 = self.rfb_d3(x3)
        x3_s = self.se_d3(x3)
        #c = 256
        #x4 = self.down3(x3)
        #x4 = self.rfb_d4(x4)
        #x4_s = self.se_d4(x4)
        #c = 512
        #x5 = self.down4(x4)
        #x5 = self.rfb_d5(x5)
        #x5_s = self.se_d5(x5)
        #c = 1024

        # base block
        base = self.down3(x3_s)
        base = self.rfb_b(base)
        b1 = base
        base = self.se_u3(base)
        #c = 2048

        # up-half_2048
        #x5_s = self.puth5(x5_s)
        #y5 = self.up5(base, x5_s)
        #y5 = self.rfb_u5(y5)
        #y5 = self.se_u4(y5)
        #c = 1024
        #x4_s = self.puth4(x4_s)
        #y4 = self.up4(base, x4_s)
        #y4 = self.rfb_u4(y4)
        #y4 = self.se_u3(y4)
        #c = 512
        x3_s = self.puth3(x3_s)
        y3 = self.up3(base, x3_s)
        y3 = self.rfb_u3(y3)
        d1 = y3
        y3 = self.se_u2(y3)
        #c = 256
        x2_s = self.puth2(x2_s)
        y2 = self.up2(y3, x2_s)
        y2 = self.rfb_u2(y2)
        d2 = y2
        y2 = self.se_u1(y2)
        #c = 128
        x1_s = self.puth1(x1_s)
        y1 = self.up1(y2, x1_s)
        y1 = self.rfb_u1(y1)
        d3 = y1
        # print(y1.shape)
        #c = 64

        # Pyramid Implementation should be something like this I think...pyup() basically stands for pyramid-up...maybe we should define that as well
        d1 = self.pyup1(b1, d1)
        d2 = self.pyup2(d1, d2)
        d3 = self.pyup3(d2, d3)
        p1 = d3  # self.py(d3)
        # d3 needs to be passed thorugh a 1x1 conv to get the final map...

        #p2 = self.convx(y1)
        logits = self.outc(p1)
        logits = self.act(logits)
        # print(logits.shape)
        return logits


def get_model(args):
    model = PYNet()
    model.to(args.device)
    return model
