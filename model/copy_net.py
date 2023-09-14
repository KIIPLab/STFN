import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
import cv2
import matplotlib.pyplot as plt


"""class decoder(nn.Module):

    def __init__(self, opt, deep_channels=768 , shallow_channels=768 , out_channels=768):
        super().__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.d_hidn = 512
        if opt.patch_size == 8:
            stride = 1
        else:
            stride = 2
        #self.conv_offset = nn.Conv2d(deep_channels, 2 * 3 * 3, 3, 1, 1)
        #self.deform = DeformConv2d(shallow_channels, out_channels, 3, 1, 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=self.d_hidn, kernel_size=3, padding=1, stride=1),
            # stride 2 -->1
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=out_channels, kernel_size=3, padding=1, stride=stride)
        )

        # 여기선 stride에 따라 모델 실행 여부가 달라짐

    def forward(self, shallow_feat, depth_feat):

        # vit_feat = F.interpolate(vit_feat, size=cnn_feat.shape[-2:], mode="nearest")
        #offset = self.conv_offset(depth_feat)
        #deform_feat = self.deform(shallow_feat, offset)


        deform_feat = torch.cat((shallow_feat, depth_feat), dim= 1)
        #deform_feat = self.conv1(deform_feat)

        return deform_feat"""

class Mediator(nn.Module):
    def __init__(self, in_dim: 768 , out_dim : 768 , d_hidn = 1024 ):
        # 1024 hidden + 2 layer 가 best
        super().__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        self.mediator = nn.Sequential(
            nn.Conv2d(in_channels=self.in_dim, out_channels = d_hidn , kernel_size=1),
            nn.Mish(),
            nn.Conv2d(in_channels= d_hidn, out_channels= self.out_dim, kernel_size=1))


    def forward(self, shallow, deep):

        differ = shallow - deep
        differ = self.mediator(differ)
        deep = deep + differ

        return deep #shallow

class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels=768 * 3, outchannels=256, d_hidn=1024): # in_channels:768 *3 + 256*3 / 768*3
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size=1)
        self.feat_smoothing = nn.Sequential(
            nn.Conv2d(in_channels=256 * 3, out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

    def forward(self, d_dis, d_ref, s_dis, s_ref):
        #print('depth:',d_dis.shape)
        #print('shallow:',s_dis.shape)


        f_dis = torch.cat((d_dis, s_dis), 1)
        f_ref = torch.cat((d_ref, s_ref), 1)
        #print('b4:',f_dis.shape)
        f_dis = self.down_channel(f_dis)
        #print('af:',f_dis.shape)
        f_ref = self.down_channel(f_ref)
        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1)
        #print("final cat:",f_cat.shape)
        feat_fused = self.feat_smoothing(f_cat)
        feat = self.conv1(feat_fused)
        f = self.conv(feat)
        w = self.conv_attent(feat)

        pred = (f * w).sum(dim=2).sum(dim=2) / w.sum(dim=2).sum(dim=2)  # 10,1

        return pred