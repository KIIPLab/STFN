import torch
import torch.nn as nn
from typing import Optional
from einops import repeat


# patchmerging from timm swin-transformer
class patchSplitting(nn.Module):
    def __init__(self, dim: Optional[int]):
        super().__init__()
        self.dim = dim
        self.out_dim = dim // 2
        self.norm = nn.LayerNorm(dim//4)
        self.reduction = nn.Sequential(nn.Linear(dim//4, self.out_dim, bias=False))


    def forward(self, x):

        B, C, H, W= x.shape
        # x는 swin Transformer가 연산한 결과를 의미하며, shape을 보면 알 수 있듯, patchsplitting에 들어가기 전에 미리 reshape으로 H와 W를 구분했음을 알 수 있다.
        # 그래서 아래 차원의 텐서를 받아서 계산할 수 있다.
        # 4 512 28 28
        # 4 256 56 56
        #x = x.reshape(B, H * 2, 1, W * 2, 1, C//4 ).permute(0,1,3,4,2,5).flatten(3) # 이것도 충분히 성능이 나오지만 본 코드보다는 안정성이나 성능이나 조금 밀림.
        x = repeat(x, 'B (C C1 C2) H W  ->B (H C1) (W C2) C',C1=2 , C2=2) #einops의 repeat을 사용해 (B,C,H,W)중 C를 쪼개서 퍼트린다.

        # layer normalization으로 줄어든 채널수를 반영한 계산을 수행한다.
        #print("repeat:",x.shape)
        x = self.norm(x)
        #print("norm:",x.shape)

        x = self.reduction(x) # linear layer 수행
        #print("reduction:",x.shape)

        B,H,W,C = x.shape
        x = x.view(B,C,H,W)

        return x

class Mediator(nn.Module):
    def __init__(self, in_dim: Optional[int]): #256 -> 1024
        # 1024 hidden + 2 layer 가 best
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.d_hidn = 2 * in_dim
        self.mediator = nn.Sequential(
            nn.Conv2d(in_channels = self.in_dim, out_channels = self.d_hidn, kernel_size=1),
            nn.Mish(),
            nn.Conv2d(in_channels = self.d_hidn, out_channels = self.out_dim, kernel_size=1),
            nn.Mish()
        )



    def forward(self, second, first):
        #print("shallow:",shallow.shape)
        #print("deep:",deep.shape)
        differ = first - second # shallow한 쪽에서 - deep쪽을 빼준다.
        differ = self.mediator(differ)

        return differ #shallow



class Pixel_Prediction(nn.Module):
    def __init__(self, inchannels= 512 , outchannels=256, d_hidn=1024): # 1024 / 1536
        super().__init__()
        self.d_hidn = d_hidn
        self.down_channel = nn.Conv2d(inchannels, outchannels, kernel_size = 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=256*3, out_channels=self.d_hidn, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv2d(in_channels=self.d_hidn, out_channels=512, kernel_size=3, padding=1),
            nn.Mish()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.Mish()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
        )

        self.conv_attent = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, first_dis_, first_ref_, first_dis, first_ref):


        f_dis = torch.cat((first_dis_, first_dis), 1)
        f_ref = torch.cat((first_ref_, first_ref), 1)

        f_dis = self.down_channel(f_dis) # 채널 수 : 256
        f_ref = self.down_channel(f_ref)

        f_cat = torch.cat((f_dis - f_ref, f_dis, f_ref), 1) # 채널 수 : 256*3

        #print("f_cat:",f_cat.shape)

        feat_fused = self.conv1(f_cat) # output : 512 channel
        #print("feat_fused:", feat_fused.shape)
        feat = self.conv2(feat_fused) # output : 256 channel
        #print("feat:", feat.shape)

        q = self.conv(feat) # 1 channel
        #print("q:", q.shape)
        k = self.conv_attent(feat)
        #print("k:", k.shape)

        pred = (q * k).sum(dim=2).sum(dim=2) / k.sum(dim=2).sum(dim=2)  # 8,1 batch size만큼 조정.

        return pred