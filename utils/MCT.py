import torch
import torch.nn as nn
from utils.SE import SE_Block
import torch.nn.functional as F


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MetaFormer(nn.Module):

    def __init__(self, drop_path=0., num_skip=3, skip_dim=[128, 256, 512],
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ):
        super().__init__()

        fuse_dim = 0
        for i in range(num_skip):
            fuse_dim += skip_dim[i]


        self.fuse_conv2 = nn.Conv2d(fuse_dim, skip_dim[0], 1, 1)
        self.fuse_conv3 = nn.Conv2d(fuse_dim, skip_dim[1], 1, 1)
        self.fuse_conv4 = nn.Conv2d(fuse_dim, skip_dim[2], 1, 1)


        self.se2 = SE_Block(skip_dim[0])
        self.se3 = SE_Block(skip_dim[1])
        self.se4 = SE_Block(skip_dim[2])

        self.down_sample1 = nn.AvgPool2d(8)
        self.down_sample2 = nn.AvgPool2d(4)
        self.down_sample3 = nn.AvgPool2d(2)

        self.bn2 = nn.BatchNorm2d(skip_dim[0])
        self.bn3 = nn.BatchNorm2d(skip_dim[1])
        self.bn4 = nn.BatchNorm2d(skip_dim[2])

        self.up_sample1 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up_sample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_sample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        self.sa = SpatialGate()


    def forward(self, x1, x2, x3, x4):
        """
        x: B, H*W, C
        """
        org2 = x2
        org3 = x3
        org4 = x4

        x2 = self.bn2(x2)
        x3 = self.bn3(x3)
        x4 = self.bn4(x4)

        x2_d = self.down_sample2(x2)
        x3_d = self.down_sample3(x3)

        list1 = [x2_d, x3_d, x4]

        # --------------------Concat sum------------------------------
        fuse = torch.cat(list1, dim=1)

        x2 = self.fuse_conv2(fuse)
        x3 = self.fuse_conv3(fuse)
        x4 = self.fuse_conv4(fuse)

        x2_up = self.up_sample2(x2)
        x3_up = self.up_sample3(x3)


        x2 = org2 + self.se2(x2_up)
        x3 = org3 + self.se3(x3_up)
        x4 = org4 + self.se4(x4)

        x22 = x2 +self.sa(self.bn2(x2))
        x33 = x3 +self.sa(self.bn3(x3))
        x44 = x4 +self.sa(self.bn4(x4))

        return x1, x22, x33, x44

