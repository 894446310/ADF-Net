import torch.nn as nn
import torch

class conv_block(nn.Module):                                        #卷积块
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):                                           #上采样
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)  # 第一次卷积通道数为64

        # self.root = root_block(ch_in=64, ch_out=64)
        # self.Conv6 = conv_block(ch_in=64, ch_out=64)

        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)   #self.Up1 = up_conv(ch_in=768, ch_out=384)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)  #self.Up_conv1 = conv_block(ch_in=768, ch_out=384)
        self.Up4 = up_conv(ch_in=512, ch_out=256)    #self.Up2=up_conv(ch_in=384, ch_out=192)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)   #self.Up_conv2 = conv_block(ch_in=384, ch_out=192)
        self.Up3 = up_conv(ch_in=256, ch_out=128)    #self.Up3=up_conv(ch_in=192, ch_out=96)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)   #self.Up_conv3 = conv_block(ch_in=192, ch_out=96)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)  # 第一次两层卷积通道数64（1➡64）

        '''x = self.Conv1(x)
        x1 = self.Conv6(x)
        x1 += x'''
        e1=x1
        x2 = self.Maxpool(x1)  # 第一次下采样
        e2=x2
        x2 = self.Conv2(x2)  # 第二次
        e3=x2

        x3 = self.Maxpool(x2)  # 第二次下采样
        e4=x3
        x3 = self.Conv3(x3)
        e5=x3

        x4 = self.Maxpool(x3)  # 第三次下采样
        e6=x4
        x4 = self.Conv4(x4)
        e7=x4

        x5 = self.Maxpool(x4)  # 第四次下采样
        e8=x5
        x5 = self.Conv5(x5)  # 输入512通道数，输出1024（512➡1024）
        e9=x5

        # decoding + concat path
        d5 = self.Up5(x5)
        e10=d5
        d5 = torch.cat((x4, d5), dim=1)
        e11=d5

        d5 = self.Up_conv5(d5)
        e12=d5

        d4 = self.Up4(d5)
        e13=d4
        d4 = torch.cat((x3, d4), dim=1)
        e14=d4
        d4 = self.Up_conv4(d4)
        e15=d4

        d3 = self.Up3(d4)
        e16=d3
        d3 = torch.cat((x2, d3), dim=1)
        e17 = d3
        d3 = self.Up_conv3(d3)
        e18 = d3

        d2 = self.Up2(d3)
        e19 = d2
        d2 = torch.cat((x1, d2), dim=1)
        e20 = d2
        d2 = self.Up_conv2(d2)
        e21 = d2

        d1 = self.Conv_1x1(d2)
        e22 = d1

        return d1

