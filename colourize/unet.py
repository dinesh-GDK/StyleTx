import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        
        return x

class DownUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.doubleconv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.doubleconv(self.maxpool(x))

class UpUnit(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x, y):
        x = self.up(x)
        x = torch.cat([x, y], 1)  # the dim might create some issues later
        x = self.doubleconv(x)
        
        return x

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.input = DoubleConv(in_channels, 16)
        self.down1 = DownUnit(16, 32)
        self.down2 = DownUnit(32, 64)
        self.down3 = DownUnit(64, 128)
        self.down4 = DownUnit(128, 256)
        
        self.up1 = UpUnit(256, 128)
        self.up2 = UpUnit(128, 64)
        self.up3 = UpUnit(64, 32)
        self.up4 = UpUnit(32, 16)
        self.out = OutConv(16, out_channels)
    
    def forward(self, x):
        
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        
        return self.out(x9)
