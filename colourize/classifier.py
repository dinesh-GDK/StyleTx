import torch.nn as nn

class DoubleConvLeaky(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.leakyrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        
        return x

class PatchClassifier(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = DoubleConvLeaky(in_channels, 16)
        self.conv2 = DoubleConvLeaky(16, 32)
        self.conv3 = nn.Conv2d(32, out_channels, kernel_size=4, stride=1, padding=1)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x
