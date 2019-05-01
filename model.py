import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.conv2 = ConvLayer(32, 64, 9, 1)

        self.pool1 = ConvLayer(64, 64, 3, 2)
        # Residual layers
        self.res1 = ResidualBlock(64)
        self.res2 = ResidualBlock(64)
        self.res3 = ResidualBlock(64)
        self.res4 = ResidualBlock(64)

        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(64, 32, 3, 1, 1, 2)
        self.deconv2 = UpsampleConvLayer(32, 16, 3, 2, 0, 2)
        self.deconv3 = ConvLayer(16, 3, 9, 1)

        # Non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        # conv-block
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        f1 = x

        # res-block x 4
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        f2 = x

        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.deconv3(x)  
              
        return f1, f2, x


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        # if upsample:
        #     self.upsample_layer = torch.nn.UpsamplingNearest2d(scale_factor=upsample)
        self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='nearest')
        # one of `nearest`, `linear`, `bilinear` and `trilinear`
    
        # reflection_padding = int(np.floor(kernel_size / 2)) 
        reflection_padding = padding
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels,channels,3,1)
        self.conv2 = ConvLayer(channels,channels,3,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out