import torch
import torchvision
from torch import nn
import fastai
from fastai.vision import *
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out


# def conv(ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
# def conv2(ni,nf): return conv_layer(ni,nf,stride=1)
# def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


class myDnCNN(nn.Module):
    def __init__(self, depth=8, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(myDnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        
        layers.append(self.conv(ni=image_channels,nf=n_channels))
        layers.append(nn.ReLU())

        for _ in range(depth-2):
            layers.append(self.conv_and_res(ni=n_channels,nf=n_channels))
            
        layers.append(self.conv(ni=n_channels,nf=image_channels))
        self.mydncnn = nn.Sequential(*layers)
        

    def forward(self, x):
        y = x
        out = self.mydncnn(x)
        return y-out

    
    
    def conv(self,ni,nf): return nn.Conv2d(ni, nf, kernel_size=3, stride=1, padding=1)
    
    def conv2(self,ni,nf): return conv_layer(ni,nf,stride=1)
    
    def conv_and_res(self,ni,nf): return nn.Sequential(self.conv2(ni, nf), res_block(nf))