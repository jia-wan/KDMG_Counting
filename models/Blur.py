import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F
from torchvision import models


class IndivBlur8(nn.Module):
    ## learn a kernel for each dot
    ## smaller image size and interpolation
    def __init__(self, downsample=8, s=15, softmax=False, small=False):
        super(IndivBlur8, self).__init__()
        self.downsample = downsample
        self.s = s
        self.softmax = softmax
        h = [32, 64, 128, 128]
        if small:
            h = [8, 16, 32, 32]
        
        self.adapt = nn.Sequential(
                                   nn.Conv2d(3, h[0], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2), 
                                   nn.Conv2d(h[0], h[1], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[1], h[2], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[2], h[3], 3, 1, 1),
                                   nn.LeakyReLU(0.01),
                                   nn.MaxPool2d(2),
                                   nn.Conv2d(h[3], self.s**2, 3, 1, 1))
        self._initialize_weights()

    def forward(self, points, img, shape):
        # generate kernels
        if img.shape[1] == 1:
            img = img.repeat(1,3,1,1)
        kernels = self.adapt(img)
        if self.softmax:
            kernels = F.softmax(kernels,1)
        else:
            kernels = kernels-torch.min(kernels,1,True)[0] #+ 1e-4
            kernels = kernels/torch.sum(kernels,1,True)# + 1e-12

        density = torch.zeros((shape)).cuda()
        # generate density for each image
        for j, idx in enumerate(points):
            n = len(idx) 
            if n == 0:
               continue 
            
            for i in range(n):
                y = max(0, int(idx[i,1]/self.downsample - (self.s+1)/2))
                x = max(0, int(idx[i,0]/self.downsample - (self.s+1)/2))
                ymax = min(y+self.s, density.shape[2])
                xmax = min(x+self.s, density.shape[3])
                # conv and sum
                k = kernels[0,:,min(kernels.shape[2]-1,int(idx[i,1]/16)),min(kernels.shape[3]-1,int(idx[i,0]/16))].view(1,1,self.s,self.s)
                if ymax-y < self.s or xmax-x < self.s:
                    xk, yk, xkmax, ykmax = 0, 0, self.s, self.s
                    if y == 0:
                        yk = self.s - (ymax-y)
                        ykmax = self.s
                    if x == 0:
                        xk = self.s - (xmax-x)
                        xkmax = self.s
                    if ymax == density.shape[2]:
                        ykmax = ymax - y
                        yk = 0
                    if xmax == density.shape[3]:
                        xkmax = xmax - x
                        xk = 0
                    k = k[:,:,yk:ykmax,xk:+xkmax]
                density[j,:,y:ymax,x:xmax] += k[0]

        return density

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

