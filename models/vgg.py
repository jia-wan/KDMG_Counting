import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torchvision import models
from torch.nn import functional as F

__all__ = ['vgg19']
model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class VGG(nn.Module):
    def __init__(self, features, down=8):
        super(VGG, self).__init__()
        self.down = down
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        if self.down < 16:
            x = F.interpolate(x, scale_factor=2)
        x = self.reg_layer(x)
        x = torch.abs(x)
        return x


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    # in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'C': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],    
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],    
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E'], batch_norm=False))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

class CSRNet(nn.Module):
    def __init__(self, down=8, bn=False):
        super(CSRNet, self).__init__()
        self.down = down
        self.features = make_layers(cfg['C'], batch_norm=bn)
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.reg_layer = make_layers(self.backend_feat, in_channels=512, batch_norm=bn, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()
        # self.features.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
        if bn:
            mod = models.vgg16_bn(pretrained = True)
        else:
            mod = models.vgg16(pretrained = True)
        fs = self.features.state_dict()
        ms = mod.state_dict()
        for key in fs:
            fs[key] = ms['features.'+key]
        self.features.load_state_dict(fs)
        # print('pretrained model loaded!')

    def forward(self, x):
        x = self.features(x)
        x = self.reg_layer(x)
        x = self.output_layer(x)
        x = torch.abs(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
