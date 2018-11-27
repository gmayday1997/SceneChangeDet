import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class vgg1024(nn.Module):

    def __init__(self):

        super(vgg1024, self).__init__()

        self.features = self._make_layers(cfg['D'])
        self.classifier = nn.Sequential(
            nn.Conv2d(512,1024,3),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(1024,1024,1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.AvgPool2d(kernel_size=3,stride=3),
            nn.Conv2d(1024, 21,1),
        )

    def forward(self,input):

        x =self.features(input)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg, batch_norm=False):

        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M' or v == 'A':
                if v == 'M':
                   layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers += [nn.AvgPool2d(kernel_size=3,stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def init_parameters(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16_1024.features.children())

        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        ####### init fc parameters (transplant) ##############

        self.fc6[0].weight.data = pretrain_vgg16.classifier[0].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16.classifier[0].bias.data.view(self.fc6[0].bias.size())
        self.fc7[0].weight.data = pretrain_vgg16.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16.classifier[3].bias.data.view(self.fc7[0].bias.size())

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M','A'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg_1024():

    model = VGG(make_layers(cfg['D']), **kwargs)
    return model
