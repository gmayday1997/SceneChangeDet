import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import layer.function as fun
import torch.nn.init as init

class fcn32s(nn.Module):

    def __init__(self,distance_flag):

        super(fcn32s, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,dilation=2,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True)
        )
        self.embedding_layer = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, dilation=2, padding=2)
        if distance_flag == 'softmax':
           self.fc8 = nn.Softmax2d()
        if distance_flag == 'l2':
           self.fc8 = fun.l2normalization(scale=1)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.embedding_layer(x)
        embedding_feature = self.fc8(x)
        return embedding_feature

class SiameseNet(nn.Module):
    def __init__(self, distance_flag='softmax'):
        super(SiameseNet, self).__init__()
        self.CNN = fcn32s(distance_flag)

    def forward(self, t0, t1):
        out_t0_conv5 = self.CNN(t0)
        out_t1_conv5 = self.CNN(t1)
        return [out_t0_conv5, out_t1_conv5]

    def init_parameters(self,pretrain_vgg16):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.CNN.conv1,
                       self.CNN.conv2,
                       self.CNN.conv3,
                       self.CNN.conv4,
                       self.CNN.conv5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(pretrain_vgg16.features.children())

        for idx, conv_block in enumerate(conv_blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        init.kaiming_uniform(self.CNN.embedding_layer.weight.data, mode='fan_in')
        init.constant(self.CNN.embedding_layer.bias.data, 0)

        ####### init fc parameters (transplant) ##############
        ''''''''''
        self.fc6[0].weight.data = pretrain_vgg16.classifier[0].weight.data.view(self.fc6[0].weight.size())
        self.fc6[0].bias.data = pretrain_vgg16.classifier[0].bias.data.view(self.fc6[0].bias.size())
        self.fc7[0].weight.data = pretrain_vgg16.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16.classifier[3].bias.data.view(self.fc7[0].bias.size())

        ###### random init socore layer parameters ###########
        assert  self.upscore.kernel_size[0] == self.upscore.kernel_size[1]
        initial_weight = get_upsampling_weight(self.upscore.in_channels, self.upscore.out_channels, self.upscore.kernel_size[0])
        self.upscore.weight.data.copy_(initial_weight)
        '''''''''

# https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/surgery.py
def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()
