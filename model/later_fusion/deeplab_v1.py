import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import utils as utils

class deeplabV1_partial(nn.Module):

    def __init__(self):
        super(deeplabV1_partial, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,ceil_mode=True),
        )
        self.conv3 = nn.Sequential(

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1 ,ceil_mode=True),
        )
        self.conv4 = nn.Sequential(

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels= 512, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1,ceil_mode=True),
        )
        self.conv5 = nn.Sequential(

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3, dilation=2 ,stride=1,padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2 ,stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512,out_channels=512, kernel_size=3, dilation=2, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 ,stride=1, padding=1,ceil_mode=True),
            nn.AvgPool2d(kernel_size=3 , stride=1, padding=1,ceil_mode=True),
        )

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class deeplabv1_late_fusion(nn.Module):

    def __init__(self,class_number=21,transition=True):

        super(deeplabv1_late_fusion, self).__init__()

        self.transition = transition
        self.feature = deeplabV1_partial()
        if transition:

            self.transition_layer = nn.Sequential(
                nn.Conv2d(in_channels=1024,out_channels=512,kernel_size=3),
                nn.ReLU(inplace=True),
            )

            self.fc6 = nn.Sequential(
                nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )
            self.fc7 = nn.Sequential(
                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )

        else:
            self.fc6 = nn.Sequential(

                nn.Conv2d(in_channels=512*2, out_channels=1024, kernel_size=3, dilation=12, padding=12),
                nn.ReLU(inplace=True),
                nn.Dropout2d()
            )
            self.fc7 = nn.Sequential(

                nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )

        self.score = nn.Conv2d(in_channels=1024,out_channels=class_number,kernel_size=1)

    def forward(self,t0,t1):

        feat_t0 = self.feature(t0)
        feat_t1 = self.feature(t1)
        if self.transition:
            x = self.transition_layer(torch.cat((feat_t0,feat_t1),dim=1))
            x = self.fc6(x)
            x = self.fc7(x)
        else:
            x = self.fc6(torch.cat((feat_t0,feat_t1),dim=1))
            x = self.fc7(x)

        out = self.score(x)
        out = F.upsample(out, t0.size()[2:],mode='bilinear')
        return out

    def init_parameters(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.feature.conv1,
                       self.feature.conv2,
                       self.feature.conv3,
                       self.feature.conv4,
                       self.feature.conv5]

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

        if self.transition:
            self.transition_layer[0].weight.data.normal_(0, 0.01)
            self.transition_layer[0].bias.data.fill_(0)

            self.fc6[0].weight.data = pretrain_vgg16_1024.classifier[0].weight.data.view(self.fc6[0].weight.size())
            self.fc6[0].bias.data = pretrain_vgg16_1024.classifier[0].bias.data.view(self.fc6[0].bias.size())

        else:
            self.fc6[0].weight.data = torch.cat((pretrain_vgg16_1024.classifier[0].weight.data,
                                                 pretrain_vgg16_1024.classifier[0].weight.data), dim=1)

            self.fc6[0].bias.data = pretrain_vgg16_1024.classifier[0].bias.data.view(self.fc6[0].bias.size())

        self.fc7[0].weight.data = pretrain_vgg16_1024.classifier[3].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16_1024.classifier[3].bias.data.view(self.fc7[0].bias.size())

    def init_parameters_from_deeplab(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]

        pretrain_conv_blocks = [pretrain_vgg16_1024.conv1,
                                pretrain_vgg16_1024.conv2,
                                pretrain_vgg16_1024.conv3,
                                pretrain_vgg16_1024.conv4,
                                pretrain_vgg16_1024.conv5]

        for idx, (conv_block,pretrain_conv_block) in enumerate(zip(conv_blocks,pretrain_conv_blocks)):
            for l1, l2 in zip(pretrain_conv_block, conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        ####### init fc parameters (transplant) ##############
        if self.transition:

            self.transition.weight.normal_(0, 0.01)
            self.transition.bias.data.fill_(0)
            self.fc6[0].weight.data = torch.cat((pretrain_vgg16_1024.fc6[0].weight.data,
                                                 pretrain_vgg16_1024.fc6[0].weight.data),dim=1)
            self.fc6[0].bias.data = pretrain_vgg16_1024.fc6[0].bias.data.view(self.fc6[0].bias.size())

        else:
            self.fc6[0].weight.data = pretrain_vgg16_1024.fc6[0].weight.data.view(self.fc6[0].weight.size())
            self.fc6[0].bias.data = pretrain_vgg16_1024.fc6[0].bias.data.view(self.fc6[0].bias.size())

        self.fc7[0].weight.data = pretrain_vgg16_1024.fc7[0].weight.data.view(self.fc7[0].weight.size())
        self.fc7[0].bias.data = pretrain_vgg16_1024.fc7[0].bias.data.view(self.fc7[0].bias.size())

        self.score.weight.data = pretrain_vgg16_1024.score.weight.data.view(self.score.weight.size())
        self.score.bias.data = pretrain_vgg16_1024.score.bias.data.view(self.score.bias.size())


