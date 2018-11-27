import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

class deeplab(nn.Module):

    def __init__(self):
        super(deeplab, self).__init__()

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
        self.fc6 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

    def forward(self,input):

        x= self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc6(x)
        fc7_features = self.fc7(x)
        #out = self.fc8(x)
        return fc7_features

class Deeplab_MS_Att_Scale(nn.Module):

    def __init__(self,class_number =21):
        super(Deeplab_MS_Att_Scale, self).__init__()

        self.truck_branch = deeplab()
        self.scale_attention_branch = nn.Sequential(

            nn.Conv2d(in_channels=1024*3,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(in_channels=512,out_channels=3,kernel_size=1)
        )
        self.fc8 = nn.Conv2d(in_channels=1024,out_channels=class_number,kernel_size=1)

    def forward(self,x):

        input_size = x.size()[2]
        self.interp1 = nn.Upsample(size=(int(input_size * 0.75) + 1, int(input_size * 0.75) + 1),mode='bilinear')
        self.interp2 = nn.Upsample(size=(int(input_size * 0.5) + 1, int(input_size * 0.5) + 1),mode='bilinear')
        self.interp3 = nn.Upsample(size=(outS(input_size), outS(input_size)),mode='bilinear')
        out = []
        x75 = self.interp1(x)
        x50 = self.interp2(x)

        fc7_x   = self.truck_branch(x)
        fc7_x75 = self.truck_branch(x75)
        fc7_x50 = self.truck_branch(x50)

        out.append(fc7_x)
        out.append(self.interp3(fc7_x75))
        out.append(self.interp3(fc7_x50))
        out_cat = torch.cat(out,dim=1)
        #out_cat = torch.stack(out,dim=1)
        #print out_cat.size()
        scale_att_mask = F.softmax(self.scale_attention_branch(out_cat))

        score_x = self.fc8(fc7_x)
        score_x50 = self.interp3(self.fc8(fc7_x50))
        score_x75 = self.interp3(self.fc8(fc7_x75))
        assert score_x.size() == score_x50.size()

        score_att_x = torch.mul(score_x,scale_att_mask[:,0,:,:].expand_as(score_x))
        score_att_x_075 = torch.mul(score_x75,scale_att_mask[:,1,:,:].expand_as(score_x75))
        score_att_x_050 = torch.mul(score_x50,scale_att_mask[:,2,:,:].expand_as(score_x50))

        score_final = score_att_x + score_att_x_075 + score_att_x_050
        #out_final = F.upsample_bilinear(score_final, x.size()[2:])
        return score_final,scale_att_mask

    def init_parameters_from_deeplab(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        conv_blocks = [self.truck_branch.conv1,
                       self.truck_branch.conv2,
                       self.truck_branch.conv3,
                       self.truck_branch.conv4,
                       self.truck_branch.conv5]

        pretrain_conv_blocks = [pretrain_vgg16_1024.truck_branch.conv1,
                                pretrain_vgg16_1024.truck_branch.conv2,
                                pretrain_vgg16_1024.truck_branch.conv3,
                                pretrain_vgg16_1024.truck_branch.conv4,
                                pretrain_vgg16_1024.truck_branch.conv5]

        for idx, (conv_block,pretrain_conv_block) in enumerate(zip(conv_blocks,pretrain_conv_blocks)):
            for l1, l2 in zip(pretrain_conv_block, conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    # print idx, l1, l2
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

        ####### init fc parameters (transplant) ##############
        self.truck_branch.fc6[0].weight.data = pretrain_vgg16_1024.truck_branch.fc6[0].weight.data.view(self.truck_branch.fc6[0].weight.size())
        self.truck_branch.fc6[0].bias.data = pretrain_vgg16_1024.truck_branch.fc6[0].bias.data.view(self.truck_branch.fc6[0].bias.size())
        self.truck_branch.fc7[0].weight.data = pretrain_vgg16_1024.truck_branch.fc7[0].weight.data.view(self.truck_branch.fc7[0].weight.size())
        self.truck_branch.fc7[0].bias.data = pretrain_vgg16_1024.truck_branch.fc7[0].bias.data.view(self.truck_branch.fc7[0].bias.size())
        self.scale_attention_branch[0].weight.data = pretrain_vgg16_1024.scale_attention_branch[0].weight.data
        self.scale_attention_branch[0].bias.data = pretrain_vgg16_1024.scale_attention_branch[0].bias.data
        self.scale_attention_branch[3].weight.data = pretrain_vgg16_1024.scale_attention_branch[3].weight.data
        self.scale_attention_branch[3].bias.data = pretrain_vgg16_1024.scale_attention_branch[3].bias.data
        self.fc8.weight.data.normal_(0, 0.01)
        self.fc8.bias.data.fill_(0)
