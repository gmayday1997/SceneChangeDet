import torch
import torch.nn as nn
import torch.nn.functional as F

def convert_dict_names_for_fucking_faults():

   deeplab_v2_dict_names_mapping = {

      'conv1.0' : 'conv1_1',
      'conv1.2' : 'conv1_2',
      'conv2.0' : 'conv2_1',
      'conv2.2' : 'conv2_2',
      'conv3.0' : 'conv3_1',
      'conv3.2' : 'conv3_2',
      'conv3.4' : 'conv3_3',
      'conv4.0' : 'conv4_1',
      'conv4.2' : 'conv4_2',
      'conv4.4' : 'conv4_3',
      'conv5.0' : 'conv5_1',
      'conv5.2' : 'conv5_2',
      'conv5.4' : 'conv5_3'}

   return deeplab_v2_dict_names_mapping

class deeplab_V2_partial(nn.Module):
    def __init__(self):
        super(deeplab_V2_partial, self).__init__()

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
        )

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class deeplab_v2_late_fusion(nn.Module):

    def __init__(self, class_number=2):
        super(deeplab_v2_late_fusion, self).__init__()

        self.CNN = deeplab_V2_partial()
        self.fc8 = nn.Conv2d(in_channels=512*2, out_channels=class_number, kernel_size=1)

    def forward(self, t0, t1):
        feat_t0 = self.CNN(t0)
        feat_t1 = self.CNN(t1)
        fc = torch.cat((feat_t0, feat_t1), dim=1)
        out = self.fc8(fc)
        score_final = F.upsample(out, t0.size()[2:], mode='bilinear')

        return score_final

    def init_parameters_from_deeplab(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        pretrain_dict_names = convert_dict_names_for_fucking_faults()
        keys = sorted(pretrain_dict_names.keys())
        conv_blocks = [self.CNN.conv1,
                       self.CNN.conv2,
                       self.CNN.conv3,
                       self.CNN.conv4,
                       self.CNN.conv5]
        ranges = [[0,2], [0,2], [0,2,4], [0,2,4], [0,2,4]]
        for key in keys:
            dic_name = pretrain_dict_names[key]
            base_conv_name,conv_index,sub_index = dic_name[:5],int(dic_name[4]),int(dic_name[-1])
            conv_blocks[conv_index -1][ranges[sub_index -1][sub_index -1]].weight.data = pretrain_vgg16_1024[key + '.weight']
            conv_blocks[conv_index- 1][ranges[sub_index -1][sub_index -1]].bias.data = pretrain_vgg16_1024[key + '.bias']

        '''''''''
        ####### init fc parameters (transplant) ##############
        self.fc6_1[0].weight.data = pretrain_vgg16_1024['fc6_1.0.weight'].view(self.fc6_1[0].weight.size())
        self.fc6_1[0].bias.data = pretrain_vgg16_1024['fc6_1.0.bias'].view(self.fc6_1[0].bias.size())

        self.fc7_1[0].weight.data = pretrain_vgg16_1024['fc7_1.0.weight'].view(self.fc7_1[0].weight.size())
        self.fc7_1[0].bias.data = pretrain_vgg16_1024['fc7_1.0.bias'].view(self.fc7_1[0].bias.size())

        self.fc6_2[0].weight.data = pretrain_vgg16_1024['fc6_2.0.weight'].view(self.fc6_2[0].weight.size())
        self.fc6_2[0].bias.data = pretrain_vgg16_1024['fc6_2.0.bias'].view(self.fc6_2[0].bias.size())

        self.fc7_2[0].weight.data = pretrain_vgg16_1024['fc7_2.0.weight'].view(self.fc7_2[0].weight.size())
        self.fc7_2[0].bias.data = pretrain_vgg16_1024['fc7_2.0.bias'].view(self.fc7_2[0].bias.size())

        self.fc6_3[0].weight.data = pretrain_vgg16_1024['fc6_3.0.weight'].view(self.fc6_3[0].weight.size())
        self.fc6_3[0].bias.data = pretrain_vgg16_1024['fc6_3.0.bias'].view(self.fc6_3[0].bias.size())

        self.fc7_3[0].weight.data = pretrain_vgg16_1024['fc7_3.0.weight'].view(self.fc7_3[0].weight.size())
        self.fc7_3[0].bias.data = pretrain_vgg16_1024['fc7_3.0.bias'].view(self.fc7_3[0].bias.size())

        self.fc6_4[0].weight.data = pretrain_vgg16_1024['fc6_4.0.weight'].view(self.fc6_4[0].weight.size())
        self.fc6_4[0].bias.data = pretrain_vgg16_1024['fc6_4.0.bias'].view(self.fc6_4[0].bias.size())

        self.fc7_4[0].weight.data = pretrain_vgg16_1024['fc7_4.0.weight'].view(self.fc7_4[0].weight.size())
        self.fc7_4[0].bias.data = pretrain_vgg16_1024['fc7_4.0.bias'].view(self.fc7_4[0].bias.size())
        
        self.fc8_voc12_1[0].weight.data = pretrain_vgg16_1024['fc8_voc12_1.0.weight'].view(self.fc8_voc12_1[0].weight.size())
        self.fc8_voc12_1[0].bias.data = pretrain_vgg16_1024['fc8_voc12_1.0.bias'].view(self.fc8_voc12_1[0].bias.size())

        self.fc8_voc12_2[0].weight.data = pretrain_vgg16_1024['fc8_voc12_2.0.weight'].view(self.fc8_voc12_2[0].weight.size())
        self.fc8_voc12_2[0].bias.data = pretrain_vgg16_1024['fc8_voc12_2.0.bias'].view(self.fc8_voc12_2[0].bias.size())

        self.fc8_voc12_3[0].weight.data = pretrain_vgg16_1024['fc8_voc12_3.0.weight'].view(self.fc8_voc12_3[0].weight.size())
        self.fc8_voc12_3[0].bias.data = pretrain_vgg16_1024['fc8_voc12_3.0.bias'].view(self.fc8_voc12_3[0].bias.size())

        self.fc8_voc12_4[0].weight.data = pretrain_vgg16_1024['fc8_voc12_4.0.weight'].view(self.fc8_voc12_4[0].weight.size())
        self.fc8_voc12_4[0].bias.data = pretrain_vgg16_1024['fc8_voc12_4.0.bias'].view(self.fc8_voc12_4[0].bias.size())
        '''''''''



