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

class deeplab_V2(nn.Module):
    def __init__(self,class_number=21):
        super(deeplab_V2, self).__init__()

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
        ####### multi-scale contexts #######
        ####### dialtion = 6 ##########
        self.fc6_1 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=6,padding=6),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc8_voc12_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=class_number, kernel_size=1)
        )
        ####### dialtion = 12 ##########
        self.fc6_2 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,dilation=12,padding=12),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc8_voc12_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=class_number, kernel_size=1)
        )
        ####### dialtion = 18 ##########
        self.fc6_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=18, padding=18),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc8_voc12_3 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=class_number, kernel_size=1)
        )
        ####### dialtion = 24 ##########
        self.fc6_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, dilation=24, padding=24),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc7_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )
        self.fc8_voc12_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=class_number, kernel_size=1)
        )

    def forward(self,x):

        size = x.size()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        fc6_1 = self.fc6_1(x)
        fc7_1 = self.fc7_1(fc6_1)
        fc8_1 = self.fc8_voc12_1(fc7_1)
        fc6_2 = self.fc6_2(x)
        fc7_2 = self.fc7_2(fc6_2)
        fc8_2 = self.fc8_voc12_2(fc7_2)
        fc6_3 = self.fc6_3(x)
        fc7_3 = self.fc7_3(fc6_3)
        fc8_3 = self.fc8_voc12_3(fc7_3)
        fc6_4 = self.fc6_4(x)
        fc7_4 = self.fc7_4(fc6_4)
        fc8_4 = self.fc8_voc12_4(fc7_4)
        score_final = fc8_1 + fc8_2 + fc8_3 + fc8_4
        #score_final_up = F.upsample_bilinear(score_final,size[2:])
        return score_final

    def init_parameters_from_deeplab(self,pretrain_vgg16_1024):

        ##### init parameter using pretrain vgg16 model ###########
        pretrain_dict_names = convert_dict_names_for_fucking_faults()
        keys = sorted(pretrain_dict_names.keys())
        conv_blocks = [self.conv1,
                       self.conv2,
                       self.conv3,
                       self.conv4,
                       self.conv5]
        ranges = [[0,2], [0,2], [0,2,4], [0,2,4], [0,2,4]]
        for key in keys:
            dic_name = pretrain_dict_names[key]
            base_conv_name,conv_index,sub_index = dic_name[:5],int(dic_name[4]),int(dic_name[-1])
            conv_blocks[conv_index -1][ranges[sub_index -1][sub_index -1]].weight.data = pretrain_vgg16_1024[key + '.weight']
            conv_blocks[conv_index- 1][ranges[sub_index -1][sub_index -1]].bias.data = pretrain_vgg16_1024[key + '.bias']

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



