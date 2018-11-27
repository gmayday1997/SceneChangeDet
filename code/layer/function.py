import torch
import torch.nn as nn

#### source code from
####  https://github.com/ignacio-rocco/cnngeometric_pytorch
####
class FeatureCorrelation(nn.Module):
    def __init__(self,scale):
        super(FeatureCorrelation, self).__init__()
        self.scale = scale

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = self.scale * feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor

class l2normalization(nn.Module):
    def __init__(self,scale):

        super(l2normalization, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        #f = x.data.cpu().numpy()
        #scal = self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)
        #sca = scal.data.cpu().numpy()
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)

class l1normalization(nn.Module):
    def __init__(self,scale):
        super(l1normalization, self).__init__()
        self.scale = scale

    def forward(self,x,dim=1):
        # out = scale * x / sum(abs(x))
        return self.scale * x * x.pow(1).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)

class scale_feature(nn.Module):
    def __init__(self,scale):
        super(scale_feature, self).__init__()
        self.scale = scale

    def forward(self,x):
        return self.scale * x

class Mahalanobis_Distance(nn.Module):
    def __init__(self):
        super(Mahalanobis_Distance, self).__init__()

    def cal_con(self):
        pass

    def cal_invert_matrix(self):
        pass

    def forward(self,x1,x2):
        dis_abs = x1 - x2





