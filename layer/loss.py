import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class ConstractiveThresholdHingeLoss(nn.Module):

    def __init__(self,hingethresh=0.0,margin=2.0):
        super(ConstractiveThresholdHingeLoss, self).__init__()
        self.threshold = hingethresh
        self.margin = margin

    def forward(self,out_vec_t0,out_vec_t1,label):

        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        similar_pair = torch.clamp(distance - self.threshold,min=0.0)
        dissimilar_pair = torch.clamp(self.margin- distance,min=0.0)
        #dissimilar_pair = torch.clamp(self.margin-(distance-self.threshold),min=0.0)
        constractive_thresh_loss = torch.sum(
            (1-label)* torch.pow(similar_pair,2) + label * torch.pow(dissimilar_pair,2)
        )
        return constractive_thresh_loss

class ConstractiveLoss(nn.Module):

    def __init__(self,margin =2.0):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin

    def forward(self,out_vec_t0,out_vec_t1,label):

        distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        #distance = 1 - F.cosine_similarity(out_vec_t0,out_vec_t1)
        constractive_loss = torch.sum((1-label)*torch.pow(distance,2 ) + \
                                       label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2))
        return constractive_loss

class ConstractiveMaskLoss(nn.Module):

    def __init__(self,thresh_flag=False):
        super(ConstractiveMaskLoss, self).__init__()

        if thresh_flag:
            self.sample_constractive_loss = ConstractiveThresholdHingeLoss(margin=2.0,hingethresh=0.0)
        else:
            self.sample_constractive_loss = ConstractiveLoss(margin=2.0)

    def forward(self,out_t0,out_t1,ground_truth):

        #out_t0 = out_t0.permute(0,2,3,1)
        n,c,h,w = out_t0.data.shape
        out_t0_rz = torch.transpose(out_t0.view(c,h*w),1,0)
        out_t1_rz = torch.transpose(out_t1.view(c,h*w),1,0)
        gt_tensor = torch.from_numpy(np.array(ground_truth.data.cpu().numpy(),np.float32))
        gt_rz = Variable(torch.transpose(gt_tensor.view(1, h * w), 1, 0)).cuda()
        #gt_rz = Variable(torch.transpose(ground_truth.view(1,h*w),1,0))
        loss = self.sample_constractive_loss(out_t0_rz,out_t1_rz,gt_rz)
        return loss




