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

    def __init__(self,margin =2.0,dist_flag='l2'):
        super(ConstractiveLoss, self).__init__()
        self.margin = margin
        self.dist_flag = dist_flag

    def various_distance(self,out_vec_t0,out_vec_t1):

        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'l1':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=1)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance

    def forward(self,out_vec_t0,out_vec_t1,label):

        #distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        distance = self.various_distance(out_vec_t0,out_vec_t1)
        #distance = 1 - F.cosine_similarity(out_vec_t0,out_vec_t1)
        constractive_loss = torch.sum((1-label)*torch.pow(distance,2 ) + \
                                       label * torch.pow(torch.clamp(self.margin - distance, min=0.0),2))
        return constractive_loss

class ConstractiveMaskLoss(nn.Module):

    def __init__(self,thresh_flag=False,hinge_thresh=0.0,dist_flag='l2'):
        super(ConstractiveMaskLoss, self).__init__()

        if thresh_flag:
            self.sample_constractive_loss = ConstractiveThresholdHingeLoss(margin=2.0,hingethresh=hinge_thresh)
        else:
            self.sample_constractive_loss = ConstractiveLoss(margin=2.0,dist_flag=dist_flag)

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

class LogDetDivergence(nn.Module):

    def __init__(self,model,param_name,dim=512):
        super(LogDetDivergence, self).__init__()
        self.param_name = param_name
        self.param_dict = dict(model.named_parameters())
        self.dim = dim
        self.identity_matrix = Variable(torch.from_numpy(np.identity(self.dim)).float()).cuda()

    def select_param(self):

        for layer_name, layer_param in self.param_dict.items():
            if self.param_name in layer_name:
                if 'weight' in layer_name:
                    return layer_param

    def forward(self):

        constrainted_matrix = self.select_param()
        matrix_ = torch.squeeze(torch.squeeze(constrainted_matrix,dim=2),dim=2)
        matrix_t = torch.t(matrix_)
        matrixs = torch.mm(matrix_t,matrix_)
        trace_ = torch.trace(torch.mm(matrixs,torch.inverse(matrixs)))
        log_det = torch.logdet(matrixs)
        maha_loss = trace_ - log_det
        return maha_loss

class Mahalanobis_Constraint(nn.Module):
    def __init__(self,model,param_name,dim=512):
        super(Mahalanobis_Constraint, self).__init__()
        self.param_name = param_name
        self.param_dict = dict(model.named_parameters())
        self.dim = dim
        self.identity_matrix = Variable(torch.from_numpy(np.identity(self.dim)).float()).cuda()

    def select_param(self):

        for layer_name, layer_param in self.param_dict.items():
            if self.param_name in layer_name:
                if 'weight' in layer_name:
                    return layer_param

    def forward(self):

        constrainted_matrix = self.select_param()
        matrxi_ = torch.squeeze(torch.squeeze(constrainted_matrix,dim=2),dim=2)
        matrxi_t = torch.t(matrxi_)
        matrxi_contrainted = (torch.mm(matrxi_t,matrxi_) - self.identity_matrix).view(self.dim ** 2)
        regularizer = torch.pow(matrxi_contrainted, 2).sum(dim=0)
        return regularizer

class SampleHistogramLoss(nn.Module):
    def __init__(self, num_steps):
        super(SampleHistogramLoss, self).__init__()
        self.step = 1.0 / (num_steps - 1)
        self.t = torch.range(0, 1, self.step).view(-1, 1).cuda()
        self.tsize = self.t.size()[0]

    def forward(self,feat_t0,feat_t1, label):
        def histogram(inds, size):
            s_repeat_ = s_repeat.clone()
            indsa = (delta_repeat == (self.t - self.step)) & inds
            indsb = (delta_repeat == self.t) & inds
            s_repeat_[~(indsb | indsa)] = 0
            s_repeat_[indsa] = (s_repeat_ - Variable(self.t) + self.step)[indsa] / self.step
            s_repeat_[indsb] = (-s_repeat_ + Variable(self.t) + self.step)[indsb] / self.step
            return s_repeat_.sum(1) / size

class BhattacharyyaDistance(nn.Module):
    def __init__(self):
        super(BhattacharyyaDistance, self).__init__()

    def forward(self,hist1,hist2):

        bh_dist = (torch.sqrt(hist1 * hist2)).sum()
        return bh_dist

class KLCoefficient(nn.Module):
    def __init__(self):
        super(KLCoefficient, self).__init__()

    def forward(self,hist1,hist2):

        kl = F.kl_div(hist1,hist2)
        dist = 1. / 1 + kl
        return dist

class HistogramMaskLoss(nn.Module):
    def __init__(self,num_steps,dist_flag='l2'):
        super(HistogramMaskLoss, self).__init__()
        self.step = 1.0 / (num_steps - 1)
        self.t = torch.range(0, 1, self.step).view(-1, 1)
        self.dist_flag = dist_flag
        self.distance = KLCoefficient()

    def various_distance(self,out_vec_t0,out_vec_t1):
        if self.dist_flag == 'l2':
            distance = F.pairwise_distance(out_vec_t0,out_vec_t1,p=2)
        if self.dist_flag == 'cos':
            similarity = F.cosine_similarity(out_vec_t0,out_vec_t1)
            distance = 1 - 2 * similarity/np.pi
        return distance

    def histogram(self):
        pass

    def forward(self,feat_t0,feat_t1,ground_truth):

        n,c,h,w = feat_t0.data.shape
        out_t0_rz = torch.transpose(feat_t0.view(c,h*w),1,0)
        out_t1_rz = torch.transpose(feat_t1.view(c,h*w),1,0)
        gt_np = ground_truth.view(h * w).data.cpu().numpy()
        #### inspired by Source code from Histogram loss ###
        ### get all pos in positive pairs and negative pairs ###
        pos_inds_np,neg_inds_np = np.squeeze(np.where(gt_np == 0), 1),np.squeeze(np.where(gt_np !=0),1)
        pos_size,neg_size = pos_inds_np.shape[0],neg_inds_np.shape[0]
        pos_inds,neg_inds = torch.from_numpy(pos_inds_np).cuda(),torch.from_numpy(neg_inds_np).cuda()
        ### get similarities(l2 distance) for all position ###
        distance = torch.squeeze(self.various_distance(out_t0_rz,out_t1_rz),dim=1)
        ### build similarity histogram of positive pairs and negative pairs ###
        pos_dist_ls,neg_dist_ls = distance[pos_inds],distance[neg_inds]
        pos_dist_ls_t,neg_dist_ls_t = torch.from_numpy(pos_dist_ls.data.cpu().numpy()),torch.from_numpy(neg_dist_ls.data.cpu().numpy())
        hist_pos = Variable(torch.histc(pos_dist_ls_t,bins=100,min=0,max=1)/pos_size,requires_grad=True)
        hist_neg = Variable(torch.histc(neg_dist_ls_t,bins=100,min=0,max=1)/neg_size,requires_grad=True)
        loss = self.distance(hist_pos,hist_neg)
        return loss
