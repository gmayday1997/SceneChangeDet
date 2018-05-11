import torch
import torchvision 
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import model.vgg1024 as vgg
import model.deeplab_msc_coco as attmodel
import json
import cv2
import pydensecrf.densecrf as dcrf
import os

def load_deeplab_v2(model_file):

    model = deeplab_v2.deeplab_vgg_v2()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    print('deeplabv2 has been load')
    return model

def load_pretrain_model(model_file):

   model = torchvision.models.vgg16(pretrained=False)
   state_dict = torch.load(model_file)
   model.load_state_dict(state_dict)
   print('model has been load')
   return model

def load_pretrain_coco_attention_model(model_file):

    model = attmodel.Deeplab_MS_Att_Scale(class_number=91)
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
    print('model has been load')
    return model

def load_deeplab_pretrain_model(model_file):

   model = vgg.vgg1024()
   state_dict = torch.load(model_file)
   model.load_state_dict(state_dict)
   print('model has been load')
   return model

def load_deeplab_best_metric_model(model_file):

    model = deeplab.deeplab()
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict['state_dict'])
    print('model has been load')
    return model

def check_dir(dir):

    if not os.path.exists(dir):
        os.mkdir(dir)

def loss_calc(out, label, gpu0):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:, :, 0, :].transpose(2, 0, 1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    out = m(out)
    return criterion(out, label)

###### source code from https://github.com/meetshah1995/pytorch-semseg #####
def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def adjust_learning_rate(learning_rate,optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (step // 40000))
    #print(str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power

def save2json(metric_dict,save_path):
    file_ = open(save_path,'w')
    file_.write(json.dumps(metric_dict,ensure_ascii=False,indent=2))
    file_.close()


def init_metric_for_class_for_cd2014(number_class):

    metric_for_various_condition = {}
    condition_names = ['badWeather','baseline','cameraJitter','dynamicBackground','intermittentObjectMotion',
                       'lowFramerate','nightVideos','PTZ','shadow','thermal','turbulence']
    for names in condition_names:
        metric_for_class = {}
        name = ['NoChange', 'Change']
        for i in range(int(number_class)):
            metric_for_each = {}
            thresh = np.array(range(0, 256)) / 255.0
            total_fp = np.zeros(thresh.shape)
            total_fn = np.zeros(thresh.shape)
            metric_for_each.setdefault('name', name[i])
            metric_for_each.setdefault('total_fp', total_fp)
            metric_for_each.setdefault('total_fn', total_fn)
            metric_for_each.setdefault('total_posnum', 0)
            metric_for_each.setdefault('total_negnum', 0)
            metric_for_class.setdefault(i, metric_for_each)
        metric_for_various_condition.setdefault(names,metric_for_class)

    return metric_for_various_condition

def init_metric_for_class(number_class):

    metric_for_class = {}
    name = ['NoChange','Change']
    for i in range(int(number_class)):

        metric_for_each = {}
        thresh = np.array(range(0, 256)) / 255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        metric_for_each.setdefault('name',name[i])
        metric_for_each.setdefault('total_fp',total_fp)
        metric_for_each.setdefault('total_fn',total_fn)
        metric_for_each.setdefault('total_posnum',0)
        metric_for_each.setdefault('total_negnum',0)
        metric_for_class.setdefault(i,metric_for_each)

    return metric_for_class

def attention_weights_collection(attention_weights):

    loc_weights_dict = {}
    locs,height,width = attention_weights.shape
    for idx in range(locs):
       loc_weights = attention_weights[idx,:,:]
       loc_attention_vec = np.reshape(loc_weights,(height * width))
       max_ = np.max(loc_attention_vec,axis=0)
       if max_ != 0:
           loc_attention_vec = loc_attention_vec/max_
           loc_attention = loc_attention_vec.reshape(height,width)
       else:
           loc_attention = loc_attention_vec.reshape(height,width)
       loc_weights_dict.setdefault(idx,loc_attention)

    return loc_weights_dict

def attention_weights_visulize(weights_dict,original_img,save_base_path):

    for idx,loc_attention_weight_vec in weights_dict.iteritems():

        height, width, channel = original_img.shape
        alpha_att_map = cv2.resize(loc_attention_weight_vec, (width,height), interpolation=cv2.INTER_LINEAR)
        alpha_att_map_ = cv2.applyColorMap(np.uint8(255 * alpha_att_map), cv2.COLORMAP_JET)
        fuse_heat_map = 0.6 * alpha_att_map_ + 0.4 * original_img
        cv2.imwrite(save_base_path + '_' + str(idx) + '.jpg',fuse_heat_map)
        #print idx

def various_scale_attention_weights_visualize(spatial_weights,original_img1,original_img2,save_base_path,filename):

    nchannel, height,width = spatial_weights.shape
    scale_list = ['common','t0','t1']
    original_imgs = [original_img1,original_img1,original_img2]
    assert len(scale_list) == len(spatial_weights)
    for idx in range(nchannel):

        height_img, width_img, channel = original_imgs[idx].shape
        scale_x = spatial_weights[idx]
        scale_name = scale_list[idx]
        scalex_x_att_map = cv2.resize(scale_x,(width_img,height_img),interpolation=cv2.INTER_LINEAR)
        scalex_x_att_map_ = cv2.applyColorMap(np.uint8(255* scalex_x_att_map),cv2.COLORMAP_JET)
        fuse_scale_att_map = 0.6 * scalex_x_att_map_ + 0.4 * original_imgs[idx]
        cv2.imwrite(save_base_path + '_' + str(filename) + '_origin_' + str(scale_name) + '.jpg', scalex_x_att_map_)
        cv2.imwrite(save_base_path + '_' + str(filename) + '_fuse_' + str(scale_name) + '.jpg', fuse_scale_att_map)

def cross_entropy(pred,label,n_class):

    soft_max = nn.LogSoftmax()
    criterion = CrossEntropy2d(n_class=n_class).cuda()
    pred = soft_max(pred)
    return criterion(pred, label)

###### ingore invalid label #########
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, n_class = 21 ,ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.n_class = n_class

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        #target_mask = (target >= 0) * (target <= self.n_class)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss
