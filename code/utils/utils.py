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
#import pydensecrf.densecrf as dcrf
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

def adjust_learning_rate(learning_rate,optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (step // 20000))
    #print(str(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adaptive_select_gamma(epoch):

    #gammas = [0,0.5,1,2,5]
    if epoch >= 10:
        gamma = 4
    else:
        gamma = 0
    return gamma

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

def load_metric_json(json_path):
    with open(json_path,'r') as f:
        metric = json.load(f)
    return  metric

def init_metric_for_testing_different_threshold_cd2014():

    metric_for_various_camera_conditions = {}
    condition_names = ['continuousPan','intermittentPan','twoPositionPTZCam','zoomInZoomOut']
    for name in condition_names:
        metric_for_conds = {}
        thresh = np.array(range(0, 256)) / 255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        metric_for_conds.setdefault('total_fp', total_fp)
        metric_for_conds.setdefault('total_fn', total_fn)
        metric_for_conds.setdefault('total_posnum', 0)
        metric_for_conds.setdefault('total_negnum', 0)
        metric_for_various_camera_conditions.setdefault(name,metric_for_conds)

    return metric_for_various_camera_conditions

def init_metric_for_class_for_cd2014(number_class):

    metric_for_various_condition = {}
    condition_names = ['badWeather','baseline','cameraJitter','dynamicBackground','intermittentObjectMotion',
                       'lowFramerate','nightVideos','PTZ','shadow','thermal','turbulence']
    for names in condition_names:
        metric_for_class = {}
        #name = ['NoChange', 'Change']
        for i in range(int(number_class)):
            metric_for_each = {}
            thresh = np.array(range(0, 256)) / 255.0
            total_fp = np.zeros(thresh.shape)
            total_fn = np.zeros(thresh.shape)
            #metric_for_each.setdefault('name', name[i])
            metric_for_each.setdefault('total_fp', total_fp)
            metric_for_each.setdefault('total_fn', total_fn)
            metric_for_each.setdefault('total_posnum', 0)
            metric_for_each.setdefault('total_negnum', 0)
            metric_for_class.setdefault(i, metric_for_each)
        metric_for_various_condition.setdefault(names,metric_for_class)

    return metric_for_various_condition

def init_metric_for_class_for_cmu(number_class):

    metric_for_class = {}
    for i in range(number_class):
        metric_for_each = {}
        thresh = np.array(range(0, 256)) / 255.0
        total_fp = np.zeros(thresh.shape)
        total_fn = np.zeros(thresh.shape)
        metric_for_each.setdefault('total_fp', total_fp)
        metric_for_each.setdefault('total_fn', total_fn)
        metric_for_each.setdefault('total_posnum', 0)
        metric_for_each.setdefault('total_negnum', 0)
        metric_for_class.setdefault(i, metric_for_each)
    return metric_for_class

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

def resize_label(label, size):

    label = np.expand_dims(label,axis=0)
    label_resized = np.zeros((1,label.shape[0],size[0],size[1]))
    interp = nn.Upsample(size=(size[0], size[1]),mode='bilinear')
    #interp = nn.Upsample(size=(size[0], size[1]),mode='bilinear',align_corners=True)
    #labelVar = torch.from_numpy(label).float()
    labelVar = Variable(torch.from_numpy(label).float())
    label_resized[:, :,:,:] = interp(labelVar).data.numpy()
    label_resized = np.array(label_resized, dtype=np.int32)
    return torch.from_numpy(np.squeeze(label_resized,axis=0)).float()
