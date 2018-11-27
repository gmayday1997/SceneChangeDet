import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import utils.transforms as trans
import utils.utils as util
import layer.loss as ls
import utils.metric as mc
import shutil
import cv2

### options = ['TSUNAMI','GSV','CMU','AICD','CD2014']
datasets = 'CD2014'
if datasets == 'TSUNAMI':
    import cfgs.TSUNAMIconfig as cfg
    import dataset.TSUNAMI as dates
if datasets == 'GSV':
    import cfgs.GSVconfig as cfg
    import dataset.GSV as dates
if datasets == 'CMU':
    import cfgs.CMUconfig as cfg
    import dataset.CMU as dates
if datasets == 'CD2014':
    import cfgs.CD2014config as cfg
    import dataset.CD2014 as dates

resume = 0

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def set_base_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' not in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_2x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' not in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def set_10x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_20x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'embedding_layer' in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def untransform(transform_img,mean_vector):

    transform_img = transform_img.transpose(1,2,0)
    transform_img += mean_vector
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def various_distance(out_vec_t0, out_vec_t1,dist_flag):
    if dist_flag == 'l2':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=2)
    if dist_flag == 'l1':
        distance = F.pairwise_distance(out_vec_t0, out_vec_t1, p=1)
    if dist_flag == 'cos':
        distance = 1 - F.cosine_similarity(out_vec_t0, out_vec_t1)
    return distance

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_change_map_dir,epoch,filename,layer_flag,dist_flag):

    interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = various_distance(out_t0_rz,out_t1_rz,dist_flag=dist_flag)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    save_change_map_dir_ = os.path.join(save_change_map_dir, 'epoch_' + str(epoch))
    check_dir(save_change_map_dir_)
    save_change_map_dir_layer = os.path.join(save_change_map_dir_,layer_flag)
    check_dir(save_change_map_dir_layer)
    save_weight_fig_dir = os.path.join(save_change_map_dir_layer, filename + '.jpg')
    cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)
    return similar_distance_map_rz.data.cpu().numpy()

def validate(net, val_dataloader,epoch,save_change_map_dir,save_roc_dir):

    net.eval()
    cont_conv5_total,cont_fc_total,cont_embedding_total,num = 0.0,0.0,0.0,0.0
    metric_for_conditions = util.init_metric_for_class_for_cmu(1)
    for batch_idx, batch in enumerate(val_dataloader):
        inputs1,input2, targets, filename, height, width = batch
        height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
        inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
        inputs1,inputs2,targets = Variable(inputs1, volatile=True),Variable(input2,volatile=True) ,Variable(targets)
        out_conv5,out_fc,out_embedding = net(inputs1,inputs2)
        out_conv5_t0, out_conv5_t1 = out_conv5
        out_fc_t0,out_fc_t1 = out_fc
        out_embedding_t0,out_embedding_t1 = out_embedding
        conv5_distance_map = single_layer_similar_heatmap_visual(out_conv5_t0,out_conv5_t1,save_change_map_dir,epoch,filename,'conv5','l2')
        fc_distance_map = single_layer_similar_heatmap_visual(out_fc_t0,out_fc_t1,save_change_map_dir,epoch,filename,'fc','l2')
        embedding_distance_map = single_layer_similar_heatmap_visual(out_embedding_t0,out_embedding_t1,save_change_map_dir,epoch,filename,'embedding','l2')
        cont_conv5 = mc.RMS_Contrast(conv5_distance_map)
        cont_fc = mc.RMS_Contrast(fc_distance_map)
        cont_embedding = mc.RMS_Contrast(embedding_distance_map)
        cont_conv5_total += cont_conv5
        cont_fc_total += cont_fc
        cont_embedding_total += cont_embedding
        num += 1
        prob_change = embedding_distance_map[0][0]
        gt = targets.data.cpu().numpy()
        FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_change, cl_index=1)
        metric_for_conditions[0]['total_fp'] += FP
        metric_for_conditions[0]['total_fn'] += FN
        metric_for_conditions[0]['total_posnum'] += posNum
        metric_for_conditions[0]['total_negnum'] += negNum
        cont_conv5_mean, cont_fc_mean,cont_embedding_mean = cont_conv5_total/num, \
                                                                            cont_fc_total/num,cont_embedding_total/num

    thresh = np.array(range(0, 256)) / 255.0
    conds = metric_for_conditions.keys()
    for cond_name in conds:
        total_posnum = metric_for_conditions[cond_name]['total_posnum']
        total_negnum = metric_for_conditions[cond_name]['total_negnum']
        total_fn = metric_for_conditions[cond_name]['total_fn']
        total_fp = metric_for_conditions[cond_name]['total_fp']
        metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                total_fn, total_fp, thresh=thresh)
        metric_for_conditions[cond_name].setdefault('metric', metric_dict)
        metric_for_conditions[cond_name].setdefault('contrast_conv5', cont_conv5_mean)
        metric_for_conditions[cond_name].setdefault('contrast_fc',cont_fc_mean)
        metric_for_conditions[cond_name].setdefault('contrast_embedding',cont_embedding_mean)

    f_score_total = 0.0
    for cond_name in conds:
        pr, recall,f_score = metric_for_conditions[cond_name]['metric']['precision'], metric_for_conditions[cond_name]['metric']['recall'],metric_for_conditions[cond_name]['metric']['MaxF']
        roc_save_epoch_dir = os.path.join(save_roc_dir, str(epoch))
        check_dir(roc_save_epoch_dir)
        roc_save_epoch_cat_dir = os.path.join(roc_save_epoch_dir)
        check_dir(roc_save_epoch_cat_dir)
        mc.save_PTZ_metric2disk(metric_for_conditions[cond_name],roc_save_epoch_cat_dir)
        roc_save_dir = os.path.join(roc_save_epoch_cat_dir,
                                        '_' + str(cond_name) + '_roc.png')
        mc.plotPrecisionRecall(pr, recall, roc_save_dir, benchmark_pr=None)
        f_score_total += f_score

    print f_score_total/(len(conds))
    return f_score_total/len(conds)

def main():

  #########  configs ###########
  best_metric = 0
  ######  load datasets ########
  train_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
  ])
  val_transform_det = trans.Compose([
      trans.Scale(cfg.TRANSFROM_SCALES),
  ])
  train_data = dates.Dataset(cfg.TRAIN_DATA_PATH,cfg.TRAIN_LABEL_PATH,
                                cfg.TRAIN_TXT_PATH,'train',transform=True,
                                transform_med = train_transform_det)
  train_loader = Data.DataLoader(train_data,batch_size=cfg.BATCH_SIZE,
                                 shuffle= True, num_workers= 4, pin_memory= True)
  val_data = dates.Dataset(cfg.VAL_DATA_PATH,cfg.VAL_LABEL_PATH,
                            cfg.VAL_TXT_PATH,'val',transform=True,
                            transform_med = val_transform_det)
  val_loader = Data.DataLoader(val_data, batch_size= cfg.BATCH_SIZE,
                                shuffle= False, num_workers= 4, pin_memory= True)
  ######  build  models ########
  base_seg_model = 'deeplab'
  if base_seg_model == 'deeplab':
      import model.siameseNet.deeplab_v2 as models
      pretrain_deeplab_path = os.path.join(cfg.PRETRAIN_MODEL_PATH, 'deeplab_v2_voc12.pth')
      model = models.SiameseNet(norm_flag='l2')
      if resume:
          checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
          model.init_parameters_from_deeplab(deeplab_pretrain_model)
  else:
      import model.siameseNet.fcn32s_tiny as models
      pretrain_vgg_path = os.path.join(cfg.PRETRAIN_MODEL_PATH,'vgg16_from_caffe.pth')
      model = models.SiameseNet(distance_flag='softmax')
      if resume:
          checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          vgg_pretrain_model = util.load_pretrain_model(pretrain_vgg_path)
          model.init_parameters(vgg_pretrain_model)

  model = model.cuda()
  MaskLoss = ls.ConstractiveMaskLoss()
  ab_test_dir = os.path.join(cfg.SAVE_PRED_PATH,'contrastive_loss')
  check_dir(ab_test_dir)
  save_change_map_dir = os.path.join(ab_test_dir, 'changemaps/')
  save_valid_dir = os.path.join(ab_test_dir,'valid_imgs')
  save_roc_dir = os.path.join(ab_test_dir,'roc')
  check_dir(save_change_map_dir),check_dir(save_valid_dir),check_dir(save_roc_dir)
  #########
  ######### optimizer ##########
  ######## how to set different learning rate for differernt layers #########
  optimizer = torch.optim.SGD(
      [
          {'params':set_base_learning_rate_for_multi_layer(model),'lr':cfg.INIT_LEARNING_RATE},
          {'params':set_2x_learning_rate_for_multi_layer(model),'lr': 2 * cfg.INIT_LEARNING_RATE,'weight_decay':0},
          {'params':set_10x_learning_rate_for_multi_layer(model),'lr':10 * cfg.INIT_LEARNING_RATE},
          {'params':set_20x_learning_rate_for_multi_layer(model),'lr': 20 * cfg.INIT_LEARNING_RATE,'weight_decay':0}
      ],lr=cfg.INIT_LEARNING_RATE,momentum=cfg.MOMENTUM,weight_decay=cfg.DECAY)
  ######## iter img_label pairs ###########
  loss_total = 0
  for epoch in range(100):
      for batch_idx, batch in enumerate(train_loader):
             step = epoch * len(train_loader) + batch_idx
             util.adjust_learning_rate(cfg.INIT_LEARNING_RATE, optimizer, step)
             model.train()
             img1_idx,img2_idx,label_idx, filename,height,width = batch
             img1,img2,label = Variable(img1_idx.cuda()),Variable(img2_idx.cuda()),Variable(label_idx.cuda())
             out_conv5, out_fc,out_embedding = model(img1, img2)
             out_conv5_t0,out_conv5_t1 = out_conv5
             out_fc_t0,out_fc_t1 = out_fc
             out_embedding_t0,out_embedding_t1 = out_embedding
             label_rz_conv5 = Variable(util.resize_label(label.data.cpu().numpy(),size=out_conv5_t0.data.cpu().numpy().shape[2:]).cuda())
             label_rz_fc = Variable(util.resize_label(label.data.cpu().numpy(),size=out_fc_t0.data.cpu().numpy().shape[2:]).cuda())
             label_rz_embedding = Variable(util.resize_label(label.data.cpu().numpy(),size=out_embedding_t0.data.cpu().numpy().shape[2:]).cuda())
             contractive_loss_conv5 = MaskLoss(out_conv5_t0,out_conv5_t1,label_rz_conv5)
             contractive_loss_fc = MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
             contractive_loss_embedding = MaskLoss(out_embedding_t0,out_embedding_t1,label_rz_embedding)
             loss = contractive_loss_conv5 + contractive_loss_fc + contractive_loss_embedding
             loss_total += loss.data.cpu()
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             if (batch_idx) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f Mask_Loss_conv5: %.4f Mask_Loss_fc: %.4f "
                      "Mask_Loss_embedding: %.4f" % (epoch, batch_idx,loss.data[0],contractive_loss_conv5.data[0],
                                                     contractive_loss_fc.data[0],contractive_loss_embedding.data[0]))
             if (batch_idx) % 1000 == 0:
                 model.eval()
                 current_metric = validate(model, val_loader, epoch,save_change_map_dir,save_roc_dir)
                 if current_metric > best_metric:
                     torch.save({'state_dict': model.state_dict()},
                             os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
                     shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'),
                              os.path.join(ab_test_dir, 'model_best.pth'))
                     best_metric = current_metric
      current_metric = validate(model, val_loader, epoch,save_change_map_dir,save_roc_dir)
      if current_metric > best_metric:
         torch.save({'state_dict': model.state_dict()},
                     os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))
         shutil.copy(os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'),
                     os.path.join(ab_test_dir, 'model_best.pth'))
         best_metric = current_metric
      if epoch % 5 == 0:
          torch.save({'state_dict': model.state_dict()},
                       os.path.join(ab_test_dir, 'model' + str(epoch) + '.pth'))

if __name__ == '__main__':
   main()
