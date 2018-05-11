import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data
import torchvision.transforms as transform
import torchvision.datasets as dates
from torch.autograd import Variable
from torch.nn import functional as F
import utils.transforms as trans
import utils.metrics as metric
import utils.utils as util
import utils.resize as rz
import utils.postprocess as post
import utils.metric as mc
import layer.loss as ls
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
if datasets == 'AICD':
    import cfgs.AICDconfig as cfg
    import dataset.AICD as dates
if datasets == 'CD2014':
    import cfgs.CD2014config as cfg
    import dataset.CD2014 as dates

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

resume = 0

def set_base_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'fc8' not in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_2x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'fc8' not in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def set_10x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'fc8' in layer_name:
            if 'weight' in layer_name:
                yield layer_param

def set_20x_learning_rate_for_multi_layer(model):

    params_dict = dict(model.named_parameters())
    for layer_name, layer_param in params_dict.items():
        if 'fc8' in layer_name:
            if 'bias' in layer_name:
                yield layer_param

def untransform(transform_img,mean_vector):

    transform_img = transform_img.transpose(1,2,0)
    transform_img += mean_vector
    transform_img = transform_img.astype(np.uint8)
    transform_img = transform_img[:,:,::-1]
    return transform_img

def single_layer_similar_heatmap_visual(output_t0,output_t1,save_spatial_att_dir,epoch,filename,layer_flag):

    root_name, sub_name, real_name = filename[:filename.find('/')],\
                                     filename[filename.find('/') + 1:filename.rfind('/') - 3], filename[filename.rfind('/') + 1:]
    interp = nn.Upsample(size=[cfg.TRANSFROM_SCALES[1],cfg.TRANSFROM_SCALES[0]], mode='bilinear')
    n, c, h, w = output_t0.data.shape
    out_t0_rz = torch.transpose(output_t0.view(c, h * w), 1, 0)
    out_t1_rz = torch.transpose(output_t1.view(c, h * w), 1, 0)
    distance = F.pairwise_distance(out_t0_rz, out_t1_rz, p=2)
    similar_distance_map = distance.view(h,w).data.cpu().numpy()
    similar_distance_map_rz = interp(Variable(torch.from_numpy(similar_distance_map[np.newaxis, np.newaxis, :])))
    similar_dis_map_colorize = cv2.applyColorMap(np.uint8(255 * similar_distance_map_rz.data.cpu().numpy()[0][0]), cv2.COLORMAP_JET)
    save_spatial_att_dir_ = os.path.join(save_spatial_att_dir, 'epoch_' + str(epoch))
    check_dir(save_spatial_att_dir_)
    root_dir, sub_dir = os.path.join(save_spatial_att_dir_, root_name), os.path.join(save_spatial_att_dir_, root_name, sub_name)
    check_dir(root_dir), check_dir(sub_dir)
    save_spatial_att_dir_layer = os.path.join(sub_dir,layer_flag)
    check_dir(save_spatial_att_dir_layer)
    save_weight_fig_dir = os.path.join(save_spatial_att_dir_layer, real_name[:real_name.find('.')] + '.jpg')
    cv2.imwrite(save_weight_fig_dir, similar_dis_map_colorize)

def validate(net, val_dataloader,epoch,save_valid_imags_dir,save_spatial_att_dir,save_roc_dir):

    net.eval()
    gts, preds = [], []
    metric_for_class = util.init_metric_for_class_for_cd2014(number_class='2')
    for batch_idx, batch in enumerate(val_dataloader):

        inputs1,input2, targets, filename, height, width = batch
        height, width, filename = height.numpy()[0], width.numpy()[0], filename[0]
        root_name,sub_name,real_name = filename[:filename.find('/')],filename[filename.find('/') + 1:filename.rfind('/') -3],filename[filename.rfind('/') + 1:]
        inputs1,input2,targets = inputs1.cuda(),input2.cuda(), targets.cuda()
        inputs1,inputs2,targets = Variable(inputs1, volatile=True),Variable(input2, volatile=True),Variable(targets)
        seg_out,output_conv5,output_fc = net(inputs1,inputs2)
        prob_seg = F.softmax(seg_out).data.cpu().numpy()[0]
              #prob = prob_seg.data.cpu().numpy()[0]
        #interp = nn.Upsample(size=(height, width), mode='bilinear')
        #### similar_distance_map ####
        out_conv5_t0, out_conv5_t1 = output_conv5
        out_fc_t0, out_fc_t1 = output_fc
        single_layer_similar_heatmap_visual(out_conv5_t0,out_conv5_t1,save_spatial_att_dir,epoch,filename,'conv5')
        single_layer_similar_heatmap_visual(out_fc_t0,out_fc_t1,save_spatial_att_dir,epoch,filename,'fc')
        #### seg prediction ###
        seg_pred = np.squeeze(seg_out.data.max(1)[1].cpu().numpy(), axis=0)
        pred_rgb = dates.decode_segmap(seg_pred, plot=False)[:, :, ::-1]
        pred_rgb_rescal = cv2.resize(pred_rgb, (width, height))
        preds.append(seg_pred)
        gt = targets.data.cpu().numpy()
        for gt_ in gt:
            gts.append(gt_)
        for i in range(len(prob_seg)):
            prob_cl = prob_seg[i]
            FN, FP, posNum, negNum = mc.eval_image_rewrite(gt[0], prob_cl, i)
            metric_for_class[root_name][i]['total_fp'] += FP
            metric_for_class[root_name][i]['total_fn'] += FN
            metric_for_class[root_name][i]['total_posnum'] += posNum
            metric_for_class[root_name][i]['total_negnum'] += negNum

        save_valid_dir = os.path.join(save_valid_imags_dir,'epoch_' + str(epoch))
        check_dir(save_valid_dir)
        root_dir, sub_dir = os.path.join(save_valid_dir,root_name),os.path.join(save_valid_dir,root_name,sub_name)
        check_dir(root_dir),check_dir(sub_dir)
        save_fig_dir = os.path.join(sub_dir, real_name[:real_name.find('.')] + '.jpg')
        cv2.imwrite(save_fig_dir, pred_rgb_rescal)

    thresh = np.array(range(0, 256)) / 255.0
    conds = metric_for_class.keys()
    for cond_name in conds:
        for i in range(2):
            total_posnum = metric_for_class[cond_name][i]['total_posnum']
            total_negnum = metric_for_class[cond_name][i]['total_negnum']
            total_fn = metric_for_class[cond_name][i]['total_fn']
            total_fp = metric_for_class[cond_name][i]['total_fp']
            metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                     total_fn, total_fp, thresh=thresh)
            metric_for_class[cond_name][i].setdefault('metric', metric_dict)
    '''''''''
    for i in range(2):
        total_posnum = metric_for_class[i]['total_posnum']
        total_negnum = metric_for_class[i]['total_negnum']
        total_fn = metric_for_class[i]['total_fn']
        total_fp = metric_for_class[i]['total_fp']
        metric_dict = mc.pxEval_maximizeFMeasure(total_posnum, total_negnum,
                                                 total_fn, total_fp, thresh=thresh)
        metric_for_class[i].setdefault('metric', metric_dict)
    '''''''''
    f_score_total = 0.0
    for cond_name in conds:
        for i in range(2):
            pr, recall,f_score = metric_for_class[cond_name][i]['metric']['precision'], metric_for_class[cond_name][i]['metric']['recall'],metric_for_class[cond_name][i]['metric']['MaxF']
            roc_save_epoch_dir = os.path.join(save_roc_dir, str(epoch))
            check_dir(roc_save_epoch_dir)
            roc_save_epoch_cat_dir = os.path.join(roc_save_epoch_dir,cond_name)
            check_dir(roc_save_epoch_cat_dir)
            mc.save_metric2disk(metric_for_class[cond_name],roc_save_epoch_cat_dir)
            roc_save_dir = os.path.join(roc_save_epoch_cat_dir,
                                        '_' + metric_for_class[cond_name][i]['name'] + '_roc.png')
            mc.plotPrecisionRecall(pr, recall, roc_save_dir, benchmark_pr=None)
            f_score_total += f_score

    '''''''''
    for i in range(2):
        pr, recall = metric_for_class[i]['metric']['precision'], metric_for_class[i]['metric']['recall']
        roc_save_epoch_dir = os.path.join(save_roc_dir,str(epoch))
        check_dir(roc_save_epoch_dir)
        roc_save_dir = os.path.join(roc_save_epoch_dir,
                                    '_' + metric_for_class[i]['name'] + '_roc.png')
        mc.plotPrecisionRecall(pr, recall, roc_save_dir, benchmark_pr=None)
    '''''''''
    score, class_iou = metric.scores(gts, preds, n_class=2)
    for k, v in score.items():
        print k, v

    for i in range(2):
        print i, class_iou[i]

    print f_score_total/(2*len(conds))
    return score['Mean IoU :']

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
  ### set transition=True gain better performance ####
  base_seg_model = 'deeplab'
  if base_seg_model == 'deeplab':
      import model.siameseNet.deeplab_v2_fusion as models
      pretrain_deeplab_path = os.path.join(cfg.PRETRAIN_MODEL_PATH, 'deeplab_v2_voc12.pth')
      model = models.SiameseNet(class_number=2,norm_flag='exp')
      if resume:
          checkpoint = torch.load(cfg.TRAINED_BEST_PERFORMANCE_CKPT)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          deeplab_pretrain_model = torch.load(pretrain_deeplab_path)
          #deeplab_pretrain_model = util.load_deeplab_pretrain_model(pretrain_deeplab_path)
          model.init_parameters_from_deeplab(deeplab_pretrain_model)
  else:
      import model.later_fusion.fcn32s as models
      pretrain_vgg_path = os.path.join(cfg.PRETRAIN_MODEL_PATH,'vgg16_from_caffe.pth')
      model = models.fcn32s(class_number=2,transition=True)
      if resume:
          checkpoint = torch.load(cfg.best_ckpt_dir)
          model.load_state_dict(checkpoint['state_dict'])
          print('resume success')
      else:
          vgg_pretrain_model = util.load_pretrain_model(pretrain_vgg_path)
          model.init_parameters(vgg_pretrain_model)

  model = model.cuda()
  MaskLoss = ls.ConstractiveMaskLoss(thresh_flag=True)
  save_training_weights_dir = os.path.join(cfg.SAVE_PRED_PATH,'weights_visual')
  check_dir(save_training_weights_dir)
  save_spatial_att_dir = os.path.join(cfg.SAVE_PRED_PATH, 'various_spatial_att/')
  check_dir(save_spatial_att_dir)
  save_valid_dir = os.path.join(cfg.SAVE_PRED_PATH,'various_valid_imgs')
  check_dir(save_valid_dir)
  save_roc_dir = os.path.join(cfg.SAVE_PRED_PATH,'ROC')
  check_dir(save_roc_dir)

  #########
  ######### optimizer ##########
  ######## how to set different learning rate for differern layer #########
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
             seg_pred,out_conv5,out_fc = model(img1,img2)
             out_conv5_t0,out_conv5_t1 = out_conv5
             out_fc_t0,out_fc_t1 = out_fc
             label_rz_conv5 = rz.resize_label(label.data.cpu().numpy(),size=out_conv5_t0.data.cpu().numpy().shape[2:])
             label_rz_fc = rz.resize_label(label.data.cpu().numpy(),size=out_fc_t0.data.cpu().numpy().shape[2:])
             label_rz_conv5 = Variable(label_rz_conv5.cuda())
             label_rz_fc = Variable(label_rz_fc.cuda())
             seg_loss = util.cross_entropy2d(seg_pred,label,size_average=False)
             constractive_loss_conv5 = MaskLoss(out_conv5_t0,out_conv5_t1,label_rz_conv5)
             constractive_loss_fc = MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
             #constractive_loss = MaskLoss(out_conv5_t0,out_conv5_t1,label_rz_conv5) + \
                                 #MaskLoss(out_fc_t0,out_fc_t1,label_rz_fc)
             loss = seg_loss + cfg.LOSS_PARAM_CONV * constractive_loss_conv5 + cfg.LOSS_PARAM_FC * constractive_loss_fc
             loss_total += loss.data.cpu()
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()
             if (batch_idx) % 20 == 0:
                #print("Epoch [%d/%d] Loss: %.4f" % (epoch, batch_idx, loss.data[0]))
                print("Epoch [%d/%d] Loss: %.4f Seg_Loss: %.4f Mask_Loss_conv5: %.4f Mask_Loss_fc: %.4f" % (epoch, batch_idx, loss.data[0],
                seg_loss.data[0], constractive_loss_conv5.data[0], constractive_loss_fc.data[0]))
             if (batch_idx) % 1000 == 0:
                 model.eval()
                 current_metric = validate(model, val_loader, epoch, save_valid_dir,save_spatial_att_dir,save_roc_dir)
                 if current_metric > best_metric:
                     torch.save({'state_dict': model.state_dict()},
                             os.path.join(cfg.SAVE_CKPT_PATH, 'model' + str(epoch) + '.pth'))
                     shutil.copy(os.path.join(cfg.SAVE_CKPT_PATH, 'model' + str(epoch) + '.pth'),
                              os.path.join(cfg.SAVE_CKPT_PATH, 'model_best.pth'))
                     best_metric = current_metric
      current_metric = validate(model, val_loader, epoch,save_valid_dir,save_spatial_att_dir,save_roc_dir)
      if current_metric > best_metric:
         torch.save({'state_dict': model.state_dict()},
                     os.path.join(cfg.SAVE_CKPT_PATH, 'model' + str(epoch) + '.pth'))
         shutil.copy(os.path.join(cfg.SAVE_CKPT_PATH, 'model' + str(epoch) + '.pth'),
                     os.path.join(cfg.SAVE_CKPT_PATH, 'model_best.pth'))
         best_metric = current_metric
      if epoch % 5 == 0:
          torch.save({'state_dict': model.state_dict()},
                       os.path.join(cfg.SAVE_CKPT_PATH, 'model' + str(epoch) + '.pth'))

if __name__ == '__main__':
   main()
