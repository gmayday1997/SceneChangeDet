#!/usr/bin/env python
#
#  THE KITTI VISION BENCHMARK SUITE: ROAD BENCHMARK
#
#  Copyright (C) 2013
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#  Authors: Tobias Kuehnl <tkuehnl@cor-lab.uni-bielefeld.de>
#           Jannik Fritsch <jannik.fritsch@honda-ri.de>
#

import numpy as np
import pylab
import os
import cv2

def getGroundTruth(fileNameGT):
    '''
    Returns the ground truth maps for roadArea and the validArea 
    :param fileNameGT:
    '''
    # Read GT
    assert os.path.isfile(fileNameGT), 'Cannot find: %s' % fileNameGT
    full_gt = cv2.imread(fileNameGT, cv2.CV_LOAD_IMAGE_UNCHANGED)
    #attention: OpenCV reads in as BGR, so first channel has Blue / road GT
    roadArea =  full_gt[:,:,0] > 0
    validArea = full_gt[:,:,2] > 0

    return roadArea, validArea


def overlayImageWithConfidence(in_image, conf, vis_channel = 1, threshold = 0.5):
    '''
    
    :param in_image:
    :param conf:
    :param vis_channel:
    :param threshold:
    '''
    if in_image.dtype == 'uint8':
        visImage = in_image.copy().astype('f4')/255
    else:
        visImage = in_image.copy()
    
    channelPart = visImage[:, :, vis_channel] * (conf > threshold) - conf
    channelPart[channelPart < 0] = 0
    visImage[:, :, vis_channel] = visImage[:, :, vis_channel] * (conf <= threshold) + (conf > threshold) * conf + channelPart
    return visImage

def evalExp(gtBin, cur_prob, thres, validMap = None, validArea=None):
    '''
    Does the basic pixel based evaluation!
    :param gtBin:
    :param cur_prob:
    :param thres:
    :param validMap:
    '''

    assert len(cur_prob.shape) == 2, 'Wrong size of input prob map'
    assert len(gtBin.shape) == 2, 'Wrong size of input prob map'
    thresInf = np.concatenate(([-np.Inf], thres, [np.Inf]))
    
    #Merge validMap with validArea
    if np.any(validMap)!=None:
        if np.any(validArea)!=None:
            validMap = (validMap == True) & (validArea == True)
    elif np.any(validArea)!=None:
        validMap=validArea

    # histogram of false negatives
    if np.any(validMap)!=None:
        #valid_array = cur_prob[(validMap == False)]
        fnArray = cur_prob[(gtBin == True) & (validMap == True)]
    else:
        fnArray = cur_prob[(gtBin == True)]
    #f = np.histogram(fnArray,bins=thresInf)
    fnHist = np.histogram(fnArray,bins=thresInf)[0]
    fn_list = list(fnHist)
    fn_sum = sum(fn_list[2:])
    fnCum = np.cumsum(fnHist)
    FN = fnCum[0:0+len(thres)];
    
    if validMap.any()!=None:
        fpArray = cur_prob[(gtBin == False) & (validMap == True)]
    else:
        fpArray = cur_prob[(gtBin == False)]
    
    fpHist  = np.histogram(fpArray, bins=thresInf)[0]
    fpCum = np.flipud(np.cumsum(np.flipud(fpHist)))
    FP = fpCum[1:1+len(thres)]

    # count labels and protos
    #posNum = fnArray.shape[0]
    #negNum = fpArray.shape[0]
    if np.any(validMap)!=None:
        posNum = np.sum((gtBin == True) & (validMap == True))
        negNum = np.sum((gtBin == False) & (validMap == True))
    else:
        posNum = np.sum(gtBin == True)
        negNum = np.sum(gtBin == False)
    return FN, FP, posNum, negNum

def pxEval_maximizeFMeasure(totalPosNum, totalNegNum, totalFN, totalFP, thresh = None):
    '''

    @param totalPosNum: scalar
    @param totalNegNum: scalar
    @param totalFN: vector
    @param totalFP: vector
    @param thresh: vector
    '''

    #Calc missing stuff
    totalTP = totalPosNum - totalFN
    totalTN = totalNegNum - totalFP


    valid = (totalTP>=0) & (totalTN>=0)
    assert valid.all(), 'Detected invalid elements in eval'

    recall = totalTP / float( totalPosNum )
    precision =  totalTP / (totalTP + totalFP + 1e-10)
    
    selector_invalid = (recall==0) & (precision==0)
    recall = recall[~selector_invalid]
    precision = precision[~selector_invalid]
        
    maxValidIndex = len(precision)
    
    #Pascal VOC average precision
    '''''''''
    AvgPrec = 0
    counter = 0
    for i in np.arange(0,1.1,0.1):
        ind = np.where(recall>=i)
        if ind == None:
            continue
        pmax = max(precision[ind])
        AvgPrec += pmax
        counter += 1
    AvgPrec = AvgPrec/counter
    '''''''''
    # F-measure operation point
    beta = 1.0
    betasq = beta**2
    F = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    index = F.argmax()
    MaxF= F[index]
    
    recall_bst = recall[index]
    precision_bst =  precision[index]

    TP = totalTP[index]
    TN = totalTN[index]
    FP = totalFP[index]
    FN = totalFN[index]
    valuesMaxF = np.zeros((1,4),'u4')
    valuesMaxF[0,0] = TP
    valuesMaxF[0,1] = TN
    valuesMaxF[0,2] = FP
    valuesMaxF[0,3] = FN

    #ACC = (totalTP+ totalTN)/(totalPosNum+totalNegNum)
    prob_eval_scores  = calcEvalMeasures(valuesMaxF)
    #prob_eval_scores['AvgPrec'] = AvgPrec
    prob_eval_scores['MaxF'] = MaxF

    #prob_eval_scores['totalFN'] = totalFN
    #prob_eval_scores['totalFP'] = totalFP
    prob_eval_scores['totalPosNum'] = totalPosNum
    prob_eval_scores['totalNegNum'] = totalNegNum

    prob_eval_scores['precision'] = precision
    prob_eval_scores['recall'] = recall
    #prob_eval_scores['precision_bst'] = precision_bst
    #prob_eval_scores['recall_bst'] = recall_bst
    prob_eval_scores['thresh'] = thresh
    if np.any(thresh) != None:
        BestThresh= thresh[index]
        prob_eval_scores['BestThresh'] = BestThresh

    #return a dict
    return prob_eval_scores


def calcEvalMeasures(evalDict, tag  = '_wp'):
    '''
    
    :param evalDict:
    :param tag:
    '''
    # array mode!
    TP = evalDict[:,0].astype('f4')
    TN = evalDict[:,1].astype('f4')
    FP = evalDict[:,2].astype('f4')
    FN = evalDict[:,3].astype('f4')
    Q = TP / (TP + FP + FN)
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    FPR = FP / N
    FNR = FN / P
    TNR = TN / N
    A = (TP + TN) / (P + N)
    precision = TP / (TP + FP)
    recall = TP / P
    #numSamples = TP + TN + FP + FN
    correct_rate = A

    # F-measure
    #beta = 1.0
    #betasq = beta**2
    #F_max = (1 + betasq) * (precision * recall)/((betasq * precision) + recall + 1e-10)
    
    outDict =dict()

    outDict['TP'+ tag] = TP
    outDict['FP'+ tag] = FP
    outDict['FN'+ tag] = FN
    outDict['TN'+ tag] = TN
    outDict['Q'+ tag] = Q
    outDict['A'+ tag] = A
    outDict['TPR'+ tag] = TPR
    outDict['FPR'+ tag] = FPR
    outDict['FNR'+ tag] = FNR
    outDict['PRE'+ tag] = precision
    outDict['REC'+ tag] = recall
    outDict['correct_rate'+ tag] = correct_rate
    return outDict

def setFigLinesBW(fig):
    """
    Take each axes in the figure, and for each line in the axes, make the
    line viewable in black and white.
    """
    for ax in fig.get_axes():
        setAxLinesBW(ax)
        
def setAxLinesBW(ax):
    """
    Take each Line2D in the axes, ax, and convert the line style to be
    suitable for black and white viewing.
    """
    MARKERSIZE = 3

#     COLORMAP = {
#         'r': {'marker': None, 'dash': (None,None)},
#         'g': {'marker': None, 'dash': [5,2]},
#         'm': {'marker': None, 'dash': [11,3]},
#         'b': {'marker': None, 'dash': [6,3,2,3]},
#         'c': {'marker': None, 'dash': [1,3]},
#         'y': {'marker': None, 'dash': [5,3,1,2,1,10]},
#         'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
#         }
    '''''''''
    COLORMAP = {
        'r': {'marker': "None", 'dash': (None,None)},
        'g': {'marker': "None", 'dash': [5,2]},
        'm': {'marker': "None", 'dash': [11,3]},
        'b': {'marker': "None", 'dash': [6,3,2,3]},
        'c': {'marker': "None", 'dash': [1,3]},
        'y': {'marker': "None", 'dash': [5,3,1,2,1,10]},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }
    '''''''''

    COLORMAP = {
        'r': {'marker': "None", 'dash': (None,None)},
        'g': {'marker': "None", 'dash': (None,None)},
        'm': {'marker': "None", 'dash': (None,None)},
        'b': {'marker': "None", 'dash': (None,None)},
        'c': {'marker': "None", 'dash': (None,None)},
        'y': {'marker': "None", 'dash': (None,None)},
        'k': {'marker': 'o', 'dash': (None,None)} #[1,2,1,10]}
        }


    for line in ax.get_lines():
        origColor = line.get_color()
        #line.set_color('black')

        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)
        
def plotPrecisionRecall(precision, recall, outFileName, benchmark_pr = None, Fig=None, drawCol=0, textLabel = None, title = None, fontsize1 = 14, fontsize2 = 10, linewidth = 3):
    '''
    
    :param precision:
    :param recall:
    :param outFileName:
    :param Fig:
    :param drawCol:
    :param textLabel:
    :param fontsize1:
    :param fontsize2:
    :param linewidth:
    '''
                      
    clearFig = False  
           
    if Fig == None:
        Fig = pylab.figure()
        clearFig = True
        
    #tableString = 'Algo avgprec Fmax prec recall accuracy fpr Q(TonITS)\n'
    linecol = ['r','m','b','c']
    #if we are evaluating SP, then BL is available
    #sectionName = 'Evaluation_'+tag+'PxProb'
    #fullEvalFile = os.path.join(eval_dir,evalName)
    #Precision,Recall,evalString = readEvaluation(fullEvalFile,sectionName,AlgoLabel)

    if benchmark_pr!= None:

        benchmark_recall = np.array(benchmark_pr['recall'])
        benchmark_precision = np.array(benchmark_pr['precision'])
        pylab.plot(100 * benchmark_recall,100 * benchmark_precision,linewidth = linewidth, color= linecol[drawCol],label = textLabel)

    else:

        pylab.plot(100 * recall, 100 * precision, linewidth=2, color=linecol[drawCol], label=textLabel)

    #writing out PrecRecall curves as graphic
    setFigLinesBW(Fig)
    if textLabel!= None:
        pylab.legend(loc='lower left',prop={'size':fontsize2})
    
    if title!= None:
        pylab.title(title, fontsize= fontsize1)

    #pylab.title(title,fontsize=24)
    pylab.ylabel('Precision [%]',fontsize= fontsize1)
    pylab.xlabel('Recall [%]',fontsize= fontsize1)

    pylab.xlim(0,100)
    pylab.xticks( [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      ('0','','0.20','','0.40','','0.60','','0.80','','1.0'), fontsize=fontsize2 )
    pylab.ylim(0,100)
    pylab.yticks( [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                      ('0','','0.20','','0.40','','0.60','','0.80','','1.0'), fontsize=fontsize2 )

    #pylab.grid(True)
    # 
    if type(outFileName) != list:
        pylab.savefig( outFileName )
    else:
        for outFn in outFileName:
            pylab.savefig( outFn )
    if clearFig:
        pylab.close()
        Fig.clear()

def eval_image(gt_image, cnn_image):
    """."""
    thresh = np.array(range(0, 256))/255.0
    road_gt = gt_image[:, :] > 0
    valid_gt = gt_image[:, :] > 0

    FN, FP, posNum, negNum = evalExp(road_gt, cnn_image,
                                         thresh, validMap=None,
                                         validArea=valid_gt)

    return FN, FP, posNum, negNum

def eval_image_rewrite(gt_image, prob,cl_index):

    thresh = np.array(range(0, 256))/255.0
    cl_gt = gt_image[:,:] == cl_index
    valid_gt = gt_image[:,:] != 255

    FN, FP, posNum, negNum = evalExp(cl_gt, prob,
                                     thresh, validMap=None,
                                     validArea=valid_gt)
    return FN, FP, posNum, negNum

def save_PTZ_metric2disk(metrics,save_path):

    import json
    #metric_dict= {}
    recall_ = list(metrics['metric']['recall'])
    precision_ = list(metrics['metric']['precision'])
    f_score = metrics['metric']['MaxF']
    try:
        iu = metrics['metric']['iu']
    except KeyError:
        iu = 0.0
    cont_embedding = metrics['contrast_embedding']
    metric_ = {'recall':recall_,'precision':precision_,'f-score':f_score,'iu':iu,
               'contrast_embedding':cont_embedding}
    file_ = open(save_path + '/metric.json', 'w')
    file_.write(json.dumps(metric_, ensure_ascii=False, indent=2))
    file_.close()

def save_metric2disk(metrics,save_path):

    import json
    length = len(metrics)
    metric_dict= {}
    for i in range(length):
        recall_ = list(metrics[i]['metric']['recall'])
        name = metrics[i]['name']
        precision_ = list(metrics[i]['metric']['precision'])
        f_score = metrics[i]['metric']['MaxF']
        try:
            iu = metrics[i]['metric']['iu']
        except KeyError:
            iu = 0.0
        metric_ = {'name':name,'recall':recall_,'precision':precision_,'f-score':f_score,'iu':iu}
        metric_dict.setdefault(i,metric_)

    file_ = open(save_path + '/metric.json', 'w')
    file_.write(json.dumps(metric_dict, ensure_ascii=False, indent=2))
    file_.close()

def load_metric_json(json_path):
    import json

    with open(json_path,'r') as f:
        metric = json.load(f)

    return  metric

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc:': acc,
            'Mean Acc :': acc_cls,
            'FreqW Acc :': fwavacc,
            'Mean IoU :': mean_iu,}, cls_iu

def RMS_Contrast(dist_map):

    n,c,h,w = dist_map.shape
    dist_map_l = np.resize(dist_map,(n*c*h*w))
    mean = np.mean(dist_map_l,axis=0)
    std = np.std(dist_map_l,axis=0,ddof=1)
    contrast = std / mean
    return contrast