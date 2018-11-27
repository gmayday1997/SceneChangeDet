import numpy as np
import os
import utils.utils as util
import pylab

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
    COLORMAP = {
        'r': {'marker': "None", 'dash': (None, None)},
        'g': {'marker': "None", 'dash': (None, None)},
        'm': {'marker': "None", 'dash': (None, None)},
        'b': {'marker': "None", 'dash': (None, None)},
        'c': {'marker': "None", 'dash': (None, None)},
        'y': {'marker': "None", 'dash': (None, None)},
        'k': {'marker': 'None', 'dash': (None, None)}  # [1,2,1,10]}
    }
    for line in ax.get_lines():
        origColor = line.get_color()
        line.set_dashes(COLORMAP[origColor]['dash'])
        line.set_marker(COLORMAP[origColor]['marker'])
        line.set_markersize(MARKERSIZE)

def get_csf_ls(base_dir,number):

    CSF_dir = os.path.join(base_dir)
    csf_conv5_ls,csf_fc6_ls,csf_fc7_ls,x = [],[],[],[]
    for i in range(number):
        CSF_sub_dir = os.path.join(CSF_dir,str(i),'metric.json')
        CSF = util.load_metric_json(CSF_sub_dir)
        csf_conv5,csf_fc6,csf_fc7 = CSF['contrast_conv5'],CSF['contrast_fc'],CSF['contrast_embedding']
        csf_conv5_ls.append(csf_conv5),csf_fc6_ls.append(csf_fc6),csf_fc7_ls.append(csf_fc7),x.append(i)
    return csf_conv5_ls,csf_fc6_ls,csf_fc7_ls,x

def main():

    l2_base_dir = '/media/admin228/00027E210001A5BD/train_pytorch/change_detection/CMU/prediction_cons/l2_5,6,7/roc'
    cos_base_dir = '/media/admin228/00027E210001A5BD/train_pytorch/change_detection/CMU/prediction_cons/dist_cos_new_5,6,7/roc'
    CSF_dir = os.path.join(l2_base_dir)
    CSF_fig_dir = os.path.join(l2_base_dir,'fig.png')
    end_number = 22
    csf_conv5_l2_ls,csf_fc6_l2_ls,csf_fc7_l2_ls,x_l2 = get_csf_ls(l2_base_dir,end_number)
    csf_conv5_cos_ls,csf_fc6_cos_ls,csf_fc7_cos_ls,x_cos = get_csf_ls(cos_base_dir,end_number)
    Fig = pylab.figure()
    setFigLinesBW(Fig)
    #pylab.plot(x,csf_conv4_ls, color='k',label= 'conv4')
    pylab.plot(x_l2,csf_conv5_l2_ls, color='m',label= 'l2:conv5')
    pylab.plot(x_l2,csf_fc6_l2_ls, color = 'b',label= 'l2:fc6')
    pylab.plot(x_l2,csf_fc7_l2_ls, color = 'g',label= 'l2:fc7')
    pylab.plot(x_cos,csf_conv5_cos_ls, color='c',label= 'cos:conv5')
    pylab.plot(x_cos,csf_fc6_cos_ls, color = 'r',label= 'cos:fc6')
    pylab.plot(x_cos,csf_fc7_cos_ls, color = 'y',label= 'cos:fc7')
    pylab.legend(loc='lower right', prop={'size': 10})
    pylab.ylabel('RMS Contrast', fontsize=14)
    pylab.xlabel('Epoch', fontsize=14)
    pylab.savefig(CSF_fig_dir)


if __name__ == '__main__':
    main()
