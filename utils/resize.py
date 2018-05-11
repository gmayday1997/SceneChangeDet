import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def resize_label(label, size):

    label = np.expand_dims(label,axis=0)
    label_resized = np.zeros((1,label.shape[0],size[0],size[1]))
    interp = nn.Upsample(size=(size[0], size[1]),mode='bilinear')
    labelVar = Variable(torch.from_numpy(label).float())
    label_resized[:, :,:,:] = interp(labelVar).data.numpy()
    label_resized = np.array(label_resized, dtype=np.int32)
    return torch.from_numpy(np.squeeze(label_resized,axis=0)).long()
