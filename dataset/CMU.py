import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import scipy.io
import scipy.misc as m
from PIL import Image
import matplotlib.pyplot as plt
import utils.transforms as trans
import cv2
import cfgs.CMUconfig as cfg

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

palette = [0, 0, 0,255,0,0]

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_pascal_labels():
    return np.asarray([[0,0,0],[0,0,255]])

def decode_segmap(temp, plot=False):

    label_colours = get_pascal_labels()
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    #rgb = np.resize(rgb,(321,321,3))
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

#### source dataset is only avaiable by sending an email request to author #####
#### upon request is shown in http://ghsi.github.io/proj/RSS2016.html ####
#### more details are presented in http://ghsi.github.io/assets/pdfs/alcantarilla16rss.pdf ###

class Dataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True, transform_med = None):

        self.label_path = label_path
        self.img_path = img_path
        #self.img2_path = img2_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.transform_med = transform_med
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag =='train':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name,image2_name,mask_name,mask_reverse_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,extract_name])

        if self.flag == 'val':
            self.label_ext = '.png'
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name, mask_reverse_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path , mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,extract_name])

        return img_label_pair_list

    def data_transform(self, img1,img2,lbl):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float64)
        img1 -= cfg.T0_MEAN_VALUE
        img1 = img1.transpose(2, 0, 1)
        img1 = torch.from_numpy(img1).float()
        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float64)
        img2 -= cfg.T1_MEAN_VALUE
        img2 = img2.transpose(2, 0, 1)
        img2 = torch.from_numpy(img2).float()
        lbl = torch.from_numpy(lbl).long()
        #lbl_reverse = torch.from_numpy(lbl_reverse).long()
        return img1,img2,lbl

    def __getitem__(self, index):

        img1_path,img2_path,label_path,filename = self.img_label_path_pairs[index]
        ####### load images #############
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        height,width,_ = np.array(img1,dtype= np.uint8).shape
        if self.transform_med != None:
           img1 = self.transform_med(img1)
           img2 = self.transform_med(img2)
        img1 = np.array(img1,dtype= np.uint8)
        img2 = np.array(img2,dtype= np.uint8)
        ####### load labels ############
        if self.flag == 'train':

            label = Image.open(label_path)
            if self.transform_med != None:
                label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)

        if self.flag == 'val':
            label = Image.open(label_path)
            if self.transform_med != None:
               label = self.transform_med(label)
            label = np.array(label,dtype=np.int32)

        if self.transform:
            img1,img2,label = self.data_transform(img1,img2,label)

        return img1,img2,label,str(filename),int(height),int(width)

    def __len__(self):

        return len(self.img_label_path_pairs)

