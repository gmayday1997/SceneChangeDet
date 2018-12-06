# Fully Convolutional Siamese Network for Scene Change Detection

Pytorch implementation of Scene Change Detection as described in [Learning to Measure Change: Fully Convolutional Siamese Metric Networks for Scene Change Detection](https://arxiv.org/pdf/1810.09111.pdf). 

![img1](https://github.com/gmayday1997/SceneChangeDet/blob/master/img/fig1.png)

## Requirements

- Python2.7
- Pytorch0.2.0_3 (see: [pytorch installation instuctions](http://pytorch.org/))
- torchvision

## Datasets
This repo is built for scene change detection. We report the performance on three datasets.

- PCD2015 dataset
 - paper: [Change detection from a street image pair using cnn features and superpixel segmentation](http://www.vision.is.tohoku.ac.jp/files/9814/3947/4830/71-Sakurada-BMVC15.pdf)
 - dataset: http://www.vision.is.tohoku.ac.jp/us/research/4d_city_modeling/pano_cd_dataset/
 
- VL_CMU_CD dataset
 - paper: [Street-view change detection with deconvolutional networks](http://www.robesafe.com/personal/roberto.arroyo/docs/Alcantarilla16rss.pdf)
 - dataset: https://ghsi.github.io/proj/RSS2016.html

- CD2014 dataset
 - paper: [changedetection.net: A new change detection benchmark dataset](https://www.merl.com/publications/docs/TR2012-044.pdf)
 - dataset: http://changedetection.net/
 # 06/12/2018 update 
 We have uploaded the modified CD2014 dataset to [[baiduyun]](https://pan.baidu.com/s/19ReVH6pmizcU79sk2Rsz5w)
 if you find cd2014 dataset is useful for your research, please cite the paper:
 
    @inproceedings{Goyette2012changedetection,
      title={changedetection.net: A New Change Detection Benchmark Dataset},
      author={Goyette, Nil and Jodoin, Pierre Marc and Porikli, Fatih and Konrad, Janusz and Ishwar, Prakash},
      booktitle={Computer Vision and Pattern Recognition Workshops},
      pages={1-8},
      year={2012},
    }
 
### Directory Structure
 
File Structure is as follows:

```
$T0_image_path/*.jpg
$T1_image_path/*.jpg
$ground_truth_path/*.jpg
```

## Pretrained Model
Backbone model, which is deeplabv2 [[baiduyun]](https://pan.baidu.com/s/1Ie8h1Lyzqn2g3GHcGxnppg) [[googledriver]](https://drive.google.com/file/d/1vma3tTX_ecKvInd91CWMEivbxhT5Xjfa/view?usp=sharing)in our work, is available, you should download it and put it to `/pretrain`

Pretrained models for PCD2015 and VL_CMU_CD also have been available.

- PCD2015: [[baiduyun]](https://pan.baidu.com/s/1kNNpRlQZJA45wOf0fJtaxw)
           [[googledrive]](https://drive.google.com/file/d/18evxU0Y4CMMe_xBtQu3kj3RAI91ZatE1/view?usp=sharing)
- VL_CMU_CD: [[baiduyun]](https://pan.baidu.com/s/1ZOo3pbJ1hQvx3dSMWXTs-w)
             [[googledrive]](https://drive.google.com/file/d/1z2lwbbxhAEvm8w0S55qebp7Q2DokkNG7/view?usp=sharing)

## Training
```shell
cd $SCD_ROOT
python train.py
```
Please consider citing this paper, if you find this repo is useful in your research   :

    @article{guo2018learning,
        title={Learning to Measure Change: Fully Convolutional Siamese Metric Networks for Scene Change Detection},
        author={Guo, Enqiang and Fu, Xinsha and Zhu, Jiawei and Deng, Min and Liu, Yu and Zhu, Qing and Li, Haifeng},
        journal={arXiv preprint arXiv:1810.09111},
        year={2018}
    }

