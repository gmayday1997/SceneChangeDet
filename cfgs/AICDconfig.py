import os

BASE_PATH = '/home/admin228/Projects/pytorch/cd_newbranch'
PRETRAIN_MODEL_PATH = os.path.join(BASE_PATH,'pretrain')
DATA_PATH = '/media/admin228/0007A0C30005763A/datasets/dataset_/AICDDataset'
TRAIN_DATA_PATH = os.path.join(DATA_PATH,'img_noshadow')
TRAIN_LABEL_PATH = os.path.join(DATA_PATH)
TRAIN_TXT_PATH = os.path.join(DATA_PATH,'train.txt')
VAL_DATA_PATH = os.path.join(DATA_PATH,'img_noshadow')
VAL_LABEL_PATH = os.path.join(DATA_PATH)
VAL_TXT_PATH = os.path.join(DATA_PATH,'test.txt')
SAVE_PATH = '/media/admin228/00027E210001A5BD/train_pytorch/change_detection/AICD'
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)
SAVE_CKPT_PATH = os.path.join(SAVE_PATH,'ckpt')
if not os.path.exists(SAVE_CKPT_PATH):
    os.mkdir(SAVE_CKPT_PATH)
SAVE_PRED_PATH = os.path.join(SAVE_PATH,'prediction')
if not os.path.exists(SAVE_PRED_PATH):
    os.mkdir(SAVE_PRED_PATH)
TRAINED_BEST_PERFORMANCE_CKPT = os.path.join(SAVE_CKPT_PATH,'model_best.pth')
INIT_LEARNING_RATE = 1e-7
DECAY = 5e-5
MOMENTUM = 0.90
MAX_ITER = 40000
BATCH_SIZE = 1
THRESH = 0.1
LOSS_PARAM = 30
TRANSFROM_SCALES= (512,512)
T0_MEAN_VALUE = (107.645,147.339,140.415)
T1_MEAN_VALUE = (109.706,150.060,142.971)
    

