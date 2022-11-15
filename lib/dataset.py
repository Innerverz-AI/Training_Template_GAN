import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lib import utils
import torch.nn.functional as F
import numpy as np
import torch
import cv2

# color, gray, mask
class DatasetInterface(Dataset):
    def __init__(self, CONFIG):
        
        self.img_size = CONFIG['BASE']['IMG_SIZE']
        self.set_tf()

    def pp_image(self, img_path, grayscale=False, flip=False):

        img = Image.open(img_path)

        # FLIP
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # RGB or GRAYSCALE
        if grayscale:
            img = img.convert("L")
            img = self.tf_gray(img)

        else: 
            img = img.convert("RGB")
            img = self.tf_color(img)

        return img

    def pp_label(self, label_path, flip=False):

        mask = Image.open(label_path).resize((self.img_size, self.img_size), Image.NEAREST)

        if flip:
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        mask = label_converter(mask)
        mask_one_hot = to_one_hot(mask)

        return mask_one_hot

    def set_tf(self):

        self.tf_gray = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.tf_color = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):    
        pass

    def __len__(self):
        pass


def label_converter(before_label):
    _before_label = np.array(before_label)
    canvas = np.zeros_like(_before_label)
    for idx in face_parsing_converter:
        canvas = np.where(_before_label==idx, face_parsing_converter[idx], canvas)
    return canvas

def to_one_hot(Xt):
    Xt_ = torch.tensor(Xt, dtype=torch.int64)
    c = 13
    h,w = Xt_.size()

    Xt_ = torch.reshape(Xt_,(1,1,h,w))
    one_hot_Xt = torch.FloatTensor(1, c, h, w).zero_()
    one_hot_Xt_ = one_hot_Xt.scatter_(1, Xt_, 1.0)
    one_hot_Xt_ = F.interpolate(one_hot_Xt_,(h,w),mode='nearest')
    return one_hot_Xt_.squeeze()

def vis_mask(one_hot_mask, name=None, grid_result=False):
    os.makedirs(f'./result/{name}', exist_ok=True)
    c, _, _ = one_hot_mask.shape

    grid = []
    for idx in range(c):
        one_ch = one_hot_mask[idx]
        _one_ch = np.array(one_ch)
        grid.append(_one_ch * 255)
        # cv2.imwrite(f'./result/{name}/{idx}.png', _one_ch * 255)
    if grid_result: cv2.imwrite(f'./result/{name}/grid.png', np.concatenate(grid,axis=-1))

face_parsing_converter = {
    0 : 0,
    1 : 1,
    2 : 2,
    3 : 3,
    4 : 4,
    5 : 5,
    6 : 0,
    7 : 1,
    8 : 1,
    9 : 0,
    10 : 1,
    11 : 6,
    12 : 7,
    13 : 8,
    14 : 1,
    15 : 0,
    16 : 0,
    17 : 9,
    18 : 0
}
