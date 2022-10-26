import glob
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from lib import utils
import numpy as np
import torch
import cv2
from lib.utils_mask import label_converter, to_one_hot

class SingleFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster):
        self.image_path_list = glob.glob('/home/jjy/Datasets/celeba/train/images/*.*')
        self.label_path_list = glob.glob('/home/jjy/Datasets/celeba/train/label/*.*')
        self.img_size = 128
        self.transforms1 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):    
        Xs = Image.open(self.image_path_list[item]).convert("L")
        Xs_equ = cv2.equalizeHist(np.array(Xs))
        Xs_ = Image.fromarray(Xs_equ)
        Xs_mask = Image.open(self.label_path_list[item]).resize((self.img_size,self.img_size),Image.NEAREST)
        _Xs_mask = label_converter(Xs_mask)
        Xs_mask_one_hot = to_one_hot(_Xs_mask)
        
        Xt = Image.open(self.image_path_list[item]).convert("RGB")

        Xt_mask = Image.open(self.label_path_list[item]).resize((self.img_size,self.img_size),Image.NEAREST)
        _Xt_mask = label_converter(Xt_mask)
        Xt_mask_flip = Xt_mask.transpose(Image.FLIP_LEFT_RIGHT)
        _Xt_mask_flip = label_converter(Xt_mask_flip)

        Xt_mask_one_hot = to_one_hot(_Xt_mask)
        Xt_mask_one_hot_flip = to_one_hot(_Xt_mask_flip)

        return self.transforms1(Xs_).repeat(3,1,1), self.transforms2(Xt), self.transforms3(Xt), Xs_mask_one_hot, Xt_mask_one_hot, Xt_mask_one_hot_flip

    def __len__(self):
        return len(self.image_path_list)
    

class SingleFaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        self.image_path_list = glob.glob('/home/jjy/Datasets/celeba/valid/images/*.*')[:500]
        self.label_path_list = glob.glob('/home/jjy/Datasets/celeba/valid/label/*.*')[:500]
        self.img_size = 128
        self.transforms1 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        self.transforms2 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transforms3 = transforms.Compose([
            transforms.Resize((self.img_size,self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):    
        Xs = Image.open(self.image_path_list[item]).convert("L")
        Xs_equ = cv2.equalizeHist(np.array(Xs))
        Xs_ = Image.fromarray(Xs_equ)
        Xs_mask = Image.open(self.label_path_list[item]).resize((self.img_size,self.img_size),Image.NEAREST)
        _Xs_mask = label_converter(Xs_mask)
        Xs_mask_one_hot = to_one_hot(_Xs_mask)
        
        Xt = Image.open(self.image_path_list[item]).convert("RGB")

        Xt_mask = Image.open(self.label_path_list[item]).resize((self.img_size,self.img_size),Image.NEAREST)
        _Xt_mask = label_converter(Xt_mask)
        Xt_mask_flip = Xt_mask.transpose(Image.FLIP_LEFT_RIGHT)
        _Xt_mask_flip = label_converter(Xt_mask_flip)

        Xt_mask_one_hot = to_one_hot(_Xt_mask)
        Xt_mask_one_hot_flip = to_one_hot(_Xt_mask_flip)

        return self.transforms1(Xs_).repeat(3,1,1), self.transforms2(Xt), self.transforms3(Xt), Xs_mask_one_hot, Xt_mask_one_hot, Xt_mask_one_hot_flip

    def __len__(self):
        return len(self.image_path_list)

class PairedFaceDatasetTrain(Dataset):
    def __init__(self, dataset_root_list, isMaster, same_prob=0.2):
        self.same_prob = same_prob
        self.image_path_list, self.image_num_list = utils.get_all_images(dataset_root_list)
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the training.")

    def __getitem__(self, item):
        idx = 0
        while item >= self.image_num_list[idx]:
            item -= self.image_num_list[idx]
            idx += 1
        image_path = self.image_path_list[idx][item]
        
        Xs = Image.open(image_path).convert("RGB")

        if random.random() > self.same_prob:
            image_path = random.choice(self.image_path_list[random.randint(0, len(self.image_path_list)-1)])
            Xt = Image.open(image_path).convert("RGB")
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
        return self.transforms(Xs), self.transforms(Xt), same_person

    def __len__(self):
        return sum(self.image_num_list)


class PairedFaceDatasetValid(Dataset):
    def __init__(self, valid_data_dir, isMaster):
        
        self.source_path_list = sorted(glob.glob(f"{valid_data_dir}/source/*.*g"))
        self.target_path_list = sorted(glob.glob(f"{valid_data_dir}/target/*.*g"))

        # take the smaller number if two dirs have different numbers of images
        self.num = min(len(self.source_path_list), len(self.target_path_list))
        
        self.transforms = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        if isMaster:
            print(f"Dataset of {self.__len__()} images constructed for the validation.")

    def __getitem__(self, idx):
        
        Xs = Image.open(self.source_path_list[idx]).convert("RGB")
        Xt = Image.open(self.target_path_list[idx]).convert("RGB")

        return self.transforms(Xs), self.transforms(Xt)

    def __len__(self):
        return self.num


