import glob
import random
from lib import utils
from lib.dataset import DatasetInterface
from torchvision import transforms


class MyDataset(DatasetInterface):
    def __init__(self, CONFIG, mode, dataset_path_list):
        super(MyDataset, self).__init__(CONFIG)
        self.mode = mode
        self.set_tf()

        self.same_prob = CONFIG['BASE']['SAME_PROB']
        
        for name, list in dataset_path_list.items() : self.__setattr__(name, list)
        
        if CONFIG['BASE']['IS_MASTER']:
            print(f"Dataset of {self.__len__()} images constructed for the {self.mode}.")

    def __getitem__(self, index):
        
        # you can use random.choice(paths) or random.sample(paths, num)
        source_color = self.pp_image(self.image_path_list[index])
        source_gray = self.pp_image(self.image_path_list[index], grayscale=True)
        source_mask = self.pp_label(self.mask_path_list[index])
        
        # random_index = self.get_random_index() if random.random() < self.same_prob else index
        random_index = self.get_random_index()
        target_color = self.pp_image(self.image_path_list[random_index])
        target_gray = self.pp_image(self.image_path_list[random_index], grayscale=True)
        target_mask = self.pp_label(self.mask_path_list[random_index])

        # target_flip = self.pp_image(self.image_path_list[random_index], flip=True)
        # target_flip_mask = self.pp_label(self.mask_path_list[random_index], flip=True)

        return [source_color, source_gray, source_mask, target_color, target_gray, target_mask]

    def __len__(self):
        return len(self.image_path_list)

    # override
    def set_tf(self):

        self.tf_gray = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        
        self.tf_color = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def divide_datasets(model, CONFIG):
    image_path_list = utils.get_all_images(CONFIG['DATASET']['TRAIN_PATH']['IMAGE'])
    mask_path_list = utils.get_all_images(CONFIG['DATASET']['TRAIN_PATH']['MASK'])
    
    if CONFIG['BASE']['DO_VALID']:
        model.train_dataset_dict = {
                'image_path_list' : image_path_list[ : -1 * CONFIG['DATASET']['VAL_SIZE']],
                'mask_path_list' : mask_path_list[ : -1 * CONFIG['DATASET']['VAL_SIZE']]
            }  
        model.valid_dataset_dict = {
                'image_path_list' : image_path_list[-1 * CONFIG['DATASET']['VAL_SIZE'] : ],
                'mask_path_list' : mask_path_list[-1 * CONFIG['DATASET']['VAL_SIZE'] : ]
            }

    else:
        model.train_dataset_dict = {
                'image_path_list' : image_path_list[ : ],
                'mask_path_list' : mask_path_list[ : ]
            }
    
    if CONFIG['BASE']['DO_TEST']:
        image_path_list = utils.get_all_images(CONFIG['DATASET']['TEST_PATH']['IMAGE'])
        mask_path_list = utils.get_all_images(CONFIG['DATASET']['TEST_PATH']['MASK'])
        model.test_dataset_dict = {
            'image_path_list' : image_path_list,
            'mask_path_list' : mask_path_list
        }
