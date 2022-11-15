import abc
import torch
from torch.utils.data import DataLoader
from MyModel.dataset import MyDataset
from lib import utils
import numpy as np
# from packages import Ranger

class ModelInterface(metaclass=abc.ABCMeta):
    """
    Base class for face GAN models. This base class can also be used 
    for neural network models with different purposes if some of concrete methods 
    are overrided appropriately. Exceptions will be raised when subclass is being 
    instantiated but abstract methods were not implemented. 
    """

    def __init__(self, CONFIG):
        """
        When overrided, super call is required.
        """
        self.CONFIG = CONFIG
        self.train_dict = {}
        self.valid_dict = {}
        self.SetupModel()

    def SetupModel(self):
        
        self.CONFIG['BASE']['IS_MASTER'] = self.CONFIG['BASE']['GPU_ID'] == 0
        self.RandomGenerator = np.random.RandomState(42)
        self.set_networks()
        self.set_optimizers()

        if self.CONFIG['BASE']['USE_MULTI_GPU']:
            self.set_multi_GPU()

        if self.CONFIG['CKPT']['TURN_ON']:
            self.load_checkpoint()

        self.set_dataset()
        self.set_data_iterator()
        self.set_validation()
        self.set_loss_collector()

        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"Model {self.CONFIG['BASE']['MODEL_ID']} has successively created")

    def load_next_batch(self):
        """
        Load next batch of source image, target image, and boolean values that denote 
        if source and target are identical.
        """
        try:
            batch_data = next(self.train_iterator)
            batch_data = [data.cuda() for data in batch_data]

        except StopIteration:
            self.train_iterator = iter(self.train_dataloader)
            batch_data = next(self.train_iterator)
            batch_data = [data.cuda() for data in batch_data]

        return batch_data

    def set_dataset(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = MyDataset(self.CONFIG)
        if self.CONFIG['BASE']['DO_VALID']:
            self.valid_dataset = MyDataset(self.CONFIG)

    def set_data_iterator(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.CONFIG['BASE']['USE_MULTI_GPU'] else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.CONFIG['BASE']['BATCH_PER_GPU'], pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
        self.train_iterator = iter(self.train_dataloader)

    def set_validation(self):
        """
        Predefine test images only if args.valid_dataset_root is specified.
        These images are anchored for checking the improvement of the model.
        """
        if self.CONFIG['BASE']['DO_VALID'] :
            sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset) if self.CONFIG['BASE']['USE_MULTI_GPU'] else None
            self.valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.CONFIG['BASE']['BATCH_PER_GPU'], pin_memory=True, sampler=sampler, drop_last=True)
            self.valid_iterator = iter(self.valid_dataloader)

    @abc.abstractmethod
    def set_networks(self):
        """
        Construct networks, send it to GPU, and set training mode.
        Networks should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train() 
        """
        pass

    def set_multi_GPU(self):
        utils.setup_ddp(self.CONFIG['BASE']['GPU_ID'], self.CONFIG['BASE']['GPU_NUM'])

        # Data parallelism is required to use multi-GPU
        self.G = torch.nn.parallel.DistributedDataParallel(self.G, device_ids=[self.CONFIG['BASE']['GPU_ID']], broadcast_buffers=False, find_unused_parameters=True).module
        self.D = torch.nn.parallel.DistributedDataParallel(self.D, device_ids=[self.CONFIG['BASE']['GPU_ID']]).module

    def save_checkpoint(self):
        """
        Save model and optimizer parameters.
        """
        utils.save_checkpoint(self.CONFIG, self.G, self.opt_G, type='G')
        utils.save_checkpoint(self.CONFIG, self.D, self.opt_D, type='D')
        
        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"\nCheckpoints are succesively saved in {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['BASE']['RUN_ID']}/ckpt/\n")
    
    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """

        self.CONFIG['BASE']['GLOBAL_STEP'] = \
        utils.load_checkpoint(self.CONFIG, self.G, self.opt_G, type="G")
        utils.load_checkpoint(self.CONFIG, self.D, self.opt_D, type="D")

        if self.CONFIG['BASE']['IS_MASTER']:
            print(f"Pretrained parameters are succesively loaded from {CONFIG['BASE']['SAVE_ROOT']}/{CONFIG['CKPT']['ID']}/ckpt/")

    def set_optimizers(self):
        if self.CONFIG['OPTIMIZER']['TYPE'] == "Adam":
            self.opt_G = torch.optim.Adam(self.G.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_G'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            self.opt_D = torch.optim.Adam(self.D.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_D'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            
        if self.CONFIG['OPTIMIZER']['TYPE'] == "Ranger":
            self.opt_G = Ranger(self.G.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_G'], betas=self.CONFIG['OPTIMIZER']['BETA'])
            self.opt_D = Ranger(self.D.parameters(), lr=self.CONFIG['OPTIMIZER']['LR_D'], betas=self.CONFIG['OPTIMIZER']['BETA'])

    @abc.abstractmethod
    def set_loss_collector(self):
        """
        Set self.loss_collector as an implementation of lib.loss.LossInterface.
        """
        pass

    @property
    @abc.abstractmethod
    def loss_collector(self):
        """
        loss_collector should be an implementation of lib.loss.LossInterface.
        This property should be assigned in self.set_loss_collector.
        """
        pass

    @abc.abstractmethod
    def go_step(self):
        """
        Implement a single iteration of training. This will be called repeatedly in a loop. 
        This method should return list of images that was created during training.
        Returned images are passed to self.save_image and self.save_image is called in the 
        training loop preiodically.
        """
        pass

    @abc.abstractmethod
    def do_validation(self):
        """
        Test the model using a predefined valid set.
        This method includes util.save_image and returns nothing.
        """
        pass

    