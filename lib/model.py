import abc
import torch
import random
from torch.utils.data import DataLoader
from core.dataset import divide_datasets, MyDataset
from lib import utils
import numpy as np
from core.loss import MyModelLoss
from tqdm import tqdm
import glob
import cv2
import os


class ModelInterface(metaclass=abc.ABCMeta):
    """
    Base class for face GAN models. This base class can also be used
    for neural network models with different purposes if some of concrete methods
    are overrided appropriately. Exceptions will be raised when subclass is being
    instantiated but abstract methods were not implemented.
    """

    def __init__(self, CONFIG, accelerator):
        """
        When overrided, super call is required.
        """

        self.accelerator = accelerator
        self.G = None
        self.D = None
        self.CONFIG = CONFIG

        self.declare_networks()
        self.set_optimizers()

        if self.CONFIG["CKPT"]["TURN_ON"]:
            self.load_checkpoint()

        divide_datasets(self, self.CONFIG)
        self.set_datasets()
        self.set_dataloaders()
        self.loss_collector = MyModelLoss(self.CONFIG)

        (
            self.G,
            self.D,
            self.opt_G,
            self.opt_D,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.G,
            self.D,
            self.opt_G,
            self.opt_D,
            self.train_dataloader,
            self.valid_dataloader,
        )

        if (
            accelerator.distributed_type == "MULTI_GPU"
        ):  # accelerator.distributed_type is "No" when signle GPU Training
            self.G = self.G.module
            self.D = self.D.module

        if self.accelerator.is_main_process:
            print(f"Model {self.CONFIG['BASE']['MODEL_ID']} has successively created")

    def update_net(self, optimizer, loss):
        optimizer.zero_grad()
        self.accelerator.backward(loss)
        optimizer.step()

    def load_next_batch(self, phase):
        """
        Load next batch of source image, target image, and boolean values that denote
        if source and target are identical.
        """

        # dataloader = self.__getattribute__(phase + "_dataloader")
        # self.__setattr__(phase + "_iterator", iter(dataloader))
        # iterator = self.__getattribute__(phase + "_iterator")

        try:
            batch_data = next(self.__getattribute__(phase + "_iterator"))

        except:
            self.__setattr__(
                phase + "_iterator", iter(self.__getattribute__(phase + "_dataloader"))
            )
            batch_data = next(self.__getattribute__(phase + "_iterator"))

        for key in batch_data.keys():
            batch_data[key] = batch_data[key].cuda()

        return batch_data

    def set_datasets(self):
        """
        Initialize dataset using the dataset paths specified in the command line arguments.
        """
        self.train_dataset = MyDataset(self.CONFIG, self.train_dataset_dict)
        self.valid_dataset = MyDataset(self.CONFIG, self.valid_dataset_dict)

        if self.accelerator.is_main_process:
            print(
                f"Dataset of {self.train_dataset.__len__()} images constructed for the TRAIN."
            )
            print(
                f"Dataset of {self.valid_dataset.__len__()} images constructed for the VALID."
            )

    def set_dataloaders(self):
        """
        Construct sampler according to number of GPUs it is utilizing.
        Using self.dataset and sampler, construct dataloader.
        Store Iterator from dataloader as a member variable.
        """
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.CONFIG["BASE"]["BATCH_PER_GPU"],
            pin_memory=True,
            num_workers=8,
            drop_last=True,
        )

        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            batch_size=self.CONFIG["BASE"]["BATCH_PER_GPU"],
            pin_memory=True,
            drop_last=True,
        )

    @abc.abstractmethod
    def declare_networks(self):
        """
        Construct networks, send it to GPU, and set training mode.
        Networks should be assigned to member variables.

        eg. self.D = Discriminator(input_nc=3).cuda(self.gpu).train()
        """
        pass

    def save_checkpoint(self):
        """
        Save model and optimizer parameters.
        """
        if self.accelerator.is_main_process:
            print(
                f"\nCheckpoints are succesively saved in {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['BASE']['RUN_ID']}/ckpt/\n"
            )

        ckpt_dict = {}
        ckpt_dict["global_step"] = self.CONFIG["BASE"]["GLOBAL_STEP"]

        ckpt_dict["model"] = self.G.state_dict()
        ckpt_dict["optimizer"] = self.opt_G.state_dict()
        torch.save(
            ckpt_dict,
            f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/G_{str(self.CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}.pt",
        )  # max 99,999,999
        torch.save(ckpt_dict, f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/G_latest.pt")

        if self.D:
            ckpt_dict["model"] = self.D.state_dict()
            ckpt_dict["optimizer"] = self.opt_D.state_dict()
            torch.save(
                ckpt_dict,
                f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/D_{str(self.CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}.pt",
            )  # max 99,999,999
            torch.save(
                ckpt_dict, f"{self.CONFIG['BASE']['SAVE_ROOT_CKPT']}/D_latest.pt"
            )

    def load_checkpoint(self):
        """
        Load pretrained parameters from checkpoint to the initialized models.
        """
        FLAG = False
        for run_id in os.listdir("./train_results"):
            if int(run_id[:3]) == self.CONFIG["CKPT"]["ID_NUM"]:
                self.CONFIG["CKPT"]["ID"] = run_id
                FLAG = True

        assert FLAG, "ID_NUM is wrong"

        ckpt_step = (
            "latest"
            if self.CONFIG["CKPT"]["STEP"] is None
            else str(self.CONFIG["CKPT"]["STEP"]).zfill(8)
        )

        ckpt_path_G = f"{self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/G_{ckpt_step}.pt"
        ckpt_dict_G = torch.load(ckpt_path_G, map_location=torch.device("cuda"))
        self.G.load_state_dict(ckpt_dict_G["model"], strict=False)
        self.opt_G.load_state_dict(ckpt_dict_G["optimizer"])

        if self.D:
            ckpt_path_D = f"{self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/D_{ckpt_step}.pt"
            ckpt_dict_D = torch.load(ckpt_path_D, map_location=torch.device("cuda"))
            self.D.load_state_dict(ckpt_dict_D["model"], strict=False)
            self.opt_D.load_state_dict(ckpt_dict_D["optimizer"])

        self.CONFIG["BASE"]["GLOBAL_STEP"] = ckpt_dict_G["global_step"]

        if self.accelerator.is_main_process:
            print(
                f"Pretrained parameters are succesively loaded from {self.CONFIG['BASE']['SAVE_ROOT']}/{self.CONFIG['CKPT']['ID']}/ckpt/"
            )

    def set_optimizers(self):
        if self.CONFIG["OPTIMIZER"]["TYPE"] == "Adam":
            self.opt_G = torch.optim.Adam(
                self.G.parameters(),
                lr=self.CONFIG["OPTIMIZER"]["LR_G"],
                betas=self.CONFIG["OPTIMIZER"]["BETA"],
            )
            if self.D:
                self.opt_D = torch.optim.Adam(
                    self.D.parameters(),
                    lr=self.CONFIG["OPTIMIZER"]["LR_D"],
                    betas=self.CONFIG["OPTIMIZER"]["BETA"],
                )

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

    def do_validation(self):
        self.set_networks_eval_mode()

        self.loss_collector.loss_dict["valid_L_G"] = 0
        self.loss_collector.loss_dict["valid_L_D"] = 0

        pbar = tqdm(range(len(self.valid_dataloader)), desc="Run validate..")
        for valid_batch_idx in pbar:
            self.valid_dict = self.load_next_batch("valid")

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.run_D(self.valid_dict)
                self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                self.loss_collector.get_loss_D(self.valid_dict, valid=True)

            if valid_batch_idx == 0:
                self.save_grid_image(phase="valid")

        self.loss_collector.loss_dict["valid_L_G"] /= len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] /= len(self.valid_dataloader)
        self.loss_collector.val_print_loss()

        self.set_networks_train_mode()

    def save_grid_image(self, phase="train"):
        data_dict = self.train_dict if phase == "train" else self.valid_dict
        images = [data_dict[data_name] for data_name in self.saving_data_names]
        grid = utils.make_grid_image(images)[:, :, ::-1]

        cv2.imwrite(
            f"{self.CONFIG['BASE']['SAVE_ROOT_IMGS']}/{str(self.CONFIG['BASE']['GLOBAL_STEP']).zfill(8)}_{phase}.png",
            grid,
        )
        cv2.imwrite(
            f"{self.CONFIG['BASE']['SAVE_ROOT_IMGS']}/_latest_{phase}_result.png",
            grid,
        )

    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)

    def set_networks_eval_mode(self):
        self.G.eval()
        self.D.eval()
