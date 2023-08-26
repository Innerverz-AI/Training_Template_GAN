import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.model import ModelInterface
from lib.discriminators import ProjectedDiscriminator
from core.nets import MyGenerator


class MyModel(ModelInterface):
    def __init__(self, CONFIG, accelerator):
        super(MyModel, self).__init__(CONFIG, accelerator)

        # from innerverz import IdExtractor
        # self.IE = IdExtractor()

        self.saving_data_names = ["source", "GT", "output"]

        self.set_networks_train_mode()

    def declare_networks(self):
        self.G = MyGenerator().cuda()
        self.D = ProjectedDiscriminator().cuda()

    def go_step(self):
        self.train_dict = self.load_next_batch(phase="train")

        self.run_G(self.train_dict)
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        self.update_net(self.opt_G, loss_G)

        self.run_D(self.train_dict)
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        self.update_net(self.opt_D, loss_D)

    def run_G(self, run_dict):
        with torch.no_grad():
            feat_real = self.D.get_feature(run_dict["source"])

        run_dict["output"] = self.G(run_dict["source"])
        g_pred_fake, feat_fake = self.D(run_dict["output"], None)

        run_dict["g_feat_fake"] = feat_fake
        run_dict["g_feat_real"] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_real, _ = self.D(run_dict["source"], None)
        d_pred_fake, _ = self.D(run_dict["output"].detach(), None)

        run_dict["d_pred_real"] = d_pred_real
        run_dict["d_pred_fake"] = d_pred_fake
