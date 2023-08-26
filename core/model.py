import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils
from lib.model import ModelInterface
from lib.discriminators import ProjectedDiscriminator
from core.loss import MyModelLoss
from core.nets import MyGenerator

class MyModel(ModelInterface):
    def declare_networks(self):
        self.G = MyGenerator().cuda()
        self.D = ProjectedDiscriminator().cuda()
        
        self.set_networks_train_mode()

        # PACKAGES
        # from id_extractor import IdExtractor
        # self.IE = IdExtractor()

    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_networks_eval_mode(self):
        self.G.eval()
        self.D.eval()

    def go_step(self):
        self.batch_data_names = ['source', 'GT']
        self.saving_data_names = ['source', 'GT', 'output']
        
        batch_data_bundle = self.load_next_batch(self.train_dataloader, self.train_iterator, 'train')
        
        for data_name, batch_data in zip(self.batch_data_names, batch_data_bundle):
            self.train_dict[data_name] = batch_data

        self.run_G(self.train_dict)
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        self.update_net(self.opt_G, loss_G)

        self.run_D(self.train_dict)
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        self.update_net(self.opt_D, loss_D)
        
        # print images
        self.train_images = [self.train_dict[data_name] for data_name in self.saving_data_names]

    def run_G(self, run_dict):
        # with torch.no_grad():
        run_dict['output'] = self.G(run_dict['source'])
        g_pred_fake, feat_fake = self.D(run_dict["output"], None)
        feat_real = self.D.get_feature(run_dict["source"])

        run_dict['g_feat_fake'] = feat_fake
        run_dict['g_feat_real'] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_real, _  = self.D(run_dict['source'], None)
        d_pred_fake, _  = self.D(run_dict['output'].detach(), None)
        
        run_dict["d_pred_real"] = d_pred_real
        run_dict["d_pred_fake"] = d_pred_fake

    @property
    def loss_collector(self):
        return self._loss_collector

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)        
