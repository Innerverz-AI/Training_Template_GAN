from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils
from lib.model import ModelInterface
from lib.discriminators import ProjectedDiscriminator
from MyModel.loss import MyModelLoss
from MyModel.nets import MyGenerator

class MyModel(ModelInterface):
    def set_networks(self):
        self.G = MyGenerator().cuda().train()
        self.D = ProjectedDiscriminator().cuda().train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)

    def go_step(self):
        source_color, source_gray, source_mask, target_color, target_gray, target_mask = self.load_next_batch()
        
        self.train_dict["source_color"] = source_color
        self.train_dict["source_gray"] = source_gray
        self.train_dict["source_mask"] = source_mask
        self.train_dict["target_color"] = target_color
        self.train_dict["target_gray"] = target_gray
        self.train_dict["target_mask"] = target_mask

        # run G
        self.run_G(self.train_dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.train_dict)
        utils.update_net(self.G, self.opt_G, loss_G, self.CONFIG['BASE']['USE_MULTI_GPU'])

        # run D
        self.run_D(self.train_dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.train_dict)
        utils.update_net(self.D, self.opt_D, loss_D, self.CONFIG['BASE']['USE_MULTI_GPU'])
        
        # print images
        self.train_images = [
            self.train_dict["source_color"],
            self.train_dict["source_gray"],
            self.train_dict["target_color"],
            self.train_dict["target_gray"],
            self.train_dict["color_map"],
            self.train_dict["fake_img"],
            self.train_dict["cycle_color_map"],
            self.train_dict["cycle_fake_img"],
            ]

    def run_G(self, run_dict):
        run_dict['blend_img'], run_dict['color_map'] = self.G([run_dict['source_color'], run_dict['source_gray'], run_dict['source_mask'], run_dict['target_color'], run_dict['target_gray'], run_dict['target_mask']])
        bg_mask = run_dict["target_mask"][:,0].unsqueeze(1)
        run_dict["fake_img"] = run_dict['blend_img'] * (1-bg_mask) + run_dict["target_color"] * bg_mask

        run_dict['cycle_blend_img'], run_dict['cycle_color_map'] = self.G([run_dict['fake_img'], run_dict['target_gray'], run_dict['target_mask'], run_dict['source_color'], run_dict['source_gray'], run_dict['source_mask']])
        bg_mask = run_dict["source_mask"][:,0].unsqueeze(1)
        run_dict["cycle_fake_img"] = run_dict['cycle_blend_img'] * (1-bg_mask) + run_dict["source_color"] * bg_mask

        g_pred_fake, feat_fake = self.D(run_dict["cycle_fake_img"], None)
        feat_real = self.D.get_feature(run_dict["source_color"])

        run_dict['g_feat_fake'] = feat_fake
        run_dict['g_feat_real'] = feat_real
        run_dict["g_pred_fake"] = g_pred_fake

    def run_D(self, run_dict):
        d_pred_fake, _  = self.D(run_dict['fake_img'].detach(), None)
        d_pred_real, _  = self.D(run_dict['target_color'], None)
        
        run_dict["d_pred_fake"] = d_pred_fake
        run_dict["d_pred_real"] = d_pred_real

    def do_validation(self, step):
        self.loss_collector.loss_dict["valid_L_G"] = .0
        self.loss_collector.loss_dict["valid_L_D"] = .0
        self.G.eval()
        self.D.eval()
        
        valid_L_G, valid_L_D = .0, .0
        pbar = tqdm(self.valid_dataloader, desc='Run validate...')
        for batch_data in pbar:
            animated, target_flip, target, animated_one_hot, target_one_hot, target_one_hot_flip = [data.cuda() for data in batch_data]

            self.valid_dict["animated"] = animated
            self.valid_dict["target_flip"] = target_flip
            self.valid_dict["target"] = target
            self.valid_dict["animated_one_hot"] = animated_one_hot
            self.valid_dict["target_one_hot_flip"] = target_one_hot_flip
            self.valid_dict["target_one_hot"] = target_one_hot

            with torch.no_grad():
                self.run_G(self.valid_dict)
                valid_L_G += self.loss_collector.get_loss_G(self.valid_dict, valid=True)
                self.run_D(self.valid_dict)
                valid_L_D += self.loss_collector.get_loss_D(self.valid_dict, valid=True)

        self.loss_collector.loss_dict["valid_L_G"] = valid_L_G / len(self.valid_dataloader)
        self.loss_collector.loss_dict["valid_L_D"] = valid_L_D / len(self.valid_dataloader)

        self.loss_collector.val_print_loss(step)

        self.G.train()
        self.D.train()

        # save last validated images
        self.valid_images = [
            self.valid_dict["target_flip"],
            self.valid_dict["animated"],
            self.valid_dict["color_reference"],
            self.valid_dict["fake_img"],
        ]

    @property
    def loss_collector(self):
        return self._loss_collector
