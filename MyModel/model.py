from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib import utils
from lib.model_interface import ModelInterface
from MyModel.loss import MyModelLoss
from MyModel.nets import Generator
from packages import ProjectedDiscriminator

class MyModel(ModelInterface):
    def set_networks(self):
        self.G = Generator().cuda(self.gpu).train()
        self.D = ProjectedDiscriminator().cuda(self.gpu).train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)

    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.args)

    def go_step(self, global_step):
        animated, target_flip, target, animated_one_hot, target_one_hot, target_one_hot_flip  = self.load_next_batch()
        
        self.dict["animated"] = animated
        self.dict["target_flip"] = target_flip
        self.dict["target"] = target
        self.dict["animated_one_hot"] = animated_one_hot
        self.dict["target_one_hot_flip"] = target_one_hot_flip
        self.dict["target_one_hot"] = target_one_hot

        # run G
        self.run_G(self.dict)

        # update G
        loss_G = self.loss_collector.get_loss_G(self.dict)
        utils.update_net(self.args, self.G, self.opt_G, loss_G)

        # run D
        self.run_D(self.dict)

        # update D
        loss_D = self.loss_collector.get_loss_D(self.dict)
        utils.update_net(self.args, self.D, self.opt_D, loss_D)
        
        # print images
        self.train_images = [
            self.dict["target_flip"],
            self.dict["animated"],
            self.dict["color_reference"],
            self.dict["fake_img"],
            ]

    def run_G(self, dict):
        color_reference = self.G.get_color_reference(dict["animated"], dict["target_flip"], dict["animated_one_hot"], dict["target_one_hot_flip"])
        blend_image = self.G.get_blend_image(dict["animated"], color_reference)

        bg_mask = dict["target_one_hot"][:,0].unsqueeze(1)

        fake_img = blend_image * (1 - bg_mask) + dict["target"] * bg_mask

        ## for Projected D
        g_pred_fake, feat_fake  = self.D(fake_img, None)

        feat_real  = self.D.get_feature(dict["target"])

        dict['color_reference'] = color_reference
        dict['blend_img'] = blend_image
        dict['fake_img'] = fake_img
        
        dict['g_feat_fake'] = feat_fake
        dict['g_feat_real'] = feat_real
        dict["g_pred_fake"] = g_pred_fake


    def run_D(self, dict):
        d_pred_fake, feat_fake  = self.D(dict['fake_img'].detach(), None)
        d_pred_real, feat_real  = self.D(dict['target'], None)
        
        dict["d_pred_fake"] = d_pred_fake
        dict["d_pred_real"] = d_pred_real
        dict['d_feat_fake'] = feat_fake
        dict['d_feat_real'] = feat_real

    def do_validation(self, step):
        self.loss_collector.loss_dict["valid_L_G"] = .0
        self.loss_collector.loss_dict["valid_L_D"] = .0
        self.G.eval()
        self.D.eval()
        
        valid_L_G, valid_L_D = .0, .0
        pbar = tqdm(self.valid_dataloader, desc='Run validate...')
        for animated, target_flip, target, animated_one_hot, target_one_hot, target_one_hot_flip in pbar:
            animated, target_flip, target, animated_one_hot, target_one_hot, target_one_hot_flip = \
                animated.to(self.gpu), target_flip.to(self.gpu), target.to(self.gpu), animated_one_hot.to(self.gpu), target_one_hot.to(self.gpu), target_one_hot_flip.to(self.gpu)

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
