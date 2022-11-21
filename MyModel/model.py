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
    def declare_networks(self):
        self.G = MyGenerator().cuda()
        self.D = ProjectedDiscriminator().cuda()
        
        self.set_networks_train_mode()

    def set_networks_train_mode(self):
        self.G.train()
        self.D.train()
        self.D.feature_network.eval()
        self.D.feature_network.requires_grad_(False)
        
    def set_networks_test_mode(self):
        self.G.eval()
        self.D.eval()

    def go_step(self):
        source_color, source_gray, source_mask, target_color, target_gray, target_mask = self.load_next_batch(self.train_dataloader, self.train_iterator, 'train')
        
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
        # with torch.no_grad():
        run_dict['blend_img'], run_dict['color_map'] = self.G([run_dict['source_color'], run_dict['source_mask'], run_dict['target_color'], run_dict['target_gray'], run_dict['target_mask']])
        bg_mask = run_dict["target_mask"][:,0].unsqueeze(1)
        run_dict["fake_img"] = run_dict['blend_img'] * (1-bg_mask) + run_dict["target_color"] * bg_mask

        run_dict['cycle_blend_img'], run_dict['cycle_color_map'] = self.G([run_dict['fake_img'], run_dict['target_mask'], run_dict['source_color'], run_dict['source_gray'], run_dict['source_mask']])
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

    def do_validation(self):
        self.valid_images = []
        self.set_networks_test_mode()
        
        pbar = tqdm(range(len(self.valid_dataloader)), desc='Run validate..')
        for _ in pbar:
            source_color, source_gray, source_mask, target_color, target_gray, target_mask = self.load_next_batch(self.valid_dataloader, self.valid_iterator, 'valid')
            
            self.valid_dict["source_color"] = source_color
            self.valid_dict["source_gray"] = source_gray
            self.valid_dict["source_mask"] = source_mask
            self.valid_dict["target_color"] = target_color
            self.valid_dict["target_gray"] = target_gray
            self.valid_dict["target_mask"] = target_mask

            with torch.no_grad():
                self.run_G(self.valid_dict)
                self.run_D(self.valid_dict)
                
                self.loss_collector.loss_dict["valid_L_G"] += (self.loss_collector.get_loss_G(self.valid_dict, valid=True) / len(self.valid_dataloader))
                self.loss_collector.loss_dict["valid_L_D"] += (self.loss_collector.get_loss_D(self.valid_dict, valid=True) / len(self.valid_dataloader))   
            
            if len(self.valid_images) < 8 : utils.stack_image_grid([self.valid_dict["source_color"], self.valid_dict["target_color"], self.valid_dict["color_map"], self.valid_dict["fake_img"]], self.valid_images)
            
        self.loss_collector.val_print_loss()
        
        self.valid_images = torch.cat(self.valid_images, dim=-1)

        self.set_networks_train_mode()
        
    def do_test(self):
        self.test_images = []
        self.set_networks_test_mode()
        
        pbar = tqdm(range(len(self.test_dataloader)), desc='Run test...')
        for _ in pbar:
            source_color, source_gray, source_mask, target_color, target_gray, target_mask = self.load_next_batch(self.test_dataloader, self.test_iterator, 'test')
            
            self.test_dict["source_color"] = source_color
            self.test_dict["source_gray"] = source_gray
            self.test_dict["source_mask"] = source_mask
            self.test_dict["target_color"] = target_color
            self.test_dict["target_gray"] = target_gray
            self.test_dict["target_mask"] = target_mask

            with torch.no_grad():
                self.run_G(self.test_dict)
                self.run_D(self.test_dict)
                             
            utils.stack_image_grid([self.test_dict["source_color"], self.test_dict["target_color"], self.test_dict["color_map"], self.test_dict["fake_img"]], self.test_images)
        
        self.test_images = torch.cat(self.test_images, dim=-1)

        self.set_networks_train_mode()

    @property
    def loss_collector(self):
        return self._loss_collector


    def set_loss_collector(self):
        self._loss_collector = MyModelLoss(self.CONFIG)        
