from lib.loss_interface import Loss, LossInterface
import torch.nn.functional as F
import time
from lib.utils_loss import VGGLoss, GANLoss

class MyModelLoss(LossInterface):
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}

        # self.ganloss = GANLoss('hinge').cuda(self.args.gpu)
        self.vggloss = VGGLoss().cuda()

    def get_loss_G(self, dict, valid=False):
        L_G = 0.0
        if self.args.W_adv:
            # L_gan = self.ganloss(dict["g_pred_fake"],True,for_discriminator=False)
            L_adv = (-dict["g_pred_fake"]).mean()
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_g_adv"] = round(L_adv.item(), 4)

        if self.args.W_vgg:
            L_vgg = self.vggloss(F.interpolate(dict["fake_img"],(256,256),mode='bilinear'), F.interpolate(dict["target"],(256,256),mode='bilinear'))
            L_G += self.args.W_vgg * L_vgg
            self.loss_dict["L_vgg"] = round(L_vgg.item(), 4)
            
        # Reconstruction loss
        if self.args.W_recon:
            L_recon = Loss.get_L1_loss(dict["fake_img"], dict["target"])
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)

        # if self.args.W_cycle:
        #     L_cycle = Loss.get_L1_loss(dict["recon_target_flip"], dict["target_flip"])
        #     L_G += self.args.W_cycle * L_cycle
        #     self.loss_dict["L_cycle"] = round(L_cycle.item(), 4)
        
        # feat loss for Projected D
        if self.args.W_feat:
            L_feat = Loss.get_L1_loss(dict["g_feat_fake"]['3'], dict["g_feat_real"]['3'])
            L_G += self.args.W_feat * L_feat
            self.loss_dict["L_feat"] = round(L_feat.item(), 4)

        ## feat loss for Multilayer D
        # if self.args.W_feat:
        #     L_feat = .0
        #     for i in range(len(dict["g_pred_fake"])):
        #         for j in range(len(dict["g_pred_fake"][i])):
        #             L_feat += Loss.get_L1_loss(dict["g_pred_fake"][i][j], dict["g_pred_real"][i][j])
        #     L_G += self.args.W_feat * (L_feat / float(len(dict["g_pred_fake"])))
        #     self.loss_dict["L_feat"] = round(L_feat.item(), 4)
            
        if valid:
            self.loss_dict["valid_L_G"] += round(L_G.item(), 4)
        else:
            self.loss_dict["L_G"] = round(L_G.item(), 4)
        return L_G

    def get_loss_D(self, dict, valid=False):
        L_D = 0.0
        L_real = (F.relu(1 - dict["d_pred_real"])).mean()
        L_fake = (F.relu(1 + dict["d_pred_fake"])).mean()
        L_D = L_real + L_fake
        ## for GANLoss
        # L_fake = self.ganloss(dict["d_pred_fake"], False, for_discriminator=True)
        # L_real = self.ganloss(dict["d_pred_real"], True, for_discriminator=True)
        # L_D += (L_real + L_fake) / 2
        
        if valid:
            self.loss_dict["valid_L_D"] += round(L_D.item(), 4)
        else:
            self.loss_dict["L_real"] = round(L_real.item(), 4)
            self.loss_dict["L_fake"] = round(L_fake.item(), 4)
            self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
        