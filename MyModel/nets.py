import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.nets import ConvBlock
from collections import namedtuple


class FPN(nn.Module):
    def __init__(self, input_nc=3):
        super(FPN, self).__init__()
        
        self.input_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 32, kernel_size=7))

        self.down1 = ConvBlock(32, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)
                           
        self.up4 = ConvBlock(512, 512, transpose=True)
        self.up3 = ConvBlock(512, 512, transpose=True)

        self.skip = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        x = self.input_layer(x) # [3, H, W] >> [32, H, W]

        x1 = self.down1(x) # [32, H, W] >> [64, H//2, W//2]
        x2 = self.down2(x1) # [64, H//2, W//2] >> [128, H//4, W//4]
        x3 = self.down3(x2) # [128, H//4, W//4] >> [256, H//8, W//8]
        x4 = self.down4(x3) # [256, H//8, W//8] >> [512, H//16, W//16]

        y = x4
        y = self.up4(y) # [512, H//16, W//16] >> [256, H//8, W//8]
        y = self.up3(y+self.skip(x3)) # [256, H//8, W//8] >> [128, H//4, W//4]

        return y


class Unet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super(Unet, self).__init__()
        
        self.input_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7))

        self.down1 = ConvBlock(64, 128)
        self.down2 = ConvBlock(128, 256)
        self.down3 = ConvBlock(256, 512)
                           
        self.up3 = ConvBlock(512, 256, transpose=True)
        self.up2 = ConvBlock(256, 128, transpose=True)
        self.up1 = ConvBlock(128, 64, transpose=True)

        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7))

    def forward(self, x):

        x = self.input_layer(x)

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)

        x = self.up3(x)
        x = self.up2(x)
        x = self.up1(x)

        x = self.output_layer(x)

        return torch.tanh(x)

class MyGenerator(nn.Module):
    def __init__(self):
        super(MyGenerator, self).__init__()
        self.feauture_net = FPN(3)
        self.blend_net = Unet(4, 3)

    def set_dict(self, input):
        self.net_dict = {}
        self.net_dict["source_color"] = input[0]
        self.net_dict["source_mask"] = input[1]
        self.net_dict["target_color"] = input[2]
        self.net_dict["target_gray"] = input[3]
        self.net_dict["target_mask"] = input[4]

    def forward(self, input):
        self.set_dict(input)

        self.get_feature()
        self.get_color_map()
        self.get_blend_img()

        return self.net_dict['blend_img'], self.net_dict['color_map']

    def get_feature(self):
        self.net_dict['source_feat'] = self.feauture_net(self.net_dict['source_color'])
        self.net_dict['target_feat'] = self.feauture_net(self.net_dict['target_color'])

    def get_blend_img(self):
        self.net_dict['blend_img'] = self.blend_net(torch.cat([self.net_dict['target_gray'], self.net_dict['color_map']], dim=1))

    def get_color_map(self):

        _, n_ch, _, _ = self.net_dict['target_mask'].shape
        b, c, h, w = self.net_dict['target_feat'].size()
        target_mask_quater = F.interpolate(self.net_dict['target_mask'], scale_factor=0.25, mode='nearest')
        source_mask_quater = F.interpolate(self.net_dict['source_mask'], scale_factor=0.25, mode='nearest')
        source_color_quater = F.interpolate(self.net_dict['source_color'], scale_factor=0.25, mode='bilinear')
        canvas = torch.zeros_like(source_color_quater)

        for b_idx in range(b):
            for c_idx in range(1, n_ch):
                target_1ch_mask, source_1ch_mask = target_mask_quater[b_idx,c_idx], source_mask_quater[b_idx,c_idx]
                
                if target_1ch_mask.sum() == 0 or target_1ch_mask.sum() == 1 or source_1ch_mask.sum() == 0 or source_1ch_mask.sum() == 1:
                    continue
            
                target_matrix = torch.masked_select(self.net_dict['target_feat'][b_idx], target_1ch_mask.bool()).reshape(c, -1) # 64, pixel_num_A
                target_matrix_bar = target_matrix - target_matrix.mean(1, keepdim=True) # 64, pixel_num_A
                target_matrix_norm = torch.norm(target_matrix_bar, dim=0, keepdim=True)
                target_matrix_ = target_matrix_bar / target_matrix_norm

                source_matrix = torch.masked_select(self.net_dict['source_feat'][b_idx], source_1ch_mask.bool()).reshape(c, -1) # 64, pixel_num_B
                source_matrix_bar = source_matrix - source_matrix.mean(1, keepdim=True) # 64, pixel_num_B
                source_matrix_norm = torch.norm(source_matrix_bar, dim=0, keepdim=True)
                source_matrix_ = source_matrix_bar / source_matrix_norm
               
                correlation_matrix = torch.matmul(target_matrix_.transpose(0,1), source_matrix_)
                correlation_matrix = F.softmax(correlation_matrix,dim=1)

                source_pixels = torch.masked_select(source_color_quater[b_idx], source_1ch_mask.bool()).reshape(3,-1)
                colorized_matrix = torch.matmul(correlation_matrix, source_pixels.transpose(0,1)).transpose(0,1)

                canvas[b_idx].masked_scatter_(target_1ch_mask.bool(), colorized_matrix) # 3 128 128

        self.net_dict['color_map'] = F.interpolate(canvas, scale_factor=4, mode='bilinear')