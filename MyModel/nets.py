import torch
import torch.nn as nn

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x

class ResnetBlock_Adain(nn.Module):
    def __init__(self, dim, latent_size, padding_type, activation=nn.ReLU(True)):
        super(ResnetBlock_Adain, self).__init__()

        p = 0
        conv1 = []
        if padding_type == 'reflect':
            conv1 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv1 += [nn.Conv2d(dim, dim, kernel_size=3, padding = p), InstanceNorm()]
        self.conv1 = nn.Sequential(*conv1)
        self.style1 = ApplyStyle(latent_size, dim)
        self.act1 = activation

        p = 0
        conv2 = []
        if padding_type == 'reflect':
            conv2 += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2 += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv2 += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), InstanceNorm()]
        self.conv2 = nn.Sequential(*conv2)
        self.style2 = ApplyStyle(latent_size, dim)


    def forward(self, x, dlatents_in_slice):
        y = self.conv1(x)
        y = self.style1(y, dlatents_in_slice)
        y = self.act1(y)
        y = self.conv2(y)
        y = self.style2(y, dlatents_in_slice)
        out = x + y
        return out


class MyGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False,
                 norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(MyGenerator, self).__init__()
        
        self.ArcFace = ArcFace()
        self.FaceParser = FaceParser()

        activation = nn.ReLU(True)
        
        self.deep = deep
        
        self.first_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
                                         norm_layer(64), activation)
        ### downsample
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   norm_layer(128), activation)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   norm_layer(256), activation)
        self.down3 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   norm_layer(512), activation)
                                   
        if self.deep:
            self.down4 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                       norm_layer(512), activation)

        ### resnet blocks
        BN = []
        for i in range(n_blocks):
            BN += [
                ResnetBlock_Adain(512, latent_size=latent_size, padding_type=padding_type, activation=activation)]
        self.BottleNeck = nn.Sequential(*BN)

        if self.deep:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512), activation
            )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), activation
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), activation
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), activation
        )
        self.last_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, kernel_size=7, padding=0))

    def get_id(self, image):
        return self.ArcFace.get_id(image)

    def get_mask(self, image):
        return self.FaceParser.get_mask(image)

    def forward(self, source, target):
        id_source = self.get_id(source)
        mask = self.get_mask(target)
        x = target 

        skip1 = self.first_layer(x)
        skip2 = self.down1(skip1)
        skip3 = self.down2(skip2)
        if self.deep:
            skip4 = self.down3(skip3)
            x = self.down4(skip4)
        else:
            x = self.down3(skip3)
        bot = []
        bot.append(x)
        features = []
        for i in range(len(self.BottleNeck)):
            x = self.BottleNeck[i](x, id_source)
            bot.append(x)

        if self.deep:
            x = self.up4(x)
            features.append(x)
        x = self.up3(x)
        features.append(x)
        x = self.up2(x)
        features.append(x)
        x = self.up1(x)
        features.append(x)
        x = self.last_layer(x)
        # x = (x + 1) / 2
        x = x*mask+target*(1-mask)

        # return x, bot, features, dlatents
        return x, id_source

## PSP
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(math.log2(spatial))
        modules = []
        modules += [
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU()
            ]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
                ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x

class GradualStyleEncoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=64):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(50)
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
            )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    bottleneck_IR_SE(
                    bottleneck.in_channel,
                    bottleneck.depth,
                    bottleneck.stride
                    ))
        self.body = nn.Sequential(*modules)

        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        self.output_layer = nn.Conv2d(64, out_dim, 1, 1, 0, bias=False)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x): # b, 3, 256, 256
        x = self.input_layer(x) # b 64 256 256

        modulelist = list(self.body._modules.values())
        c0 = x
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 0:
                c1 = x
            elif i == 6:
                c2 = x
                
            elif i == 20:
                c3 = x
            elif i == 23:
                c4 = x
        
        p3 = self._upsample_add(self.latlayer1(c4), c3) # 256 32 32
        p2 = self._upsample_add(self.latlayer2(p3), c2) # 128 64 64
        p1 = self._upsample_add(self.latlayer3(p2), c1) # 64 128 128
        p0 = self._upsample_add(self.latlayer4(p1), c0) # 64 256 256
        out= self.output_layer(p0)

        return out # b 64 256 256


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.color_transfer_net = GradualStyleEncoder(3, 64)
        self.blender_net = GradualStyleEncoder(6, 3)

    def get_color_reference(self, gray_image, rgb_image, gray_label, rgb_label):
        # 256 
        gray_feature_map = self.color_transfer_net(gray_image)
        rgb_feature_map = self.color_transfer_net(rgb_image)

        # animated image with target color
        color_reference = self.do_RC(gray_feature_map, rgb_feature_map, rgb_image, gray_label, rgb_label)

        return color_reference 


    def get_blend_image(self, gray_image, color_reference):
        cat_data = torch.cat((color_reference,gray_image), dim=1)
        _cat_data = F.interpolate(cat_data,(256,256))
        blend_image = self.blender_net(_cat_data)

        return blend_image

    def do_RC(self, gray_feature_map, rgb_feature_map, rgb_image, gray_one_hot, rgb_one_hot):
        _, n_ch, _, _ = gray_one_hot.shape
        canvas = torch.zeros_like(rgb_image)
        b, c, h, w = gray_feature_map.size()
        for b_idx in range(b):
            for c_idx in range(1, n_ch):
                gray_mask, rgb_mask = gray_one_hot[b_idx,c_idx], rgb_one_hot[b_idx,c_idx]
                if gray_mask.sum() == 0:
                    continue

                gray_matrix = torch.masked_select(gray_feature_map[b_idx], gray_mask.bool()).reshape(c, -1) # 64, pixel_num_A
                gray_matrix_bar = gray_matrix - gray_matrix.mean(0, keepdim=True) # 64, pixel_num_A
                gray_matrix_norm = torch.norm(gray_matrix_bar, dim=0, keepdim=True)
                gray_matrix_ = gray_matrix_bar / gray_matrix_norm

                rgb_matrix = torch.masked_select(rgb_feature_map[b_idx], rgb_mask.bool()).reshape(c, -1) # 64, pixel_num_B
                rgb_matrix_bar = rgb_matrix - rgb_matrix.mean(0, keepdim=True) # 64, pixel_num_B
                rgb_matrix_norm = torch.norm(rgb_matrix_bar, dim=0, keepdim=True)
                rgb_matrix_ = rgb_matrix_bar / rgb_matrix_norm
               
                rgb_pixels = torch.masked_select(rgb_image[b_idx], rgb_mask.bool()).reshape(3,-1)
               
                correlation_matrix = torch.matmul(gray_matrix_.transpose(0,1), rgb_matrix_)
                correlation_matrix = F.softmax(correlation_matrix,dim=1)
                
                colorized_matrix = torch.matmul(correlation_matrix, rgb_pixels.transpose(0,1)).transpose(0,1)

                canvas[b_idx].masked_scatter_(gray_mask.bool(), colorized_matrix) # 3 128 128

        return canvas