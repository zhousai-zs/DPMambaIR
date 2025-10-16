import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import sys
sys.path.append('/zs/DegAE_DegradationAutoencoder/codes/models/modules')
sys.path.append('/zs/DegAE_DegradationAutoencoder/codes/models')
sys.path.append('/zs/DegAE_DegradationAutoencoder/codes/')
sys.path.append('/zs/DegAE_DegradationAutoencoder/')
import models.modules.module_util as mutil
# import module_util as mutil
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import math

from modules.lk import dwconv_hwc, decouple_dwconv_hw_cdc

class preprocess_bf(nn.Module):
    def __init__(self, nf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, nf, 7, 1, 3, 1, 1, True)
        self.conv2 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv3 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv4 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv5 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv6 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv7 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv8 = nn.Conv2d(nf, nf, 7, 1, 3, 1, nf, True)

        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8], 0.1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = out1 + F.relu(self.conv2(out1))
        out3 = out2 + F.relu(self.conv3(out2))
        out4 = out3 + F.relu(self.conv4(out3))
        out5 = out4 + F.relu(self.conv5(out4))
        out6 = out5 + F.relu(self.conv6(out5))
        out7 = out6 + F.relu(self.conv7(out6))
        out8 = out7 + F.relu(self.conv8(out7))
        return out8

class DeepDegradationEncoder_bfv2(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_bfv2, self).__init__()
        
        self.DDR_extractor = preprocess_bf(nf)        
        self.conv1 = nn.Conv2d(nf, nf*2, 4, 2, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf*4, 4, 2, 0, bias=True)
        self.conv3 = nn.Conv2d(nf*4, nf*8, 4, 2, 0, bias=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.conv1x1_2 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
             
    def forward(self, x):
        DeepDegradationRepre = self.DDR_extractor(x)
        out = self.lrelu(self.conv1(DeepDegradationRepre)) # in: [B, 64, H, W]  out: [B, 128, H/2, W/2]
        out = self.lrelu(self.conv2(out)) # in: [B, 128, H/2, W/2]  out: [B, 256, H/4, W/4]
        out = self.lrelu(self.conv3(out)) # in: [B, 256, H/4, W/4]  out: [B, 512, H/8, W/8]
        out = self.adapool(out) # in: [B, 512, H/8, W/8]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_1(out)) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.conv1x1_2(out) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        
        DDR_embedding = out
        return DDR_embedding

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class MSRResNet_wGR_i_fea_backup(nn.Module):
    ''' modified SRResNet'''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=2):
        super(MSRResNet_wGR_i_fea_backup, self).__init__()
        print('Model: MSRResNet_wGR_i (return feature)')
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk2 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk3 = mutil.make_layer(basic_block, nb//4)
        # self.recon_trunk4 = mutil.make_layer(basic_block, nb//4)
        self.conv_last = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        # B C H W -> B 64 H W
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk1(fea)
        out = self.recon_trunk2(out)
        out = self.recon_trunk3(out)
        # out = self.recon_trunk4(out)
        out = self.lrelu(self.conv_last(out))
        return out
        
class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv2 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out
    
class MSRResNet_wGR_i_fea(nn.Module):
    ''' modified SRResNet'''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, rfea_layer='RB16'):
        super(MSRResNet_wGR_i_fea, self).__init__()
        print('Model: MSRResNet_wGR_i (return feature)')
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        # basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk2 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk3 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk4 = mutil.make_layer(basic_block, nb//4)

       
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first], 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        # fea_out_Conv1 = fea.view(1,-1)
        out = self.recon_trunk1(fea)
        # fea_out_RB4 = out#.view(1,-1)
        out = self.recon_trunk2(out)
        # fea_out_RB8 = out#.view(1,-1)
        out = self.recon_trunk3(out)
        # fea_out_RB12 = out#.view(1,-1)
        out = self.recon_trunk4(out)
        fea_out_RB16 = out#.view(1,-1)
        return fea_out_RB16
        
        
class DeepDegradationEncoder_v1(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(DeepDegradationEncoder_v1, self).__init__()
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        load_net = torch.load(checkpoint)
        self.DDR_extractor.load_state_dict(load_net, strict=True)
        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        self.conv = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.adapool = nn.AdaptiveAvgPool2d(8)
        self.linear1 = nn.Linear(nf*8*8, 512)      
        self.linear2 = nn.Linear(512, 512)       
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        out = self.lrelu(self.conv(DeepDegradationRepre))
        out = self.adapool(out)
        
        B, C, H, W = out.size()
        out = out.view(B, -1)
        
        out = self.lrelu(self.linear1(out))
        out = self.linear2(out)
        
        
        DDR_embedding = out
        
        return DDR_embedding
        
class DeepDegradationEncoder_v2_backup(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=2, checkpoint=None):
        super(DeepDegradationEncoder_v2, self).__init__()
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        self.conv1 = nn.Conv2d(in_channels=nf*2, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(5, 5)
        self.pool3 = nn.MaxPool2d(18, 18)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_last = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1, padding=0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.fc = nn.Linear(2048, 2048)
             
    def forward(self, x):
        b, c, h, w = x.shape
        DeepDegradationRepre = self.DDR_extractor(x) # b c h w -> b 64 h w
        # h w = 360 360
        out = self.lrelu(self.conv1(DeepDegradationRepre)) # b 64 h w -> b 128 h w
        out = self.pool1(out)
        out = self.lrelu(self.conv2(out))  # b 128 h/2 w/2 -> b 256 180 180
        out = self.pool2(out) # b 256 36 36
        out = self.lrelu(self.conv3(out)) # b 256 36 36 -> b 512 36 36
        out = self.pool3(out) # b 512 2 2
        out = self.lrelu(self.conv4(out))
        out = self.pool(out).view(b, 1024, 1, 1) # b 1024
        out = self.conv_last(out)
        return out

class preprocess(nn.Module):
    def __init__(self, nf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = dwconv_hwc(3, nf, 7, 1, 3, 1, 1, True)
        self.conv2 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv3 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv4 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv5 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv6 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv7 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)
        self.conv8 = dwconv_hwc(nf, nf, 7, 1, 3, 1, nf, True)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = out1 + F.relu(self.conv2(out1))
        out3 = out2 + F.relu(self.conv3(out2))
        out4 = out3 + F.relu(self.conv4(out3))
        out5 = out4 + F.relu(self.conv5(out4))
        out6 = out5 + F.relu(self.conv6(out5))
        out7 = out6 + F.relu(self.conv7(out6))
        out8 = out7 + F.relu(self.conv8(out7))
        return out8


class DeepDegradationEncoder_v2(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v2, self).__init__()
        
        # self.DDR_extractor = preprocess(nf)    
        self.DDR_extractor = MSRResNet_wGR_i_fea(nf=nf)    
        self.conv1 = nn.Conv2d(nf, nf*2, 4, 2, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf*4, 4, 2, 0, bias=True)
        self.conv3 = nn.Conv2d(nf*4, nf*8, 4, 2, 0, bias=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.conv1x1_2 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
             
    def forward(self, x):
        DeepDegradationRepre = self.DDR_extractor(x)
        out = self.lrelu(self.conv1(DeepDegradationRepre)) # in: [B, 64, H, W]  out: [B, 128, H/2, W/2]
        out = self.lrelu(self.conv2(out)) # in: [B, 128, H/2, W/2]  out: [B, 256, H/4, W/4]
        out = self.lrelu(self.conv3(out)) # in: [B, 256, H/4, W/4]  out: [B, 512, H/8, W/8]
        out = self.adapool(out) # in: [B, 512, H/8, W/8]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_1(out)) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.conv1x1_2(out) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        
        DDR_embedding = out
        return DDR_embedding
class DeepDegradationEncoder_v3(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v3, self).__init__()
        
        print('DeepDegradationEncoder_v3')
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        #checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
        else:
            print('No pretrained_DDR_model found!')
            exit()
        self.DDR_extractor.load_state_dict(load_net, strict=True)

        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        self.conv1 = nn.Conv2d(nf, nf*2, 4, 2, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf*2, 4, 2, 0, bias=True)
        self.conv3 = nn.Conv2d(nf*2, nf*2, 4, 2, 0, bias=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(nf*2, nf*2, 1, bias=True)
        self.conv1x1_2 = nn.Conv2d(nf*2, nf*2, 1, bias=True)
        self.conv1x1_3 = nn.Conv2d(nf*2, nf*2, 1, bias=True)
        
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
             
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        out = self.lrelu(self.conv1(DeepDegradationRepre)) # in: [B, 64, H, W]  out: [B, 64, H/2, W/2]
        out = self.lrelu(self.conv2(out)) # in: [B, 64, H/2, W/2]  out: [B, 64, H/4, W/4]
        out = self.lrelu(self.conv3(out)) # in: [B, 64, H/4, W/4]  out: [B, 64, H/8, W/8]
        out = self.adapool(out) # in: [B, 64, H/8, W/8]  out: [B, 64, 1, 1]
        out = self.lrelu(self.conv1x1_1(out)) # in: [B, 64, 1, 1]  out: [B, 64, 1, 1]
        out = self.lrelu(self.conv1x1_2(out)) # in: [B, 64, 1, 1]  out: [B, 64, 1, 1]
        out = self.conv1x1_3(out) # in: [B, 128, 1, 1]  out: [B, 128, 1, 1]
                
        
        DDR_embedding = out
        
        return DDR_embedding

class DeepDegradationEncoder_v4(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v4, self).__init__()
        
        print('DeepDegradationEncoder_v4')
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        #checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
        else:
            print('No pretrained_DDR_model found!')
            exit()
        self.DDR_extractor.load_state_dict(load_net, strict=True)

        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        self.E = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
             
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        fea = self.E(DeepDegradationRepre).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)
                
        
        DDR_embedding = out
        
        return DDR_embedding


class DeepDegradationEncoder_v5(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v5, self).__init__()
        
        print('DeepDegradationEncoder_v5')
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        #checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
        else:
            print('No pretrained_DDR_model found!')
            exit()
        self.DDR_extractor.load_state_dict(load_net, strict=True)

        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        self.E = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 1, bias=True)
        )
             
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        fea = self.E(DeepDegradationRepre).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)
                
        
        DDR_embedding = out
        
        return DDR_embedding

class DeepDegradationEncoder_v6(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v6, self).__init__()
        
        print('DeepDegradationEncoder_v6')
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        #checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
        else:
            print('No pretrained_DDR_model found!')
            exit()
        self.DDR_extractor.load_state_dict(load_net, strict=True)

        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        self.E = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=7, stride=4, padding=0),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=5, stride=3, padding=0),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 256),
        )
             
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        fea = self.E(DeepDegradationRepre).squeeze(-1).squeeze(-1)
        out = self.mlp(fea)
                
        
        DDR_embedding = out
        
        return DDR_embedding

class DeepDegradationEncoder_v7(nn.Module):
    ''' DeepDegradationEncoder'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder_v7, self).__init__()
        
        print('DeepDegradationEncoder_v7')
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)
        #checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        if checkpoint is not None:
            print('pretrained_DDR_model: {}'.format(checkpoint))
            load_net = torch.load(checkpoint)
            self.DDR_extractor.load_state_dict(load_net, strict=True)
        else:
            print('No pretrained_DDR_model found!')
            #exit()
        

        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            #i.requires_grad = False
            i.requires_grad = True
        
        self.E = nn.Sequential(
            #nn.Conv2d(64, 64, kernel_size=3, padding=1),
            #nn.BatchNorm2d(64),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=7, stride=4, padding=0),
            #nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm2d(128),
            #nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=5, stride=3, padding=0),
            #nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            #nn.Conv2d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm2d(256),
            #nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=True),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, 1, bias=True),
        )
             
        

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        fea = self.E(DeepDegradationRepre)
        out = self.mlp(fea)
                
        
        DDR_embedding = out
        
        return DDR_embedding

def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components

        


class ConditionNet_GFM(nn.Module):
    def __init__(self, in_nc=64, nf=64, condition_padding_mode='Reflection', condition_norm='None'):
        super(ConditionNet_GFM, self).__init__()
        print('Modulation: ConditionNet_GFM')
        print('Condition nf: {}'.format(nf))
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)
        checkpoint = '/home/yhliu/BasicSR_DDG/pretrained_DDR_models/B02_MSRGAN_wGR_i_DIV2K_clean/models/400000_G.pth'
        load_net = torch.load(checkpoint)
        self.DDR_extractor.load_state_dict(load_net, strict=True)
        print('Load Deep Degradation Representation Extractor successfully!')
        
        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        
        
        stride = 2
        self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, 0, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, 0, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, 0, bias=True)
                
        if condition_padding_mode  == 'Reflection':
            print('Condition Network Padding: Reflection')
            self.explicit_pad = True   
            self.pad_1 = nn.ReflectionPad2d(1)
        elif condition_padding_mode  == 'Replication':
            print('Condition Network Padding: Replication')
            self.explicit_pad = True
            self.pad_1 = nn.ReplicationPad2d(1)
        elif condition_padding_mode  == 'ZeroPad':
            print('Condition Network Padding: ZeroPad')
            self.explicit_pad = True
            self.pad_1 = nn.ZeroPad2d(1)
        elif condition_padding_mode  == 'ConvPad':
            print('Condition Network Padding: ConvPad')
            self.explicit_pad = False
            self.conv1 = nn.Conv2d(in_nc, nf, 7, stride, 1, bias=True)
            self.conv2 = nn.Conv2d(nf, nf, 3, stride, 1, bias=True)
            self.conv3 = nn.Conv2d(nf, nf, 3, stride, 1, bias=True)
        elif condition_padding_mode  == 'None':
            print('Condition Network Padding: None')
            self.explicit_pad = False
        
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.act = nn.ReLU(inplace=True)
        
        
        self.norm = condition_norm
        print('Condition Norm: ', self.norm)

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
        x = DeepDegradationRepre
        if self.explicit_pad == True:
            x = self.pad_1(x)
        conv1_out = self.act(self.conv1(x))
        
        if self.explicit_pad == True:
            conv1_out = self.pad_1(conv1_out)
        conv2_out = self.act(self.conv2(conv1_out))
        
        if self.explicit_pad == True:
            conv2_out = self.pad_1(conv2_out)
        conv3_out = self.act(self.conv3(conv2_out))
        
        '''
        print(conv3_out.size())
        print(conv3_out[0,0,:,:].size())
         
        B, C, H, W = conv3_out.size()
        import matplotlib.pyplot as plt
        for i in range(C):
            plt.subplot(4,8,i+1)
            plt.imshow(conv3_out[0,i,:,:].cpu().numpy(),cmap='gray')
        plt.show()
        '''

        out = self.pooling(conv3_out)
        
        
        if self.norm == 'UN':
            #print(self.norm)
            scale = torch.rsqrt(torch.mean(out ** 2, dim=[1], keepdim=True) + 1e-8) 
            out_cond = out * scale
        elif self.norm == 'UN_noN':
            scale = out/(torch.norm(out,dim=1,keepdim=True) + 1e-8)
            out_cond = out * scale
        elif self.norm == 'Softmax':
            out_cond = F.softmax(out)
        elif self.norm == 'Softmax_weight':
            scale = F.softmax(out)
            out_cond = out * scale
        elif self.norm == 'Sigmoid':
            out_cond = F.sigmoid(out)
        elif self.norm == 'Sigmoid_weight':
            scale = F.sigmoid(out)
            out_cond = out * scale             
        elif self.norm == 'Z_score':    
            out_cond = (out-torch.mean(out,dim=1,keepdim=True))/torch.std(out,dim=1,keepdim=True)
        elif self.norm == 'Linear':
            max_val, _ = torch.max(out, dim=1, keepdim=True)
            min_val, _ = torch.min(out, dim=1, keepdim=True)
            out_cond = (out - min_val)/(max_val - min_val + 1e-8)
        elif self.norm == 'None':    
            out_cond = out
        else:
            raise Exception('Invalid normalization type {}'.format(self.norm))
        
        
        '''
        fname = 'NormTest007_CSRNet_GFM_3layer_ZeroPad_FiveK_Linear_600000.h5'
        condition_vector = out_cond.view(1, -1).cpu().numpy()
        #print(condition_vector.shape)
        if not os.path.exists(fname):
            f = tables.open_file(fname, mode='w')
            atom = tables.Float64Atom()
            
            array_c = f.create_earray(f.root, 'data', atom, (0, 32))
            
            array_c.append(condition_vector)
            f.close()
        
        else:
            f = tables.open_file(fname, mode='a')
            f.root.data.append(condition_vector)
            f.close()
        '''
        
        
        return out_cond

# 7 layers with GFM
class CSRNet_GFM_7(nn.Module):
    def __init__(self, in_nc=3, nf=64, cond_dim=64, base_conv=3, condition_padding_mode='Reflection', noise=False):
        super(CSRNet_GFM_7, self).__init__()
        print('Main framework: GFM')
        print('Base network: 7 layers; {} channels'.format(nf))

        self.GFM_nf = nf
        fc_bias = True

        self.cond_scale_first = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_HR = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_HR2 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_HR3 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_HR4 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_HR5 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_scale_last = nn.Linear(cond_dim, 3, bias=fc_bias)

        self.cond_shift_first = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_HR = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_HR2 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_HR3 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_HR4 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_HR5 = nn.Linear(cond_dim, nf, bias=fc_bias)
        self.cond_shift_last = nn.Linear(cond_dim, 3, bias=fc_bias)

        # base network
        if base_conv == 1:
            print('Base Network conv: 1x1')
            self.conv_first = nn.Conv2d(in_nc, nf, 1, 1, bias=True)  # 64f
            self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv3 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv5 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
            
        elif base_conv == 3:
            print('Base Network conv: 3x3')
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)  # 64f
            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

            # bigger kernel size
            # self.conv_first = nn.Conv2d(in_nc, nf, 7, 1, 3, bias=True)  # 64f
            # self.HRconv = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
            # self.HRconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            # self.HRconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            # self.HRconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            # self.HRconv5 = nn.Conv2d(nf, nf, 5, 1, 2, bias=True)
            # self.conv_last = nn.Conv2d(nf, 3, 5, 1, 2, bias=True)

            
        
        self.act = nn.ReLU(inplace=True)
        
        self.noise = noise
        if self.noise:        
            self.weight_noise = nn.Parameter(torch.zeros(1))  # for noise injection
            self.activation = nn.LeakyReLU(0.1, True)
        
        print('Add random noise layer: ', self.noise)
        


    def forward(self, x, embedding):
       
        cond_mapped = embedding
        B, C, H, W = cond_mapped.size()
        cond_mapped = cond_mapped.view(B, -1)

        scale_first = self.cond_scale_first(cond_mapped)
        shift_first = self.cond_shift_first(cond_mapped)

        scale_HR = self.cond_scale_HR(cond_mapped)
        shift_HR = self.cond_shift_HR(cond_mapped)
        
        scale_HR2 = self.cond_scale_HR2(cond_mapped)
        shift_HR2 = self.cond_shift_HR2(cond_mapped)
        
        scale_HR3 = self.cond_scale_HR3(cond_mapped)
        shift_HR3 = self.cond_shift_HR3(cond_mapped)
        
        scale_HR4 = self.cond_scale_HR4(cond_mapped)
        shift_HR4 = self.cond_shift_HR4(cond_mapped)
        
        scale_HR5 = self.cond_scale_HR5(cond_mapped)
        shift_HR5 = self.cond_shift_HR5(cond_mapped)

        scale_last = self.cond_scale_last(cond_mapped)
        shift_last = self.cond_shift_last(cond_mapped)

        out = self.conv_first(x)
        out = out * scale_first.view(-1, self.GFM_nf, 1, 1) + shift_first.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        out = self.HRconv(out)
        out = out * scale_HR.view(-1, self.GFM_nf, 1, 1) + shift_HR.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)
        
        if self.noise:
            b, _, h, w = out.shape
            n = out.new_empty(b, 1, h, w).normal_()
            out = out + self.weight_noise * n
            out = self.activation(out)
        
        out = self.HRconv2(out)
        out = out * scale_HR2.view(-1, self.GFM_nf, 1, 1) + shift_HR2.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)
        
        if self.noise:
            out = out + self.weight_noise * n
            out = self.activation(out)
        
        out = self.HRconv3(out)
        out = out * scale_HR3.view(-1, self.GFM_nf, 1, 1) + shift_HR3.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)
        
        if self.noise:
            out = out + self.weight_noise * n
            out = self.activation(out)
        
        out = self.HRconv4(out)
        out = out * scale_HR4.view(-1, self.GFM_nf, 1, 1) + shift_HR4.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)
        
        if self.noise:
            out = out + self.weight_noise * n
            out = self.activation(out)
  
        out = self.HRconv5(out)
        out = out * scale_HR5.view(-1, self.GFM_nf, 1, 1) + shift_HR5.view(-1, self.GFM_nf, 1, 1) + out
        out = self.act(out)

        if self.noise:
            out = out + self.weight_noise * n
            out = self.activation(out)


        out = self.conv_last(out)
        out = out * scale_last.view(-1, 3, 1, 1) + shift_last.view(-1, 3, 1, 1) + out

        return out

class DeepDegradationGenerator(nn.Module):
    ''' DeepDegradationGenerator'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(DeepDegradationGenerator, self).__init__()
        
        
        self.conv = nn.Conv2d(nf, nf, 4, 2, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.adapool = nn.AdaptiveAvgPool2d(8)
        self.linear1 = nn.Linear(nf*8*8, 512)      
        self.linear2 = nn.Linear(512, 512)       
        

    def forward(self, x, embedding):
        _, DeepDegradationRepre = self.DDR_extractor(x)
            
        out = self.lrelu(self.conv(DeepDegradationRepre))
        out = self.adapool(out)
        
        B, C, H, W = out.size()
        out = out.view(B, -1)
        
        out = self.lrelu(self.linear1(out))
        out = self.linear2(out)
        
        
        DDR_embedding = out
        
        return DDR_embedding        
 

class StyleGAN2_mod(nn.Module):

    def __init__(self, out_channels, modul_channels, kernel_size = 3, noise=False):
        super().__init__()


        #self.conv = ResBlock(in_channels, out_channels)

        # generate global conv weights
        fan_in = out_channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.out_channels = out_channels
        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(modul_channels, self.out_channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.out_channels, kernel_size, kernel_size)
        )
        self.conv_last = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        self.noise = noise
        if self.noise:        
            self.weight_noise = nn.Parameter(torch.zeros(1))  # for noise injection
        
        self.activation = nn.LeakyReLU(0.1, True)
        
        print('Add random noise layer: ', self.noise)


    def forward(self, x, xg):
        # for global adjustation
        B, _, H, W = x.size()
        
        C = self.out_channels
        style = self.modulation(xg).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )
        
        
        #x = self.conv(x)

        x_input = x.view(1, B * C, H, W)
        x_global = F.conv2d(x_input, weight, padding=self.padding, groups=B)
        x_global = x_global.view(B, C, H, W)
        
        if self.noise:
            b, _, h, w = x_global.shape
            n = x_global.new_empty(b, 1, h, w).normal_()
            x_global += self.weight_noise * n
            

        x = self.conv_last(x_global)
        x = self.activation(x)

        return x

# 7 layers with GFM
class CSRNet_GFM_7_StyleGAN2(nn.Module):
    def __init__(self, in_nc=3, nf=64, embed_ch=512, base_conv=3):
        super(CSRNet_GFM_7_StyleGAN2, self).__init__()
        print('Main framework: CSRNet_GFM_7_StyleGAN2')
        print('Base network: 7 layers; {} channels'.format(nf))

        self.GFM_nf = nf


        # base network
        if base_conv == 1:
            print('Base Network conv: 1x1')
            self.conv_first = nn.Conv2d(in_nc, nf, 1, 1, bias=True)  # 64f
            self.HRconv = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv2 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv3 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv4 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.HRconv5 = nn.Conv2d(nf, nf, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, 3, 1, 1, bias=True)
            
        elif base_conv == 3:
            print('Base Network conv: 3x3')
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)  # 64f
            self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.HRconv5 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        
        self.act = nn.LeakyReLU(0.1, True)
        
        self.modulation0 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 3, noise=True)
        self.modulation1 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 3, noise=True)
        self.modulation2 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 3, noise=True)
        self.modulation3 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 3, noise=False)
        self.modulation4 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 3, noise=False)
        self.modulation5 = StyleGAN2_mod(out_channels=nf, modul_channels=embed_ch, kernel_size = 5, noise=False)
  
        
        


    def forward(self, x, embedding):
       
        out = self.conv_first(x)
        out = self.act(out)
        out = self.modulation0(out, embedding)

        out = self.HRconv(out)
        out = self.modulation1(out, embedding)
        
        
        out = self.HRconv2(out)
        out = self.modulation2(out, embedding)
        
        out = self.HRconv3(out)
        self.modulation3(out, embedding)
        
        out = self.HRconv4(out)
        self.modulation4(out, embedding)
        
  
        out = self.HRconv5(out)
        self.modulation5(out, embedding)


        out = self.conv_last(out)
        

        return out

###########################################################################
class ResBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bottle_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.double_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.bottle_conv(x)
        x = self.double_conv(x) + x
        return x / math.sqrt(2)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        return x

class Down(nn.Module):
    """Downscaling with stride conv then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 4, 2, 1),
            nn.LeakyReLU(0.1, True),
            # DoubleConv(in_channels, out_channels)
            ResBlock(in_channels, out_channels)
        )
        
    def forward(self, x):
        x = self.main(x)

        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, modul_channels, kernel_size = 3, bilinear=True, noise=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            self.conv = ResBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = ResBlock(in_channels, out_channels)

        # generate global conv weights
        fan_in = out_channels * kernel_size ** 2
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.out_channels = out_channels
        self.scale = 1 / math.sqrt(fan_in)
        self.modulation = nn.Conv2d(modul_channels, self.out_channels, 1)
        self.weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.out_channels, kernel_size, kernel_size)
        )
        self.conv_last = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        
        self.noise = noise
        if self.noise:        
            self.weight_noise = nn.Parameter(torch.zeros(1))  # for noise injection
        self.activation = nn.LeakyReLU(0.1, True)
        
        print('Add random noise layer: ', self.noise)


    def forward(self, x1, x2, xg):
        # for global adjustation
        B, _, H, W = x2.size()
        
        C = self.out_channels
        style = self.modulation(xg).view(B, 1, C, 1, 1)
        weight = self.scale * self.weight * style
        # demodulation
        demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
        weight = weight * demod.view(B, C, 1, 1, 1)

        weight = weight.view(
            B * C, C, self.kernel_size, self.kernel_size
        )
        
        
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
    

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        x_input = x.view(1, B * C, H, W)
        x_global = F.conv2d(x_input, weight, padding=self.padding, groups=B)
        x_global = x_global.view(B, C, H, W)
        
        if self.noise:
            b, _, h, w = x_global.shape
            n = x_global.new_empty(b, 1, h, w).normal_()
            x_global += self.weight_noise * n
            

        x = self.conv_last(x_global)
        x = self.activation(x)

        return x
 
class UNet4(nn.Module):
    def __init__(self, n_channels=3, modul_channels=64, bilinear=True, noise=False):
        super(UNet4, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, modul_channels, 3, bilinear)
        self.up2 = Up(512, 256 // factor, modul_channels, 3, bilinear)
        self.up3 = Up(256, 128 // factor, modul_channels, 5, bilinear)
        self.up4 = Up(128, 64, modul_channels, 5, bilinear)
        self.rec = nn.Sequential(
                DoubleConv(64, 64),
                DoubleConv(64, 64)
        )
        self.outc = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.1, True),
                nn.Conv2d(64, 3, 3, 1, 1),
        )

    def forward(self, x, embedding):
        # print(torch.max(x[0]), torch.min(x[0])) #[-1, 1] gray image L
        # print(torch.max(x[1]), torch.min(x[1])) # color vector

        #embedding = embedding.unsqueeze(2).unsqueeze(2)
        
        x1 = self.inc(x)
        x2 = self.down1(x1) # [B, 128, 128, 128]
        x3 = self.down2(x2) # [B, 256, 64, 64]
        x4 = self.down3(x3) # [B, 512, 32, 32]
        x5 = self.down4(x4) # [B, 512, 16, 16]

        x6 = self.up1(x5, x4, embedding) # [B, 256, 32, 32]
        x7 = self.up2(x6, x3, embedding) # [B, 128, 64, 64]
        x8 = self.up3(x7, x2, embedding) # [B, 64, 128, 128]
        x9 = self.up4(x8, x1, embedding) # [B, 64, 256, 256]
        out = self.rec(x9)
        out = self.outc(out)

        return out


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class single_dwconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, mode='cdc7'):
        super().__init__()
        assert mode in ['cdc7', 'cdc5', 'cdc3', 'conv']
        if mode == 'cdc7':
            self.conv = decouple_dwconv_hw_cdc(in_channels, out_channels, 7, stride, 3, dilation, groups, bias)
        elif mode == 'cdc5':
            self.conv = decouple_dwconv_hw_cdc(in_channels, out_channels, 5, stride, 2, dilation, groups, bias)
        elif mode == 'cdc3':
            self.conv = decouple_dwconv_hw_cdc(in_channels, out_channels, 3, stride, 1, dilation, groups, bias)
        elif mode == 'conv':
            self.conv = nn.Conv2d(in_channels, out_channels, 7, stride, 3, dilation, groups, bias)
    
    def forward(self, x):
        return self.conv(x)
    
class single_preprocess(nn.Module):
    def __init__(self, nf, mode):
        super().__init__()
        self.conv1 = single_dwconv(3, nf, 7, 1, 3, 1, 1, True, mode)
        self.conv2 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv3 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv4 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv5 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv6 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv7 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)
        self.conv8 = single_dwconv(nf, nf, 7, 1, 3, 1, nf, True, mode)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = out1 + F.relu(self.conv2(out1))
        out3 = out2 + F.relu(self.conv3(out2))
        out4 = out3 + F.relu(self.conv4(out3))
        out5 = out4 + F.relu(self.conv5(out4))
        out6 = out5 + F.relu(self.conv6(out5))
        out7 = out6 + F.relu(self.conv7(out6))
        out8 = out7 + F.relu(self.conv8(out7))
        return out8

class DeepDegradationEncoder_v2_parallel(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super().__init__()
        # 四路 preprocess
        self.DDR_extractor_cdc7 = single_preprocess(nf, mode='cdc7')
        self.DDR_extractor_cdc5 = single_preprocess(nf, mode='cdc5')
        self.DDR_extractor_cdc3 = single_preprocess(nf, mode='cdc3')
        self.DDR_extractor_conv = single_preprocess(nf, mode='conv')

        # 下采样和1x1卷积，每路独立
        def make_downsample_block():
            return nn.Sequential(
                nn.Conv2d(nf, nf*2, 4, 2, 0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf*2, nf*4, 4, 2, 0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf*4, nf*8, 4, 2, 0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(nf*8, nf*8, 1, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(nf*8, nf*8, 1, bias=True)
            )
        self.down_cdc7 = make_downsample_block()
        self.down_cdc5 = make_downsample_block()
        self.down_cdc3 = make_downsample_block()
        self.down_conv = make_downsample_block()

    def forward(self, x):
        # 四路并行
        out_cdc7 = self.down_cdc7(self.DDR_extractor_cdc7(x))  # [B, 512, 1, 1]
        out_cdc5 = self.down_cdc5(self.DDR_extractor_cdc5(x))
        out_cdc3 = self.down_cdc3(self.DDR_extractor_cdc3(x))
        out_conv = self.down_conv(self.DDR_extractor_conv(x))
        # 拼接
        DDR_embedding = torch.cat([out_cdc7, out_cdc5, out_cdc3, out_conv], dim=1)  # [B, 2048, 1, 1]
        return DDR_embedding
    

    
        
if __name__ ==  '__main__':
    DDE = DeepDegradationEncoder_v2_parallel().cuda()
    # DDG = UNet4()
    x = torch.randn(1, 3, 360, 360).cuda()
    # DDE.train()
    out_embedding = DDE(x)
    flops = FlopCountAnalysis(DDE, x)
    print(f"FLOPs: {flops.total()/1e9}")
    print(out_embedding.shape)
    # 计算参数量
    params = parameter_count_table(DDE)
    print(params)

    # DDE.eval()
    # out_embedding = DDE(x)
    # flops = FlopCountAnalysis(DDE, x)
    # print(f"FLOPs: {flops.total()/1e9}")
    # print(out_embedding.shape)
    # # 计算参数量
    # params = parameter_count_table(DDE)
    # print(params)

    # import time
    # import torch
    # from tqdm import tqdm

    # def test_inference_speed(net, input_size=(1, 3, 224, 224), num_iterations=100, warmup=10):
    #     device = next(net.parameters()).device
    #     # net.eval()
    #     input_tensor = torch.randn(*input_size).to(device)
        
    #     # 预热阶段（不记录时间）
    #     with torch.no_grad():
    #         for _ in range(warmup):
    #             _ = net(input_tensor)
    #         if device.type == 'cuda':
    #             torch.cuda.synchronize()  # 确保GPU操作完成
        
    #     # 正式计时
    #     timings = []
    #     with torch.no_grad():
    #         for _ in tqdm(range(num_iterations)):
    #             start_time = time.perf_counter()
    #             _ = net(input_tensor)
    #             if device.type == 'cuda':
    #                 torch.cuda.synchronize()  # 同步GPU操作
    #             end_time = time.perf_counter()
    #             timings.append(end_time - start_time)
        
    #     avg_time_per_iteration = sum(timings) / num_iterations
    #     print(f"Average inference time: {avg_time_per_iteration * 1000:.3f} ms")
    #     print(f"FPS: {1 / avg_time_per_iteration:.1f}")
    
    # DDE.train()
    # test_inference_speed(DDE, num_iterations=100, warmup=20)
    # DDE.eval()
    # test_inference_speed(DDE, num_iterations=100, warmup=20)


    
class AttentionNet(nn.Module):
    ''' AttentionNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(AttentionNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.CA_Block, nf=nf)
        self.recon_trunk = mutil.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            mutil.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


# SwinIR