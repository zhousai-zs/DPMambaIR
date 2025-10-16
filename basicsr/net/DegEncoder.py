import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import net.module_util as mutil
import math
import copy

class MSRResNet_wGR_i_fea(nn.Module):
    ''' modified SRResNet'''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, rfea_layer='RB16'):
        super(MSRResNet_wGR_i_fea, self).__init__()
        print('Model: MSRResNet_wGR_i (return feature)')
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_trunk1 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk2 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk3 = mutil.make_layer(basic_block, nb//4)
        self.recon_trunk4 = mutil.make_layer(basic_block, nb//4)

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
        elif self.upscale == 1:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        mutil.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last], 0.1)
        if self.upscale == 4:
            mutil.initialize_weights(self.upconv2, 0.1)
        
        self.rfea_layer = rfea_layer
        print('Return feature layer: {}'.format(self.rfea_layer))

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        fea_out_Conv1 = fea.view(1,-1)
        out = self.recon_trunk1(fea)
        fea_out_RB4 = out#.view(1,-1)
        out = self.recon_trunk2(out)
        fea_out_RB8 = out#.view(1,-1)
        out = self.recon_trunk3(out)
        fea_out_RB12 = out#.view(1,-1)
        out = self.recon_trunk4(out)
        fea_out_RB16 = out#.view(1,-1)
 
        if self.rfea_layer == 'Conv1':
            return out, fea_out_Conv1
        elif self.rfea_layer == 'RB4':
            return out, fea_out_RB4
        elif self.rfea_layer == 'RB8':
            return out, fea_out_RB8
        elif self.rfea_layer == 'RB12':
            return out, fea_out_RB12
        elif self.rfea_layer == 'RB16':
            return out, fea_out_RB16

class DeepDegradationEncoder(nn.Module):
    ''' DeepDegradationEncoder'''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4, checkpoint=None):
        super(DeepDegradationEncoder, self).__init__()
        
        self.DDR_extractor = MSRResNet_wGR_i_fea(in_nc, out_nc, nf, nb, upscale=4)

        for i in self.DDR_extractor.parameters():
            i.requires_grad = False
        self.conv1 = nn.Conv2d(nf, nf*2, 4, 2, 0, bias=True)
        self.conv2 = nn.Conv2d(nf*2, nf*4, 4, 2, 0, bias=True)
        self.conv3 = nn.Conv2d(nf*4, nf*8, 4, 2, 0, bias=True)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1_1 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.conv1x1_2 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.conv1x1_3 = nn.Conv2d(nf*8, nf*8, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        _, DeepDegradationRepre = self.DDR_extractor(x)
        out = self.lrelu(self.conv1(DeepDegradationRepre)) # in: [B, 64, H, W]  out: [B, 128, H/2, W/2]
        out = self.lrelu(self.conv2(out)) # in: [B, 128, H/2, W/2]  out: [B, 256, H/4, W/4]
        out = self.lrelu(self.conv3(out)) # in: [B, 256, H/4, W/4]  out: [B, 512, H/8, W/8]
        out = self.adapool(out) # in: [B, 512, H/8, W/8]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_1(out)) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.lrelu(self.conv1x1_2(out)) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        out = self.conv1x1_3(out) # in: [B, 512, 1, 1]  out: [B, 512, 1, 1]
        
        DDR_embedding = out
        return DDR_embedding

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

        # initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8], 0.1)

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

class decouple_dwconv_hw_cdc(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.location = [kernel_size // 2, kernel_size // 2]
        # print(self.location[0], self.location[1])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input ):

        return self.decouple_dwconv_hw_cdc_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def decouple_dwconv_hw_cdc_func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        
        yc = F.conv2d(x, zero_weight, stride=stride, padding=padding, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y - yc
    
    def get_weight(self):
        weights = copy.deepcopy(self.weight)
        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        weights = weights - zero_weight
        return weights

class dwconv_hwc(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # self.conv1 = decouple_dwconv_hw_cac(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        #                                     padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.cdc = decouple_dwconv_hw_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        #                                     padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
        #                                     padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.cdc7 = decouple_dwconv_hw_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride,
                                            padding=3, dilation=dilation, groups=groups, bias=bias)
        self.cdc5 = decouple_dwconv_hw_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=stride,
                                            padding=2, dilation=dilation, groups=groups, bias=bias)
        self.cdc3 = decouple_dwconv_hw_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                                            padding=1, dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride,
                                            padding=3, dilation=dilation, groups=groups, bias=bias)

        
    def forward(self, x):
        if self.training:
            out =  self.conv(x) + self.cdc7(x) + self.cdc5(x) + self.cdc3(x)
            return out
        else:
            return self.test_forward(x)
        
    def re_para(self):
        self.conv_w = self.conv.weight.data
        self.conv_w = self.conv_w.to(self.conv.weight.device)

        self.cdc7_w = self.cdc7.get_weight()
        self.cdc7_w = self.cdc7_w.to(self.conv.weight.device)

        self.cdc5_w = self.cdc5.get_weight()
        self.cdc5_w = self.cdc5_w.to(self.conv.weight.device)

        self.cdc3_w = self.cdc3.get_weight()
        self.cdc3_w = self.cdc3_w.to(self.conv.weight.device)        

        self.K = self.cdc7_w + F.pad(self.cdc5_w, (1,1,1,1)) + F.pad(self.cdc3_w, (2,2,2,2)) + self.conv_w

        if self.conv.bias  is not None:
            self.conv_b = self.conv.bias.data 
            self.conv_b = self.conv_b.to(self.conv.weight.device)
            self.cdc7_b = self.cdc7.bias.data 
            self.cdc7_b = self.cdc7_b.to(self.conv.weight.device)
            self.cdc5_b = self.cdc5.bias.data 
            self.cdc5_b = self.cdc5_b.to(self.conv.weight.device)
            self.cdc3_b = self.cdc3.bias.data 
            self.cdc3_b = self.cdc3_b.to(self.conv.weight.device)
            self.B = self.conv_b + self.cdc7_b + self.cdc5_b + self.cdc3_b
        

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x, weight=self.K, bias=self.B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

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
        
        self.DDR_extractor = preprocess(nf)        
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

if __name__ == "__main__":
    netE = DeepDegradationEncoder_v2_parallel()
    netE.load_state_dict(torch.load('/zs/DegAE_DegradationAutoencoder/experiments/Train_Degradation_Encoder_Para4/models/latest_DDG_Encoder.pth'), strict=True)

    x = torch.randn(1, 3, 64, 64)
    print(x.shape)
    out1 = netE(x)
    print(out1.shape)