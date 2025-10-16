import math
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """
    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_hw = dwconv_hw(gc, gc, square_kernel_size, padding=square_kernel_size//2, groups=gc)
        self.dwconv_w = dwconv_w(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=gc)
        self.dwconv_h = dwconv_h(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), 
            dim=1,
        )

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
    
class decouple_dwconv_hw_cac(nn.Module):
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

        return self.decouple_dwconv_hw_cac_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def decouple_dwconv_hw_cac_func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        
        yc = F.conv2d(x, zero_weight, stride=stride, padding=padding, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y + yc
    
    def get_weight(self):

        weights = copy.deepcopy(self.weight)
        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        weights = weights + zero_weight
        return weights
    
class decouple_dwconv_w_cdc(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(1, 3), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.location = [0, kernel_size[1] // 2]
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
    
class decouple_dwconv_w_cac(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(1, 3), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.location = [0, kernel_size[1] // 2]
        # print(self.location[0], self.location[1])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input ):

        return self.decouple_dwconv_hw_cac_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def decouple_dwconv_hw_cac_func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        
        yc = F.conv2d(x, zero_weight, stride=stride, padding=padding, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y + yc
    
    def get_weight(self):

        weights = copy.deepcopy(self.weight)
        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        weights = weights + zero_weight
        return weights

class decouple_dwconv_h_cdc(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(3, 1), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.location = [kernel_size[0] // 2, 0]
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
    
class decouple_dwconv_h_cac(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(3, 1), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size[0], kernel_size[1]))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.location = [kernel_size[0] // 2, 0]
        # print(self.location[0], self.location[1])

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input ):

        return self.decouple_dwconv_hw_cac_func(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def decouple_dwconv_hw_cac_func(self , x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):

        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        
        yc = F.conv2d(x, zero_weight, stride=stride, padding=padding, groups=groups)
        y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        return y + yc
    
    def get_weight(self):

        weights = copy.deepcopy(self.weight)
        weights_c = weights.sum(dim=[2, 3])
        zero_weight = torch.zeros(weights.size()).to(weights.device)
        zero_weight[:,:,self.location[0],self.location[1]] = weights_c
        weights = weights + zero_weight
        return weights

class dwconv_hw(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cac = decouple_dwconv_hw_cac(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.cdc = decouple_dwconv_hw_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.cdc_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.cac_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def get_theta(self):
        cac_kernel_mean = self.cac.weight
        cdc_kernel_mean= self.cdc.weight
        conv_kernel_mean= self.conv.weight
        conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
        conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
        cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
        cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
        #merge kernel
        merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
        #get theta from kernel
        conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
        cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
        cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

        conv_theta = conv_theta.view(self.out_channels,1, 1, 1)
        cdc_theta = cdc_theta.view(self.out_channels,1, 1, 1)
        cac_theta = cac_theta.view(self.out_channels,1, 1, 1)
        return conv_theta , cdc_theta, cac_theta
    
    def forward(self, x):
        if self.training:
            cac_kernel_mean = self.cac.weight
            cdc_kernel_mean= self.cdc.weight
            conv_kernel_mean= self.conv.weight
            conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
            conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
            cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
            cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
            #merge kernel
            merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
            #get theta from kernel
            conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
            cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
            cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

            conv_theta = conv_theta.view(1,self.out_channels,1,1)
            cdc_theta = cdc_theta.view(1,self.out_channels,1,1)
            cac_theta = cac_theta.view(1,self.out_channels,1,1)
            out =  conv_theta * self.conv(x) + cdc_theta * self.cdc(x) + cac_theta * self.cac(x)
            return out
        else:
            return self.test_forward(x)
        
    def re_para(self):
        conv_theta, cdc_theta, cac_theta = self.get_theta()
        # conv_w = self.conv.get_weight()
        self.conv_w = self.conv.weight.data
        self.conv_w = self.conv_w.to(self.conv.weight.device)

        self.cdc_w = self.cdc.get_weight()
        self.cdc_w = self.cdc_w.to(self.conv.weight.device)

        self.cac_w = self.cac.get_weight()
        self.cac_w = self.cac_w.to(self.conv.weight.device)
        self.K = conv_theta * self.conv_w + cdc_theta * self.cdc_w + cac_theta * self.cac_w

        if self.conv.bias  is not None:
            cdc_theta = torch.squeeze(cdc_theta)
            conv_theta = torch.squeeze(conv_theta)
            cac_theta = torch.squeeze(cac_theta)
            self.conv_b =  self.conv.bias.data 
            self.conv_b = self.conv_b.to(self.conv.weight.device)
            self.cdc_b =  self.cdc.bias.data 
            self.cdc_b = self.cdc_b.to(self.conv.weight.device)
            self.cac_b =  self.cac.bias.data 
            self.cac_b = self.cac_b.to(self.conv.weight.device)
            self.B = conv_theta * self.conv_b + cdc_theta * self.cdc_b + cac_theta * self.cac_b
        else:
            # self.conv_b = None
            self.B = None

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x, weight=self.K, bias=self.B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
    
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
    

class dwconv(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
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
            out =  self.cdc7(x) + self.cdc5(x) + self.cdc3 + self.conv(x)
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

        self.K = self.cdc7_w + self.cdc5_w + self.cdc3_w + self.conv_w

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
        else:
            # self.conv_b = None
            self.B = None

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x, weight=self.K, bias=self.B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
    
class dwconv_w(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(1, 3), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cac = decouple_dwconv_w_cac(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.cdc = decouple_dwconv_w_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.cdc_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.cac_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def get_theta(self):
        cac_kernel_mean = self.cac.weight
        cdc_kernel_mean= self.cdc.weight
        conv_kernel_mean= self.conv.weight
        conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
        conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
        cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
        cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
        #merge kernel
        merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
        #get theta from kernel
        conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
        cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
        cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

        conv_theta = conv_theta.view(self.out_channels,1, 1, 1)
        cdc_theta = cdc_theta.view(self.out_channels,1, 1, 1)
        cac_theta = cac_theta.view(self.out_channels,1, 1, 1)
        return conv_theta , cdc_theta, cac_theta
    
    def forward(self, x):
        if self.training:
            cac_kernel_mean = self.cac.weight
            cdc_kernel_mean= self.cdc.weight
            conv_kernel_mean= self.conv.weight
            conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
            conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
            cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
            cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
            #merge kernel
            merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
            #get theta from kernel
            conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
            cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
            cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

            conv_theta = conv_theta.view(1,self.out_channels,1,1)
            cdc_theta = cdc_theta.view(1,self.out_channels,1,1)
            cac_theta = cac_theta.view(1,self.out_channels,1,1)
            out =  conv_theta * self.conv(x) + cdc_theta * self.cdc(x) + cac_theta * self.cac(x)
            return out
        else:
            return self.test_forward(x)
        
    def re_para(self):
        conv_theta, cdc_theta, cac_theta = self.get_theta()
        # conv_w = self.conv.get_weight()
        self.conv_w = self.conv.weight.data
        self.conv_w = self.conv_w.to(self.conv.weight.device)

        self.cdc_w = self.cdc.get_weight()
        self.cdc_w = self.cdc_w.to(self.conv.weight.device)

        self.cac_w = self.cac.get_weight()
        self.cac_w = self.cac_w.to(self.conv.weight.device)
        self.K = conv_theta * self.conv_w + cdc_theta * self.cdc_w + cac_theta * self.cac_w

        if self.conv.bias  is not None:
            cdc_theta = torch.squeeze(cdc_theta)
            conv_theta = torch.squeeze(conv_theta)
            cac_theta = torch.squeeze(cac_theta)
            self.conv_b =  self.conv.bias.data 
            self.conv_b = self.conv_b.to(self.conv.weight.device)
            self.cdc_b =  self.cdc.bias.data 
            self.cdc_b = self.cdc_b.to(self.conv.weight.device)
            self.cac_b =  self.cac.bias.data 
            self.cac_b = self.cac_b.to(self.conv.weight.device)
            self.B = conv_theta * self.conv_b + cdc_theta * self.cdc_b + cac_theta * self.cac_b
        else:
            # self.conv_b = None
            self.B = None
        
    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x, weight=self.K, bias=self.B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class dwconv_h(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=(3, 1), stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cac = decouple_dwconv_h_cac(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.cdc = decouple_dwconv_h_cdc(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.cdc_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.conv_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.cac_kernel_linear = nn.Sequential(
            nn.Linear(out_channels*3,out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def get_theta(self):
        cac_kernel_mean = self.cac.weight
        cdc_kernel_mean= self.cdc.weight
        conv_kernel_mean= self.conv.weight
        conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
        cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
        conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
        cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
        cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
        #merge kernel
        merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
        #get theta from kernel
        conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
        cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
        cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

        conv_theta = conv_theta.view(self.out_channels,1, 1, 1)
        cdc_theta = cdc_theta.view(self.out_channels,1, 1, 1)
        cac_theta = cac_theta.view(self.out_channels,1, 1, 1)
        return conv_theta , cdc_theta, cac_theta
    
    def forward(self, x):
        if self.training:
            cac_kernel_mean = self.cac.weight
            cdc_kernel_mean= self.cdc.weight
            conv_kernel_mean= self.conv.weight
            conv_kernel_mean = conv_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cdc_kernel_mean = cdc_kernel_mean.mean(dim=(1,2,3)).contiguous()
            cac_kernel_mean = cac_kernel_mean.mean(dim=(1,2,3)).contiguous()
            conv_kernel_mean=conv_kernel_mean.view(1,self.out_channels)
            cdc_kernel_mean=cdc_kernel_mean.view(1,self.out_channels)
            cac_kernel_mean=cac_kernel_mean.view(1,self.out_channels)
            #merge kernel
            merge_kernel = torch.cat((cdc_kernel_mean, conv_kernel_mean, cac_kernel_mean),dim=1)
            #get theta from kernel
            conv_theta = torch.sigmoid(self.conv_kernel_linear(merge_kernel)).contiguous()
            cdc_theta = torch.sigmoid(self.cdc_kernel_linear(merge_kernel)).contiguous()
            cac_theta = torch.sigmoid(self.cac_kernel_linear(merge_kernel)).contiguous()

            conv_theta = conv_theta.view(1,self.out_channels,1,1)
            cdc_theta = cdc_theta.view(1,self.out_channels,1,1)
            cac_theta = cac_theta.view(1,self.out_channels,1,1)
            out =  conv_theta * self.conv(x) + cdc_theta * self.cdc(x) + cac_theta * self.cac(x)
            return out
        else:
            return self.test_forward(x)
        
    def re_para(self):
        conv_theta, cdc_theta, cac_theta = self.get_theta()
        # conv_w = self.conv.get_weight()
        self.conv_w = self.conv.weight.data
        self.conv_w = self.conv_w.to(self.conv.weight.device)

        self.cdc_w = self.cdc.get_weight()
        self.cdc_w = self.cdc_w.to(self.conv.weight.device)

        self.cac_w = self.cac.get_weight()
        self.cac_w = self.cac_w.to(self.conv.weight.device)
        self.K = conv_theta * self.conv_w + cdc_theta * self.cdc_w + cac_theta * self.cac_w

        if self.conv.bias  is not None:
            cdc_theta = torch.squeeze(cdc_theta)
            conv_theta = torch.squeeze(conv_theta)
            cac_theta = torch.squeeze(cac_theta)
            self.conv_b =  self.conv.bias.data 
            self.conv_b = self.conv_b.to(self.conv.weight.device)
            self.cdc_b =  self.cdc.bias.data 
            self.cdc_b = self.cdc_b.to(self.conv.weight.device)
            self.cac_b =  self.cac.bias.data 
            self.cac_b = self.cac_b.to(self.conv.weight.device)
            self.B = conv_theta * self.conv_b + cdc_theta * self.cdc_b + cac_theta * self.cac_b
        else:
            # self.conv_b = None
            self.B = None
        

    def test_forward(self,x):
        self.re_para()
        self.K = self.K.to(x.device)
        if self.B is not None:
            self.B = self.B.to(x.device)
        return F.conv2d(input=x, weight=self.K, bias=self.B, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)


if __name__ == "__main__":

    conv = dwconv_hwc(in_channels=30, out_channels=3, kernel_size=7, stride=1, padding=3, dilation=1, groups=3, bias=True).cuda()
    x = torch.randn(1, 30, 40, 40).cuda()
    y1 = conv(x)
    print(y1.shape)
    # w = conv.get_weight()
    # y2 = F.conv2d(x, w,  conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
    # print(y2)
    # print(y1 - y2)
    conv.eval()
    y2 = conv(x)
    print(y2.shape)
    print(torch.mean(y1 - y2))
    # print(y1 - y2)
