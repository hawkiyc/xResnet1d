#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:34:18 2023
Latest Change on Wed May 03 09:56:10 2023

@author: hawkiyc
"""

#%%
'Import Libraries'

import torch
import torch.nn as nn

from params import *
from preprocessing import *

#%%
'Set Activation Function'
if __name__ == "__main__":
    
    stem_k, block_k = 11, 5
    activation=nn.ReLU(inplace=True)
    data_dim =12
    out_dim=5

#%%
'Bulid th Model'

class conv(nn.Module):
    
    def __init__(self, in_ch, out_ch, k_size = 25, stride = 1, 
                 drop_r = None, zero_batch_norm = False, 
                 bias = False, use_act_fun = True,
                 act_fun: nn.Module = activation):
        
        assert k_size % 2 == 1, 'kernel size shall be odd number'
        super(conv, self).__init__()
        self.conv1d = nn.Conv1d(in_ch, out_ch, k_size, stride,
                                padding=(k_size-1)//2, bias=bias,)
        self.batch_norm =nn.BatchNorm1d(out_ch)
        nn.init.constant_(self.batch_norm.weight,
                          0. if zero_batch_norm else 1.)
        self.act_fun = act_fun
        self.drop_r, self.drop = drop_r, nn.Dropout(drop_r
                                                    ) if drop_r else None
        self.use_act_fun = use_act_fun
        
    def forward(self, x):
        
        x = self.conv1d(x)
        x = self.batch_norm(x)
        if self.use_act_fun:
            x = self.act_fun(x)
        if self.drop_r:
            x = self.drop(x)
        
        return x

class XResNetBlock(nn.Module):
    
    def __init__(self, expansion, in_ch, between_ch, k = 9, 
                 stride = 1, b_verbose = None,
                 act_fun: nn.Module = activation):
        
        assert expansion in [1,4] , 'expansion shall be 1 or 4'
        super(XResNetBlock, self).__init__()
        
        in_ch = in_ch * expansion
        out_ch = between_ch * expansion
        
        if expansion == 1:
            
            layers = [conv(in_ch, between_ch, 
                           k, stride=stride),
                      conv(between_ch, out_ch, k, 
                           zero_batch_norm=True, 
                           use_act_fun = False)]
        
        else:
            
            layers = [conv(in_ch, between_ch, 1),
                      conv(between_ch, between_ch, 
                           k, stride = stride,),
                      conv(between_ch, out_ch, 1,
                           zero_batch_norm=True, 
                           use_act_fun = False)]
        
        self.xres_block = nn.ModuleList(layers)
        
        self.res_conv = conv(in_ch,out_ch,1,use_act_fun=False
                             ) if in_ch != out_ch else None
        self.res_pool = nn.AvgPool1d(2, ceil_mode=True
                                     ) if stride != 1 else None
        self.act_fun = act_fun
        self.b_verbose = b_verbose if b_verbose else None
        
    def forward(self, x):
        
        identity = x
        
        for l in self.xres_block:
            x = l(x)
            print('res_torch_size:', x.shape) if self.b_verbose else None
        
        identity = self.res_pool(identity) if self.res_pool else identity
        identity = self.res_conv(identity) if self.res_conv else identity
        print('identity_torch_size:', x.shape) if self.b_verbose else None
        
        x += identity
        x = self.act_fun(x)
        
        return x

class ConcatPool(nn.Module):
    
    def __init__(self, dim=1):
        
        super().__init__()
        
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        self.dim = dim

    def forward(self, x):
        
        maxpooled = self.maxpool(x).squeeze(self.dim)
        avgpooled = self.avgpool(x).squeeze(self.dim)
        
        return torch.cat((maxpooled, avgpooled), dim=self.dim)

class XResNet1d(nn.Module):
    
    def __init__(self, expansion, num_layers, stem_k, 
                 block_k, in_ch=data_dim, c_out=out_dim,
                 model_drop_r = None, verbose = False,
                 b_verbose = False, original_f_number = False,
                 fc_drop = None):
        
        super(XResNet1d, self).__init__()
        
        stem_filters = [in_ch, 32, 32, 64]
        
        stem = [conv(stem_filters[i], stem_filters[i+1], k_size = stem_k,
                     stride=2 if i==0 else 1, drop_r = model_drop_r,
                     ) for i in range(3)]
        self.stem = nn.ModuleList(stem)
        
        self.stem_pool = nn.MaxPool1d(3,2, padding=1)
        self.model_drop_r = nn.Dropout(model_drop_r
                                       ) if model_drop_r else None
        self.b_verbose = b_verbose if b_verbose else None
        
        if original_f_number:
            
            block_filters = [64//expansion] + [(o) for o in [
                64,128,256,512] +[256]*(len(num_layers)-4)]
        else:
            
            block_filters = [64//expansion] + [(o) for o in [
                64,64,64,64] +[32]*(len(num_layers)-4)]
        
        self.block_k = block_k
        block = [self.make_layers(expansion, block_filters[i],
                                  block_filters[i+1], n_blocks=l, 
                                  stride=1 if i==0 else 2, 
                                  ) for i, l in enumerate(num_layers)]
        self.block = nn.ModuleList(block)
        
        self.concat_pool = ConcatPool()
        self.fc1 = nn.Linear(block_filters[-1]*expansion*2,128)
        self.fc_batch_norm = nn.BatchNorm1d(128)
        self.fc_drop = nn.Dropout(fc_drop) if fc_drop else None
        self.fc_out = nn.Linear(128, c_out)
        self.expansion = expansion
        self.verbose = verbose
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
    
    def make_layers(self, expansion, n_inputs,  
                    n_filters, n_blocks, stride, 
                    ):
        
        sub_block = []
        
        if self.model_drop_r:
            
            for i in range(n_blocks):
                sub_block.append(XResNetBlock(expansion, 
                                              n_inputs if i==0 else n_filters, 
                                              n_filters, self.block_k, 
                                              stride if i==0 else 1,
                                              b_verbose = self.b_verbose 
                                                  if self.b_verbose else None,
                                              ))
                sub_block.append(self.model_drop_r)
        
        else:
            sub_block = [XResNetBlock(expansion, 
                                      n_inputs if i==0 else n_filters, 
                                      n_filters, self.block_k, 
                                      stride if i==0 else 1,
                                      b_verbose = self.b_verbose 
                                          if self.b_verbose else None,
                                      )for i in range(n_blocks)]
        
        return nn.Sequential(*sub_block)
    
    def forward(self, x):
        
        for l in self.stem:
            x = l(x)
            print('stem_torch_size:', x.shape) if self.verbose else None
        
        x = self.stem_pool(x)
        
        for b in self.block:
            x = b(x)
            print('block_torch_size:', x.shape) if self.verbose else None
        
        x = self.concat_pool(x)
        print('concat_pool_torch_size:', x.shape) if self.verbose else None
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc_batch_norm(x)
        x = self.fc_drop(x) if self.fc_drop else x
        x = self.fc_out(x)
        
        return x

#%%

def xresnet1d18(**kwargs): 
    return XResNet1d(1, [2, 2,  2, 2], stem_k, block_k, **kwargs)
def xresnet1d34(**kwargs): 
    return XResNet1d(1, [3, 4,  6, 3], stem_k, block_k, **kwargs)
def xresnet1d50(**kwargs): 
    return XResNet1d(4, [3, 4,  6, 3], stem_k, block_k, **kwargs)
def xresnet1d101(**kwargs): 
    return XResNet1d(4, [3, 4, 23, 3], stem_k, block_k, **kwargs)
def xresnet1d152(**kwargs): 
    return XResNet1d(4, [3, 8, 36, 3], stem_k, block_k, **kwargs)
def xresnet1d18_deep(**kwargs): 
    return XResNet1d(1, [2,2,2,2,1,1], stem_k, block_k, **kwargs)
def xresnet1d34_deep(**kwargs): 
    return XResNet1d(1, [3,4,6,3,1,1], stem_k, block_k, **kwargs)
def xresnet1d50_deep(**kwargs): 
    return XResNet1d(4, [3,4,6,3,1,1], stem_k, block_k, **kwargs)
def xresnet1d18_deeper(**kwargs): 
    return XResNet1d(1, [2,2,1,1,1,1,1,1], stem_k, block_k, **kwargs)
def xresnet1d34_deeper(**kwargs): 
    return XResNet1d(1, [3,4,6,3,1,1,1,1], stem_k, block_k, **kwargs)
def xresnet1d50_deeper(**kwargs): 
    return XResNet1d(4, [3,4,6,3,1,1,1,1], stem_k, block_k, **kwargs)

#%%
'Test'

if __name__ == "__main__":
    
    from torch.autograd import Variable
    
    m = xresnet1d101(verbose=True, model_drop_r=.3)
    out = m(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    
    m1 = xresnet1d101(verbose=True, original_f_number=True)
    out1 = m1(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    
    m2 = xresnet1d50_deeper(verbose=True, )
    out2 = m2(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    
    m3 = xresnet1d50(verbose=True, model_drop_r=.3)
    out3 = m3(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    
    m4 = xresnet1d18(verbose=True, )
    out4 = m4(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    
    m5 = xresnet1d18(verbose=True, original_f_number=True, model_drop_r=.3)
    out5 = m5(Variable(torch.randn(10, 12, 5000)))
    print("==========================")
    