import numpy as np
import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import scipy

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import transformers
from qtorch import quant

def quantize_bfp(tensor, bits = 4, block_size = 128, along_rows=False):
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        tensor = quant.block_quantize(tensor, bits, dim = 0, rounding='nearest')
        tensor = tensor.reshape(*shape)
        tensor = tensor.transpose(-1,-2)
    return tensor

def quantize_sbfp(tensor, bits = 4, block_size = 128, along_rows=False):
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        scale = ((2**(bits-1) - 1.)  / torch.max(torch.abs(tensor), dim = -1, keepdims=True).values).half().float()
        #print(scale)
        tensor = torch.round(tensor * scale)
        #print(tensor)
        tensor /= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        #print(tensor)
        scale = ((2**(bits-1) - 1.)  / torch.max(torch.abs(tensor), dim = -1, keepdims=True).values).half().float()
        #print(scale)
        #print(scale.shape)
        tensor = torch.round(tensor * scale)
        #print(tensor)
        tensor /= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor.reshape(*shape).transpose(-1,-2)

class FCWrapper(nn.Linear):
    def __init__(self, mlp):
        #nn.Module()a.__init__()
        for attr, val in mlp.__dict__.items():
            #print(attr)
            self.__setattr__(attr, val)
        self.dtype = self.get_parameter('weight').data.dtype
        
        proj = self.get_parameter('weight').data.to(torch.float32)
        
        norm3 = torch.norm(proj,p=2, dim=0)
        self.order = torch.argsort(-norm3)  #largest first
        self.inverse_order = torch.empty_like(self.order)
        self.inverse_order[self.order] = torch.arange(self.order.shape[0], device=self.order.device)
        #proj = proj[:,self.order]
        proj = quantize_bfp(proj, 4, 128, True)
        #proj = proj[:, self.inverse_order].to(self.dtype)
        self.get_parameter('weight').data = proj.to(self.dtype)

        

def wrap_model(model):
    for idx in range(model.config.num_hidden_layers):
        # model.model.layers[idx].self_attn = LlamaAttentionWrapperQK(model.model.layers[idx].self_attn)
        model.model.decoder.layers[idx].fc2 = FCWrapper(model.model.decoder.layers[idx].fc2)

