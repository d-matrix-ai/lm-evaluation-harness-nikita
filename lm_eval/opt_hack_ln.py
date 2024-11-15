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
from transformers import Cache
from qtorch import quant

def calculate_scaler(llama_decoder, prev_lnorm_weight=1.): # prev_lnorm_weight):
    with torch.no_grad():
        W_V = llama_decoder.self_attn.v_proj.get_parameter('weight')
        P = llama_decoder.self_attn.out_proj.get_parameter('weight')
        norm = P@W_V
        assert(norm.shape[0] == norm.shape[1])
        norm += torch.eye(norm.shape[0], device=norm.device)
        #prev_lnorm_weight = llama_decoder.input_layernorm.get_parameter('weight')
        norm *= prev_lnorm_weight
        norm = norm.to(torch.float32)
        return torch.linalg.norm(norm, 'fro').item()



def calculate_scaler_output(layer):
    with torch.no_grad():
        prev_lnorm_weight = layer.self_attn_layer_norm.get_parameter('weight').to(torch.float32)
        F1 = layer.fc1.get_parameter('weight').to(torch.float32)
        F2 = layer.fc2.get_parameter('weight').to(torch.float32)

        norm = F2@F1
        assert(norm.shape[0] == norm.shape[1])
        norm += torch.eye(norm.shape[0], device=norm.device)
        norm *= prev_lnorm_weight
        
        return torch.linalg.norm(norm, 'fro').item()


class LayerNormWrapper(torch.nn.Module):
    def __init__(self, layerNorm, scale):
        #scale = 1
        super().__init__()
        #for attr, val in layerNorm.__dict__.items():
        #    self.__setattr__(attr, val)
        self.layerNorm = layerNorm
        self.layerNorm.eps /= scale**2
        self.scale = scale
        self.weight = layerNorm.weight
        #self.scale = 2. #scale
        #self.summ_vals = []
        #self.summ_unscaled = []
        #self.eps /= scale
    
    def forward(self, hidden_sts):
        #pow2 = hidden_states.to(torch.float16).pow(2)
        #summ = pow2.sum(-1, keepdim=True)
        #self.summ_vals.extend(list((summ.view(-1)/(self.scale**2)).cpu()))
        #self.summ_unscaled.extend(list((summ.view(-1)).cpu()))
        #hidden_states = hidden_states.to(torch.float32)
        hidden_states = hidden_sts / self.scale
        print(hidden_states.dtype)
        mean = hidden_states.mean(-1, keepdim=True)
        hidden_states -= mean
        var = hidden_states.pow(2).sum(-1, keepdim=True)
        print( (torch.min(var).item(), torch.max(var).item()))
        var /= hidden_states.shape[-1]
        var += self.layerNorm.eps
        hidden_states *= torch.rsqrt(var)
        hidden_states *= self.layerNorm.weight
        hidden_states += self.layerNorm.bias
        return hidden_states.to(torch.float16)


def wrap_model(model, proxy_ln = False):
    if not proxy_ln:
        return model
    prev_lnorm_weight = 1.
    sc2 = 1.
    for idx in range(model.config.num_hidden_layers):
        model.layers[idx].self_attn_layer_norm = LayerNormWrapper(model.layers[idx].self_attn_layer_norm, sc2)
        sc = calculate_scaler(model.layers[idx], prev_lnorm_weight)
        model.layers[idx].final_layer_norm = LayerNormWrapper(model.layers[idx].final_layer_norm, sc)

        print("pre attn scaling : {}, pre output scaling: {}".format(sc2, sc))
        sc2 = calculate_scaler_output(model.layers[idx])

        prev_lnorm_weight = model.layers[idx].final_layer_norm.get_parameter('weight').to(torch.float32)

    return model
