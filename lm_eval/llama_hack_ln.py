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


def calculate_scaler(llama_decoder): # prev_lnorm_weight):
    with torch.no_grad():
        W_V = llama_decoder.self_attn.v_proj.get_parameter('weight').to(torch.float32)
        P = llama_decoder.self_attn.o_proj.get_parameter('weight').to(torch.float32)
        norm = P@W_V
        assert(norm.shape[0] == norm.shape[1])
        norm += torch.eye(norm.shape[0], device=norm.device)
        prev_lnorm_weight = llama_decoder.input_layernorm.get_parameter('weight').to(torch.float32)
        norm *= prev_lnorm_weight
        norm = norm.to(torch.float32)
        return torch.linalg.norm(norm, 'fro').item()

#careful,
#scale used at the next level

def calculate_scaler_input(llama_decoder):
    with torch.no_grad():
        prev_lnorm_weight = llama_decoder.post_attention_layernorm.get_parameter('weight').to(torch.float32)
        W_gate = llama_decoder.mlp.gate_proj.get_parameter('weight').to(torch.float32)
        W_up = llama_decoder.mlp.up_proj.get_parameter('weight').to(torch.float32)
        W_down = llama_decoder.mlp.down_proj.get_parameter('weight').to(torch.float32)
        #norm = W_down@( (W_gate*prev_lnorm_weight)*(W_up*prev_lnorm_weight))

        norm = torch.linalg.norm(W_down @ (W_up*prev_lnorm_weight) , 'fro') * \
                torch.linalg.norm ( W_gate*prev_lnorm_weight, 2)
        #norm *= prev_lnorm_weight**2
        #assert(norm.shape[0] == norm.shape[1])
        #norm += torch.eye(norm.shape[0], device=norm.device) * prev_lnorm_weight
        
        #norm = norm.to(torch.float32)
        #return torch.linalg.norm(norm, 'fro')
        return norm.item()

class LlamaRMSNormWrapper(nn.Module):
    def __init__(self, llama_rms, scale):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        for attr, val in llama_rms.__dict__.items():
            self.__setattr__(attr, val)

        
        self.variance_epsilon /= (scale**2)
        self.variance_epsilon = self.variance_epsilon.half()
        self.scale = scale
        #print(type(scale))


    def forward(self, hidden_states):
        #hidden_states = hidden_states / self.scale
        input_dtype = hidden_states.dtype
        #hidden_states = hidden_states.to(torch.float32)
        pow2 = hidden_states.pow(2)
        summ = pow2.sum(-1, keepdim=True)
        
        variance = pow2.mean(-1, keepdim=True)
        #print(variance.dtype)

        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def wrap_model(model, proxy_ln=False):
    if not proxy_ln:
        return model

    sc = 1.
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].input_layernorm = LlamaRMSNormWrapper(model.model.layers[idx].input_layernorm, sc)


        sc2 = calculate_scaler(model.model.layers[idx])
        #print(sc2)
        model.model.layers[idx].post_attention_layernorm = LlamaRMSNormWrapper(model.model.layers[idx].post_attention_layernorm, sc2)

        print("input scaling: {}, post attn scaling : {}".format(sc, sc2))
        sc = calculate_scaler_input(model.model.layers[idx])
        print(sc)
        #model.encoder.layer[idx].attention.output = OutputWrapper(model.encoder.layer[idx].attention.output, 33.)
        #model.encoder.layer[idx].output = OutputWrapper(model.encoder.layer[idx].output)
        #prev_lnorm_weight = model.encoder.layer[idx].output.LayerNorm.get_parameter('weight')
    #return model
    print("last norm scaling: {}".format(sc))
    model.model.norm = LlamaRMSNormWrapper(model.model.norm, sc)
    return model

