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

def quantize_sfp(tensor, bits = 4, block_size = 128, along_rows=False):
    assert(bits==4)
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        scale = ((torch.max(torch.abs(tensor), dim = -1, keepdims=True).values)/6.).float()
        scale = quant.float_quantize(scale, 5, 4, rounding='nearest')
        tensor = quant.float_quantize( (tensor / scale).float(), 2, 1, rounding='nearest')
        tensor *= scale
        return tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        #print(shape)
        pad = shape[-1]
        pad = ( (pad + block_size - 1) // block_size) * block_size - pad
        #print(pad)
        tensor =  torch.nn.functional.pad(tensor, (0, pad), value=0.)
        tensor = tensor.reshape(-1, block_size)
        scale = ((torch.max(torch.abs(tensor), dim = -1, keepdims=True).values)/6.).float()
        scale = quant.float_quantize(scale, 5, 4, rounding='nearest')
        tensor = quant.float_quantize( (tensor / scale).float(), 2, 1, rounding='nearest')
        tensor *= scale
        #padded_shape = shape[:-1] + [shape[-1] + pad]
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        #print (shape)
        #print (tensor.shape)
        return tensor[..., :shape[-1]].reshape(*shape).transpose(-1,-2)

def quantize_mxfp(tensor, bits = 4, block_size = 128, along_rows=False):
    assert(bits==4)
    if along_rows:
        shape = tensor.shape
        tensor = tensor.reshape(-1, block_size)
        #scale = ((2**(bits-1) - 1.)  / torch.max(torch.abs(tensor), dim = -1, keepdims=True).values).half().float()
        scale = ((torch.max(torch.abs(tensor), dim = -1, keepdims=True).values)).float()
        scale = 2**(torch.ceil(torch.log2(scale))-3)

        #print(scale)
        tensor = quant.float_quantize( (tensor / scale).float(), 2, 1, rounding='nearest')
        #print(tensor)
        tensor *= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor.reshape(*shape)
    else:
        tensor = tensor.transpose(-1,-2)
        shape = tensor.shape
        pad = shape[-1]
        pad = ( (pad + block_size - 1) // block_size) * block_size - pad
        #print(pad)
        tensor =  torch.nn.functional.pad(tensor, (0, pad), value=0.)
        tensor = tensor.reshape(-1, block_size)
        #print(tensor)
        #scale = ((2**(bits-1) - 1.)  / torch.max(torch.abs(tensor), dim = -1, keepdims=True).values).half().float()
        scale = ((torch.max(torch.abs(tensor), dim = -1, keepdims=True).values)).float()
        scale = 2**(torch.ceil(torch.log2(scale))-3)
        #print(scale)
        #print(scale.shape)
        tensor = quant.float_quantize( (tensor / scale).float(), 2, 1, rounding='nearest')
        #print(tensor)
        tensor *= scale
        # dim along which quantization is independent
        # e.g. dim = -1 means each column has its own scaling factor
        return tensor[..., :shape[-1]].reshape(*shape).transpose(-1,-2)


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

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttentionWrapperQK(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attention, quantize=False, reorder=False, pnorm=2, block_size=128):
        super().__init__()
        for attr, val in attention.__dict__.items():
            #print(attr)
            self.__setattr__(attr, val)
        #self.Kactivation = None

        self.quantize = quantize
        self.reorder = reorder
        self.block_size=block_size

        self.dtype = self.k_proj.get_parameter('weight').data.dtype
        dtype = self.dtype
        v_proj = self.v_proj.get_parameter('weight').data.to(torch.float32)

        if reorder:
            norm3 = torch.norm(v_proj, p=pnorm,  dim=-1)
            self.order = torch.argsort(-norm3.view(self.num_heads,self.head_dim))  #largest first
            self.inverse_order = torch.empty_like(self.order)
            self.inverse_order[torch.arange(self.num_heads).unsqueeze(-1), self.order] = torch.arange(self.head_dim, device=self.order.device).unsqueeze(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if self.quantize:
            #query_states = quantize_bfp(query_states.float(), 8, self.block_size, True).to(self.dtype)
            key_states = quantize_bfp(key_states.float(), 4, self.block_size, True).to(self.dtype)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class GemmaRotaryEmbeddingPhase(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
        self.phase = 0.3926990817

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1) + self.phase
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



'''class RotaryWrapper(nn.Module):
    def __init__(self, rotary_emb, phase=0.3926990817*2):
        super().__init__()
        for attr, val in rotary_emb.__dict__.items():
            print(attr)
            self.__setattr__(attr, val)
        self.phase = phase
        #print(self.inv_freq.device)
        #self.phase = (torch.arange(self.dim//2, device = self.inv_freq.device) > ((self.dim//2) * 0.9) ).half()
        #self.phase *= phase
        #self.phase = torch.cat((self.phase, self.phase), dim=-1)
        print(self.phase)
        #print(self.phase.device)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1) + self.phase
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)'''



def wrap_model(model, quantize = False, reorder=False, pnorm=2, block_size=16, turn=False):
    for idx in range(model.config.num_hidden_layers):
        model.model.layers[idx].self_attn = GemmaAttentionWrapperQK(model.model.layers[idx].self_attn, quantize, reorder, pnorm, block_size)
        if turn:
            model.model.layers[idx].self_attn.rotary_emb = GemmaRotaryEmbeddingPhase(model.model.layers[idx].self_attn.head_dim,
            max_position_embeddings=model.model.layers[idx].self_attn.max_position_embeddings,
            base=model.model.layers[idx].self_attn.rope_theta)
