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


def calculate_norm(k_proj, head_dim=128, do_calc=False):
    if do_calc:
        reshaped = k_proj.view(-1, head_dim, k_proj.shape[-1])
        norm = torch.norm(reshaped, dim=-1)
        #return (norm[::2] + norm[1::2]).repeat_interleave(2) / 2.
        return (((norm[:, : head_dim//2] + norm[:, head_dim//2:])/2.).repeat(1,2).view(-1))
    else:
        return torch.ones_like(k_proj[:,0])

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

def permute_W(proj, head_dim=128):
    reshaped = proj.view(-1, head_dim, proj.shape[-1])
    result = torch.empty_like(reshaped)
    result[:, ::2, :] = reshaped[:, :head_dim//2, :]
    result[:, 1::2, :] = reshaped[:, head_dim//2:, :]
    return result.view(*proj.shape)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    result = torch.empty_like(x)
    result[..., :: 2] = - x[..., 1 :: 2]
    result[..., 1 :: 2] = x[..., :: 2]
    return result

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class LlamaRotaryEmbeddingNew(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = freqs.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


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

class LlamaAttentionWrapperQK(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, attention):
        super().__init__()
        for attr, val in attention.__dict__.items():
            #print(attr)
            self.__setattr__(attr, val)
        #self.Kactivation = None

        self.dtype = self.k_proj.get_parameter('weight').data.dtype
        dtype = self.dtype
        k_proj = self.k_proj.get_parameter('weight').data.to(torch.float32)
        q_proj = self.q_proj.get_parameter('weight').data.to(torch.float32)
        
        norm2 = calculate_norm(k_proj, self.head_dim)
        #print(norm2)

        #self.k_proj.get_parameter('weight').data = quantize_sbfp( (k_proj / norm2.view(-1,1)), 4, 64).to(dtype)
        self.k_proj.get_parameter('weight').data = permute_W((k_proj / norm2.view(-1,1)).to(dtype), self.head_dim)
        self.q_proj.get_parameter('weight').data = permute_W((q_proj * norm2.view(-1,1)).to(dtype), self.head_dim)
        self.rope_theta=10000
        k_proj = self.k_proj.get_parameter('weight').data.to(torch.float32)

        norm3 = torch.norm(k_proj, dim=-1)
        self.order = torch.argsort(-norm3.view(self.num_heads,self.head_dim))  #largest first
        '''
        order_list = [ [] for _ in range(self.num_heads) ]
        for head_num in range(self.num_heads):
            used = set()
            for idxtensor in order[head_num]:
                idx = idxtensor.item()
                if idx not in used:
                    used.add(idx)
                    order_list[head_num].append(idx)

                    if idx % 2 == 0:
                        another = idx + 1
                    else:
                        another = idx - 1

                    used.add(another)
                    order_list[head_num].append(another)

        self.order = torch.LongTensor(order_list, device=order.device)
        '''
        self.inverse_order = torch.empty_like(self.order)
        self.inverse_order[torch.arange(self.num_heads).unsqueeze(-1), self.order] = torch.arange(self.head_dim, device=self.order.device).unsqueeze(0)


        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbeddingNew(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            raise ValueError("Scalings are not implemented") 

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split((self.num_heads * self.head_dim) // self.pretraining_tp, dim=0)
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        #self.kact_log_unquant.append(key_states.cpu().numpy())
        #print('lol')
        #self.kact_log.append(key_states.cpu().numpy())

        #self.Kactivation = key_states.cpu()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        #self.Kactivation = key_states.cpu()
        ind1=torch.arange(bsz, device=key_states.device).view(-1,1,1,1)
        ind2=torch.arange(self.num_heads, device=key_states.device).view(1,-1,1,1)
        ind3=torch.arange(q_len, device=key_states.device).view(1,1,-1,1)
        ind4=self.order.view(1, self.num_heads, 1, self.head_dim)
        ind4inv = self.inverse_order.view(1, self.num_heads, 1, self.head_dim)
        #query_states = query_states[ind1,ind2,ind3,ind4]
        #key_states = key_states[ind1,ind2,ind3,ind4]

        #query_states = quantize_bfp(query_states.float(), 8, 128, True).to(self.dtype)
        #key_states = quantize_sbfp(key_states.float(), 4, 128, True).to(self.dtype)
	
        #query_states = query_states[ind1,ind2,ind3,ind4inv]
        #key_states = key_states[ind1,ind2,ind3,ind4inv]


        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)


        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def wrap_model(model):
    for idx in range(32):
        model.model.layers[idx].self_attn = LlamaAttentionWrapperQK(model.model.layers[idx].self_attn)
