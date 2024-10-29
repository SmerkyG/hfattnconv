# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Callable

from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers import AutoConfig, AutoModelForCausalLM

from fla.ops.gla.chunk import chunk_gla, ChunkGLAFunction
from pydoc import locate

from transformers.utils import logging
logger = logging.get_logger(__name__)

def fla_chunk_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,  # log decay
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    g = g.float()
    initial_state = None
    output_final_state = False
    o, final_state = ChunkGLAFunction.apply(q, k, v, g, scale, initial_state, output_final_state)
    return o, final_state

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

class RWKV6Attention(nn.Module):
    def __init__(self, config: Any, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=getattr(config, 'attention_output_bias', config.attention_bias))

        self.gate = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        # start gate out with no effect
        nn.init.zeros_(self.gate.weight)
        with torch.no_grad():
            self.gate.bias[:] = 1.227

        n_layer = self.config.num_hidden_layers
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_idx

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            ddd = torch.zeros(1, 1, n_embd)
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_k = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_v = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_w = nn.Parameter(torch.zeros_like(ddd))
            self.time_maa_g = nn.Parameter(torch.zeros_like(ddd))

            D_MIX_LORA = 32 if n_embd < 4096 else 64
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            # RWKV-6
            decay_speed = torch.ones(dim_att)
            for n in range(dim_att):
                decay_speed[n] = -6 + 5 * (n / (dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,dim_att))
            D_DECAY_LORA = 64 if n_embd < 4096 else 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, dim_att).uniform_(-0.01, 0.01))
            # tmp = torch.zeros(dim_att)
            # for n in range(dim_att):
            #     zigzag = ((n + 1) % 3 - 1) * 0.1
            #     tmp[n] = ratio_0_to_1 * (1 - (n / (dim_att - 1))) + zigzag
            # self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

    def segsum(self, w_log): # B H L 1
        w_log_cumsum = torch.cumsum(w_log, dim=-2) # (B, H, L, 1)
        w_mask = torch.exp((w_log_cumsum - w_log_cumsum.mT).tril()).tril() # (B, H, L, L)
        return w_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, hidden_dim = hidden_states.size()
        H = self.num_heads

        x = hidden_states
        dxprev = torch.nn.functional.pad(x, (0, 0, 1, -1)) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(bsz*q_len, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), bsz, q_len, hidden_dim)

        mr, mk, mv, mw, mg = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        xg = x + dxprev * (self.time_maa_g + mg)

        query_states = self.q_proj(xr)
        key_states = self.k_proj(xk)
        value_states = self.v_proj(xv)
        decay_states = (self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2).to(query_states.dtype)
        gate_states = F.silu(self.gate(xg))

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        decay_states = decay_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        decay_states_log = -decay_states.float().exp()
        #decay_states_log = decay_states_log.clamp(-5) # FIXME - is this necessary?
        key_states = (key_states * (1 - decay_states_log.exp())).to(key_states.dtype)

        query_states = query_states.to(value_states.dtype)
        key_states = key_states.to(value_states.dtype)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_weights = torch.empty(0, device=x.device)

        #attn_output = fla_chunk_simple_gla(query_states, key_states, value_states, decay_states_log.view(bsz, self.num_heads, q_len))[0]
        attn_output = fla_chunk_gla(query_states, key_states, value_states, decay_states_log)[0]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output * gate_states)

        return attn_output, attn_weights, past_key_value

class AttentionDistillationWrapper(nn.Module):
    def __init__(self, original_self_attn:nn.Module, ReplacementSelfAttentionType:Callable, model_config:Any, attention_distillation_stage:int):
        super().__init__()
        self.teacher_attn = original_self_attn
        self.student_attn = ReplacementSelfAttentionType(model_config, original_self_attn.layer_idx)
        assert attention_distillation_stage == 2
        self.attention_distillation_stage = attention_distillation_stage

        # copy in teacher's starting parameter values into student during stage 2
        student_params_dict = dict(self.student_attn.named_parameters())
        for n, p in self.teacher_attn.named_parameters():
            if n in student_params_dict:
                student_params_dict[n].requires_grad_(False)
                student_params_dict[n].copy_(p)
                student_params_dict[n].requires_grad_(p.requires_grad)

    def forward(self, 
        # hidden_states: torch.Tensor,
        # attention_mask: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.LongTensor] = None,
        # past_key_value: Optional[Cache] = None,
        # output_attentions: bool = False,
        # use_cache: bool = False,
        # cache_position: Optional[torch.LongTensor] = None,
        # position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        #if self.attention_distillation_stage == 2:
        # even though we must return our special loss in as 'attentions', we don't need to obtain the actual attentions from the model for stage 2, only stage 1
        kwargs['output_attentions'] = False

        # NOTE - instead of returning attentions here we return a special attention loss
        student_outputs = self.student_attn(*args, **kwargs)
        teacher_outputs = self.teacher_attn(*args, **kwargs)
        assert self.attention_distillation_stage == 2
        # special attention loss is the vector norm of the difference between the student and teacher attn outputs
        student_hidden_states = student_outputs[0]
        teacher_hidden_states = teacher_outputs[0]
        special_attn_loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states, dim=-1).mean() * (teacher_hidden_states[0].size(-1) ** -0.5)

        return (teacher_outputs[0], special_attn_loss, ) + teacher_outputs[2:]

def load_and_patch_model_with_attention_replacement(model_path:str, attn_classes_path:str, ReplacementSelfAttentionType:Callable, attention_distillation_stage:int):
    model_config = AutoConfig.from_pretrained(model_path)

    # FIXME - hardcoded for now, but it'd be great if we could specify this in data somewhere per model type (or even analyze the weights to see)
    # NOTE - when loading a custom Qwen2RWKV model we don't need to set model_config.attention_bias and model_config.attention_output_bias, because the model config contains it
    if 'Qwen/Qwen' in model_path:
        model_config.attention_bias = True
        model_config.attention_output_bias = False

    # replace attention classes
    attn_classes_dict = locate(attn_classes_path)
    attn_classes_dict_original_copy:dict = attn_classes_dict.copy()
    assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
    if attention_distillation_stage >= 3:
        for key in list(attn_classes_dict.keys()):
            attn_classes_dict[key] = ReplacementSelfAttentionType

    model = AutoModelForCausalLM.from_pretrained(model_path, config=model_config)

    # reset attention classes for upcoming teacher module's use
    for key, value in attn_classes_dict_original_copy.items():
        attn_classes_dict[key] = value

    # patch model
    if attention_distillation_stage == 2:
        # requires_grad_(False) on entire model, so it acts as teacher
        model.requires_grad_(False)

        # monkeypatch conditionally executed student attention replacements (which do require grad)
        for layer in model.model.layers:
            layer.self_attn = AttentionDistillationWrapper(layer.self_attn, ReplacementSelfAttentionType, model_config, attention_distillation_stage)

        # student attention replacements do require grad in both stages 1 and 2
        for layer in model.model.layers:
            student_attn = layer.self_attn.student_attn
            student_attn.requires_grad_(True)

    elif attention_distillation_stage >= 3:
        if model_config.tie_word_embeddings:
            # copy untied embeddings
            model.get_output_embeddings().weight = nn.Parameter(model.get_input_embeddings().weight.clone())
            # untie the embeddings in the config, too
            model_config.tie_word_embeddings = False

    return model
