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
"""PyTorch RWKV7Qwen2 model."""

import math
import inspect
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.utils.checkpoint
from torch import nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from .configuration_rwkv7qwen2 import RWKV7Qwen2Config

# MIT License

# Copyright (c) 2024 Songlin Yang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Copyright (c) 2024, Johan Sokrates Wind

import torch as th
import triton
import triton.language as tl

@triton.jit
def IND3(a,b,c,nb,nc):
    return (a*nb+b)*nc+c
@triton.jit
def IND4(a,b,c,d,nb,nc,nd):
    return ((a*nb+b)*nc+c)*nd+d
@triton.jit
def IND5(a,b,c,d,e,nb,nc,nd,ne):
    return (((a*nb+b)*nc+c)*nd+d)*ne+e

@triton.jit
def _prod(a,b): return a*b

# inv(I-A) where A is a strictly lower triangular nxn matrix
@triton.jit
def tri_minv(A, n:tl.constexpr, prec:tl.constexpr):
    i = tl.arange(0,n)
    prod = (i[None,:]==i[:,None]).to(tl.float32)
    for j in range(n-1):
        prod += tl_dot(prec, prod, (A*((i[None,:]==j)*(i[:,None]>i[None,:]))).trans())
    return prod.trans()

@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def fw_attn_triton(w_,q_,k_,v_,a_,b_, s0_,y_,s_,sT_, wq_,wa_,kwi_,bwi_,fw_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)
    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            state = tl.load(s0_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(s_+IND5(bi,hi,0,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))

    for t0 in range(T//dT):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            wa_state = tl.zeros((dT,dC), tl.float32)
            wq_state = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                wa_state += tl_dot(prec, wa, state.trans())
                wq_state += tl_dot(prec, wq, state.trans())

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            yy = tl_dot(prec, qk, sv) + tl_dot(prec, qb, u) + wq_state
            tl.store(y_+IND4(bi,t,hi,i, T,H,C), yy.to(tl.bfloat16))

            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C))

                state = state * fw + tl_dot(prec, sv.trans(), kwi*fw) + tl_dot(prec, u.trans(), bwi*fw)

                if t0+1 < T//dT:
                    tl.store(s_+IND5(bi,hi,t0+1,i.trans(),j, H,T//dT,C,C), state.to(tl.float32))
                else:
                    tl.store(sT_+IND4(bi,hi,i.trans(),j, H,C,C), state.to(tl.bfloat16))


@triton.autotune(configs=[triton.Config({'dC': dC}, num_stages=1) for dC in [16,32,64]], key=['T','H','C','dT','prec'])
@triton.jit
def bw_attn_triton(w_,q_,k_,v_,a_,b_, dy_,s_,dsT_,ds_, dw_,dq_,dk_,dv_,da_,db_,ds0_, wq_,wa_,kwi_,bwi_,fw_,u_,dab_u_, B:tl.constexpr,T:tl.constexpr,H:tl.constexpr,C:tl.constexpr,dT:tl.constexpr, prec:tl.constexpr, dC:tl.constexpr):
    tl.static_assert(C%dC == 0)
    bi = tl.program_id(1)
    hi = tl.program_id(0)

    for i0 in range(0,C,dC):
        i = i0+tl.arange(0,dC)[None,:]
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            dstate = tl.load(dsT_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
            tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))

    for t0 in range(T//dT-1,-1,-1):
        dt = tl.arange(0,dT)[:,None]
        t = t0*dT+dt
        tl.debug_barrier()
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]
            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sq = tl.load(q_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sk = tl.load(k_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            w = (-sw.exp()).exp()
            fw = tl.reduce(w, 0, _prod, keep_dims=True)
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            wq = sq * incl_pref
            wa = sa * non_incl_pref
            kwi = sk * inv_incl_pref
            bwi = sb * inv_incl_pref

            tl.store(wq_+IND4(bi,hi,dt,j, H,dT,C), wq.to(tl.float32))
            tl.store(wa_+IND4(bi,hi,dt,j, H,dT,C), wa.to(tl.float32))
            tl.store(kwi_+IND4(bi,hi,dt,j, H,dT,C), kwi.to(tl.float32))
            tl.store(bwi_+IND4(bi,hi,dt,j, H,dT,C), bwi.to(tl.float32))
            tl.store(fw_+IND3(bi,hi,j, H,C), fw.to(tl.float32))
        tl.debug_barrier()

        ab = tl.zeros((dT,dT), tl.float32)
        ak = tl.zeros((dT,dT), tl.float32)
        qb = tl.zeros((dT,dT), tl.float32)
        qk = tl.zeros((dT,dT), tl.float32)
        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            sa = tl.load(a_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            sb = tl.load(b_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)

            ab += tl_dot(prec, wa, bwi.trans())
            ak += tl_dot(prec, wa, kwi.trans())
            qb += tl_dot(prec, wq, bwi.trans())
            qk += tl_dot(prec, wq, kwi.trans())

        mask1 = (t > t.trans())
        mask2 = (t >= t.trans())
        ab *= mask1
        ak *= mask1
        qb *= mask2
        qk *= mask2

        ab_inv = tri_minv(ab, dT, prec)

        dab = tl.zeros((dT,dT), tl.float32)
        dak = tl.zeros((dT,dT), tl.float32)
        dqb = tl.zeros((dT,dT), tl.float32)
        dqk = tl.zeros((dT,dT), tl.float32)

        tl.debug_barrier()
        for i0 in range(0,C,dC):
            i = i0+tl.arange(0,dC)[None,:]
            wa_state = tl.zeros((dT,dC), tl.float32)
            bwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            kwi_dw_dstate = tl.zeros((dT,dC), tl.float32)
            for j0 in range(0,C,dC):
                j = j0+tl.arange(0,dC)[None,:]
                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
                fw = tl.load(fw_+IND3(bi,hi,j, H,C))

                wa_state += tl_dot(prec, wa, state.trans())
                bwi_dw_dstate += tl_dot(prec, bwi*fw, dstate.trans())
                kwi_dw_dstate += tl_dot(prec, kwi*fw, dstate.trans())

            sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
            sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

            ab_u = tl_dot(prec, ak, sv) + wa_state
            u = tl_dot(prec, ab_inv, ab_u)
            du = tl_dot(prec, qb.trans(), sdy) + bwi_dw_dstate
            dab_u = tl_dot(prec, ab_inv.trans(), du)

            tl.store(u_+IND4(bi,hi,dt,i, H,dT,C), u.to(tl.float32))
            tl.store(dab_u_+IND4(bi,hi,dt,i, H,dT,C), dab_u.to(tl.float32))

            dv = tl_dot(prec, qk.trans(), sdy) + kwi_dw_dstate + tl_dot(prec, ak.trans(), dab_u)
            tl.store(dv_+IND4(bi,t,hi,i, T,H,C), dv.to(tl.bfloat16))

            dab += tl_dot(prec, dab_u, u.trans()) * mask1
            dak += tl_dot(prec, dab_u, sv.trans()) * mask1
            dqb += tl_dot(prec, sdy, u.trans()) * mask2
            dqk += tl_dot(prec, sdy, sv.trans()) * mask2
        tl.debug_barrier()

        for j0 in range(0,C,dC):
            j = j0+tl.arange(0,dC)[None,:]

            dy_state = tl.zeros((dT,dC), tl.float32)
            dab_u_state = tl.zeros((dT,dC), tl.float32)
            fw_u_dstate = tl.zeros((dT,dC), tl.float32)
            fw_v_dstate = tl.zeros((dT,dC), tl.float32)
            state_dstate = tl.zeros((1,dC), tl.float32)

            fw = tl.load(fw_+IND3(bi,hi,j, H,C))
            wa = tl.load(wa_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            wq = tl.load(wq_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            for i0 in range(0,C,dC):
                i = i0+tl.arange(0,dC)[None,:]

                u = tl.load(u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                dab_u = tl.load(dab_u_+IND4(bi,hi,dt,i, H,dT,C)).to(tl.float32)
                sv = tl.load(v_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)
                sdy = tl.load(dy_+IND4(bi,t,hi,i, T,H,C)).to(tl.float32)

                state = tl.load(s_+IND5(bi,hi,t0,i.trans(),j, H,T//dT,C,C)).to(tl.float32)
                tl.debug_barrier()
                dstate = tl.load(ds_+IND4(bi,hi,i.trans(),j, H,C,C)).to(tl.float32)
                tl.debug_barrier()

                dab_u_state += tl_dot(prec, dab_u, state)
                fw_u_dstate += fw * tl_dot(prec, u, dstate)
                fw_v_dstate += fw * tl_dot(prec, sv, dstate)
                dy_state += tl_dot(prec, sdy, state)

                state_dstate += tl.sum(state*dstate, axis=0,keep_dims=True)

                dstate = dstate * fw + tl_dot(prec, sdy.trans(), wq) + tl_dot(prec, dab_u.trans(), wa)
                if t0 > 0:
                    tl.store(ds_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.float32))
                else:
                    tl.store(ds0_+IND4(bi,hi,i.trans(),j, H,C,C), dstate.to(tl.bfloat16))

            sw = tl.load(w_+IND4(bi,t,hi,j, T,H,C)).to(tl.float32)
            w = (-sw.exp()).exp()
            incl_pref = tl.cumprod(w,axis=0)
            non_incl_pref = incl_pref / w
            inv_incl_pref = 1 / incl_pref

            bwi = tl.load(bwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)
            kwi = tl.load(kwi_+IND4(bi,hi,dt,j, H,dT,C)).to(tl.float32)

            da = non_incl_pref * (tl_dot(prec, dab, bwi) + tl_dot(prec, dak, kwi) + dab_u_state)
            tl.store(da_+IND4(bi,t,hi,j, T,H,C), da.to(tl.bfloat16))

            dq = incl_pref * (tl_dot(prec, dqb, bwi) + tl_dot(prec, dqk, kwi) + dy_state)
            tl.store(dq_+IND4(bi,t,hi,j, T,H,C), dq.to(tl.bfloat16))

            db = inv_incl_pref * (tl_dot(prec, dab.trans(), wa) + tl_dot(prec, dqb.trans(), wq) + fw_u_dstate)
            tl.store(db_+IND4(bi,t,hi,j, T,H,C), db.to(tl.bfloat16))

            dk = inv_incl_pref * (tl_dot(prec, dak.trans(), wa) + tl_dot(prec, dqk.trans(), wq) + fw_v_dstate)
            tl.store(dk_+IND4(bi,t,hi,j, T,H,C), dk.to(tl.bfloat16))

            dw0 = fw * state_dstate
            for k in range(t0*dT,t0*dT+dT):
                lmask = (t<k).trans()
                A = (tl_dot(prec, dab*lmask, bwi) + tl_dot(prec, dak*lmask, kwi)) * wa * (t>k)
                A += (tl_dot(prec, dqb*lmask, bwi) + tl_dot(prec, dqk*lmask, kwi)) * wq * (t>=k)
                A += (fw_v_dstate*kwi + fw_u_dstate*bwi) * (t<k)
                A += dab_u_state*wa * (t>k) + dy_state*wq * (t>=k)
                dw = tl.sum(A, axis=0,keep_dims=True) + dw0

                wk = tl.load(w_+IND4(bi,k,hi,j, T,H,C)).to(tl.float32)
                dw *= -wk.exp()
                tl.store(dw_+IND4(bi,k,hi,j, T,H,C), dw.to(tl.bfloat16))



class TritonRWKV7(th.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,a,b,s0, dot_prec):
        K = 16
        B,T,H,C = w.shape
        assert T%K == 0
        assert C%16 == 0
        s0 = th.zeros(B,H,C,C, dtype=w.dtype,device=w.device) if s0 is None else s0
        y = th.empty_like(v)
        sT = th.empty_like(s0)
        s = th.zeros(B,H,T//K,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(4)]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        fw_attn_triton[(H,B)](w,q,k,v,a,b, s0,y,s,sT, wq,wa,kwi,bwi,fw, B,T,H,C,K, dot_prec)
        ctx.dot_prec = dot_prec
        ctx.save_for_backward(w,q,k,v,a,b,s)
        return y, sT
    @staticmethod
    def backward(ctx, dy, dsT):
        K = 16
        w,q,k,v,a,b,s = ctx.saved_tensors
        B,T,H,C = w.shape
        dw,dq,dk,dv,da,db,ds0 = [th.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
        fw = th.empty(B,H,C, dtype=th.float32,device=w.device)
        ds = th.empty(B,H,C,C, dtype=th.float32,device=w.device)
        wq,wa,kwi,bwi,u,dab_u = [th.empty(B,H,K,C, dtype=th.float32,device=w.device) for i in range(6)]
        bw_attn_triton[(H,B)](w,q,k,v,a,b, dy,s,dsT,ds, dw,dq,dk,dv,da,db,ds0, wq,wa,kwi,bwi,fw,u,dab_u, B,T,H,C,K, ctx.dot_prec)
        return dw,dq,dk,dv,da,db,ds0,None

@triton.jit
def tl_dot(prec:tl.constexpr, a, b):
    if prec == 'fp32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=False)
    elif prec == 'tf32':
        return tl.dot(a.to(tl.float32),b.trans().to(tl.float32).trans(), allow_tf32=True)
    elif prec == 'bf16':
        return tl.dot(a.to(tl.bfloat16),b.trans().to(tl.bfloat16).trans(), allow_tf32=True)
    else:
        tl.static_assert(False)

def attn_triton_bighead(r,w,k,v,a,b, s0, HEAD_DIM, dot_prec = 'fp32'):
    B,T,HC = w.shape
    C = HEAD_DIM
    H = HC//C
    r,w,k,v,a,b = [i.view(B,T,H,C) for i in [r,w,k,v,a,b]]
    x_out, s_out = TritonRWKV7.apply(w,r,k,v,a,b,s0,dot_prec)
    return x_out.view(B,T,HC), s_out

from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2MLP, Qwen2RMSNorm

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "recursal/QRWKV7-7B-Instruct-Preview-v0.1"
_CONFIG_FOR_DOC = "RWKV7Qwen2Config"

class RWKV7State(Cache):
    def __init__(self) -> None:
        super().__init__()
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.layer_kv_states: List[torch.Tensor] = []
        self.layer_shift_states:  List[torch.Tensor] = []

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.layer_kv_states)

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = 0) -> int:
        """Given the sequence length of the new inputs, returns the usable length of the cache."""
        # Linear Attention variants do not have a maximum length
        return new_seq_length

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError('Cannot reorder Linear Attention state')

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens

    def get_max_cache_shape(self) -> Optional[int]:
        """Returns the maximum sequence length of the cache object. DynamicCache does not have a maximum length."""
        return None

    def get_max_length(self) -> Optional[int]:
        """
        Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length.
        """
        return None

    # def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
    #     """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format. Used for
    #     backward compatibility."""
    #     legacy_cache = ()
    #     for layer_idx in range(len(self)):
    #         legacy_cache += ((self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx]),)
    #     return legacy_cache

    # @classmethod
    # #@deprecate_kwarg("num_hidden_layers", version="4.47.0")
    # def from_legacy_cache(
    #     cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None, num_hidden_layers: int | None = None
    # ) -> "RWKV7State":
    #     """Converts a cache in the legacy cache format into an equivalent `DynamicCache`. Used for
    #     backward compatibility."""
    #     cache = cls()
    #     if past_key_values is not None:
    #         for layer_idx in range(len(past_key_values)):
    #             layer_kv_state, layer_shift_state = past_key_values[layer_idx]
    #             cache.update(layer_kv_state, layer_shift_state, layer_idx)
    #     return cache

    def crop(self, max_length: int):
        # can't implement this for linear attention variants
        return

    @torch.no_grad
    def update(
        self,
        kv_state: torch.Tensor,
        shift_state: torch.Tensor,
        token_count: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:        
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += token_count

        # Update the cache
        # There may be skipped layers, fill them with empty lists
        for _ in range(len(self.layer_kv_states), layer_idx + 1):
            self.layer_kv_states.append(torch.zeros_like(kv_state).requires_grad_(False))
            self.layer_shift_states.append(torch.zeros_like(shift_state).requires_grad_(False))
        self.layer_kv_states[layer_idx].copy_(kv_state)
        self.layer_shift_states[layer_idx].copy_(shift_state)

        return self.layer_kv_states[layer_idx], self.layer_shift_states[layer_idx]

    # @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    # def batch_split(
    #     self, full_batch_size: int, split_size: int, num_hidden_layers: int = None
    # ) -> List["DynamicCache"]:
    #     """Split the current instance into a list of `DynamicCache` by the batch size. This will be used by
    #     `_split_model_inputs()` in `generation.utils`"""
    #     out = []
    #     for i in range(0, full_batch_size, split_size):
    #         current_split = DynamicCache()
    #         current_split._seen_tokens = self._seen_tokens
    #         current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
    #         current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
    #         out.append(current_split)
    #     return out

    # @classmethod
    # @deprecate_kwarg("num_hidden_layers", version="4.47.0")
    # def from_batch_splits(cls, splits: List["DynamicCache"], num_hidden_layers: int = None) -> "DynamicCache":
    #     """This is the opposite of the above `batch_split()` method. This will be used by `stack_model_outputs` in
    #     `generation.utils`"""
    #     cache = cls()
    #     for idx in range(len(splits[0])):
    #         key_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
    #         value_cache = [current.key_cache[idx] for current in splits if current.key_cache[idx] != []]
    #         if key_cache != []:
    #             layer_keys = torch.cat(key_cache, dim=0)
    #             layer_values = torch.cat(value_cache, dim=0)
    #             cache.update(layer_keys, layer_values, idx)
    #     return cache

    # def batch_repeat_interleave(self, repeats: int):
    #     """Repeat the cache `repeats` times in the batch dimension. Used in contrastive search."""
    #     for layer_idx in range(len(self)):
    #         self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
    #         self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)

    # def batch_select_indices(self, indices: torch.Tensor):
    #     """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
    #     for layer_idx in range(len(self)):
    #         self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
    #         self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]

try:
    #from fla.ops.gla.chunk import chunk_gla
    from fla.ops.gla.fused_recurrent import fused_recurrent_gla
except ImportError:
    print("Required module is not installed. Please install it using the following commands:")
    print("pip install -U git+https://github.com/fla-org/flash-linear-attention")
    print("Additionally, ensure you have at least version 2.2.0 of Triton installed:")
    print("pip install triton>=2.2.0")

class RWKV7Attention(nn.Module):
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        C = self.hidden_size = config.hidden_size
        H = self.num_heads = config.num_attention_heads
        N = self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
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

        calc_lora_rank = lambda exponent, multiplier: max(1, round(self.hidden_size ** exponent * multiplier / 32)) * 32
        lora_rank_decay = config.lora_rank_decay or calc_lora_rank(0.5, 1.8)
        lora_rank_iclr = config.lora_rank_iclr or calc_lora_rank(0.5, 1.8)
        lora_rank_value_residual_mix = config.lora_rank_value_residual_mix or calc_lora_rank(0.5, 1.3)
        lora_rank_gate = config.lora_rank_gate or calc_lora_rank(0.8, 0.6)

        self.x_r = nn.Parameter(torch.empty(1,1,C))
        self.x_w = nn.Parameter(torch.empty(1,1,C))
        self.x_k = nn.Parameter(torch.empty(1,1,C))
        self.x_v = nn.Parameter(torch.empty(1,1,C))
        self.x_a = nn.Parameter(torch.empty(1,1,C))
        self.x_g = nn.Parameter(torch.empty(1,1,C))

        self.w0 = nn.Parameter(torch.empty(1,1,C))
        self.w1 = nn.Parameter(torch.empty(C, lora_rank_decay))
        self.w2 = nn.Parameter(torch.empty(lora_rank_decay, C))

        self.a0 = nn.Parameter(torch.empty(1,1,C))
        self.a1 = nn.Parameter(torch.empty(C, lora_rank_iclr))
        self.a2 = nn.Parameter(torch.empty(lora_rank_iclr, C))

        if layer_idx > 0:
            self.v0 = nn.Parameter(torch.empty(1,1,C))
            self.v1 = nn.Parameter(torch.empty(C, lora_rank_value_residual_mix))
            self.v2 = nn.Parameter(torch.empty(lora_rank_value_residual_mix, C))

        self.g1 = nn.Parameter(torch.empty(C, lora_rank_gate))
        self.g2 = nn.Parameter(torch.empty(lora_rank_gate, C))

        self.k_k = nn.Parameter(torch.empty(1,1,C))
        self.k_a = nn.Parameter(torch.empty(1,1,C))
        self.r_k = nn.Parameter(torch.empty(H,N))

        # self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # self.receptance = nn.Linear(C, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.key = nn.Linear(C, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.value = nn.Linear(C, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.output = nn.Linear(self.num_heads * self.head_dim, C, bias=getattr(config, 'attention_output_bias', config.attention_bias))
        self.ln_x = nn.GroupNorm(H, C, eps=self.head_dim * 1e-5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[RWKV7State] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        output_shift_state = hidden_states[:, -1:].detach().clone()

        x = hidden_states

        B, T, C = hidden_states.shape
        H = self.num_heads
        N = self.head_dim
        q_len = T

        if use_cache and past_key_value is not None and len(past_key_value) > self.layer_idx:
            input_vk_state, input_shift_state = past_key_value[self.layer_idx]
        else:
            input_vk_state, input_shift_state = torch.zeros(B,H,N,N, dtype=torch.float32,device=x.device), torch.zeros_like(x[:, -1:])

        shifted = torch.cat([input_shift_state, x[:, :-1]], dim=1)
        xx = shifted - x

        xr = x+xx*self.x_r
        xw = x+xx*self.x_w
        xk = x+xx*self.x_k
        xv = x+xx*self.x_v
        xa = x+xx*self.x_a
        xg = x+xx*self.x_g

        r = self.q_proj(xr)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        v = v.view(B, T, -1, 1, self.head_dim).expand(-1, -1, -1, self.num_key_value_groups, -1).reshape(B, T, -1)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        kk = torch.nn.functional.normalize((k * self.k_k).view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        k = k * (1 + (a-1) * self.k_a)
        if self.layer_idx == 0: v_first = v
        else: v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)        

        if T == 1 or not self.training:
            w = torch.exp(-0.606531 * torch.sigmoid((self.w0 + w).float())) # 0.606531 = exp(-0.5)
            output_vk_state = input_vk_state
            for t in range(T):
                r_, w_, k_, v_, kk_, a_ = r[:,t], w[:,t], k[:,t], v[:,t], kk[:,t], a[:,t]
                vk = v_.view(B,H,N,1) @ k_.view(B,H,1,N)
                ab = (-kk_).view(B,H,N,1) @ (kk_*a_).view(B,H,1,N)
                output_vk_state = output_vk_state * w_.view(B,H,1,N) + output_vk_state @ ab.float() + vk.float()
                xx[:,t] = (output_vk_state.to(dtype=x.dtype) @ r_.view(B,H,N,1)).view(B,H*N)
            # FIXME - support fast triton kernel for non-training pre-fill with state in and out
        else:
            w = -torch.nn.functional.softplus(-(self.w0 + w)) - 0.5
            xx, output_vk_state = attn_triton_bighead(r, w, k, v, -kk, kk*a, input_vk_state, self.head_dim)

        xx = torch.nn.functional.group_norm(xx.view(B*T,H*N), num_groups=H, weight=self.ln_x.weight, bias=self.ln_x.bias, eps = self.ln_x.eps).view(B,T,H*N)
        xx = xx + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        xx = self.o_proj(xx * g)

        output_final_state = not self.training and use_cache and past_key_value is not None
        if output_final_state:
            past_key_value.update(output_vk_state, output_shift_state, q_len, self.layer_idx)

        return xx, v_first, past_key_value
    
class RWKV7Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: RWKV7Qwen2Config, layer_idx: int):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size

        self.self_attn = RWKV7Attention(config, layer_idx) #QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None, # unnecessary, but kept here for BC
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # unnecessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, v_first, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            v_first=v_first,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, v_first,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    

RWKV7QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`RWKV7Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    RWKV7QWEN2_START_DOCSTRING,
)
class RWKV7Qwen2PreTrainedModel(PreTrainedModel):
    config_class = RWKV7Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["RWKV7Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


RWKV7QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance, see our
            [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache);
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""

@add_start_docstrings(
    "The bare RWKV7Qwen2 Model outputting raw hidden-states without any specific head on top.",
    RWKV7QWEN2_START_DOCSTRING,
)
class RWKV7Qwen2Model(RWKV7Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: RWKV7Qwen2Config
    """

    def __init__(self, config: RWKV7Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RWKV7Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        #self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        #return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, RWKV7State):
            #return_legacy_cache = True
            past_key_values = RWKV7State()
            # if past_key_values is None:
            #     past_key_values = DynamicCache()
            # else:
            #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            #     logger.warning_once(
            #         "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
            #         "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
            #         "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            #     )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        causal_mask = None

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = None #self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        v_first = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    v_first,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    v_first=v_first,
                )

            hidden_states = layer_outputs[0]
            v_first = layer_outputs[1]

            i = 2
            if output_attentions:
                all_self_attns += (layer_outputs[i],)
                i += 1

            if use_cache:
                next_decoder_cache = layer_outputs[i]
                i += 1
            
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        #if return_legacy_cache:
        #    next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class RWKV7Qwen2ForCausalLM(RWKV7Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(RWKV7QWEN2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, RWKV7Qwen2ForCausalLM

        >>> model = RWKV7Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **loss_kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[1] :]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs

@add_start_docstrings(
    """
    The RWKV7Qwen2 Model transformer with a sequence classification head on top (linear layer).

    [`RWKV7Qwen2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    RWKV7QWEN2_START_DOCSTRING,
)
class RWKV7Qwen2ForSequenceClassification(RWKV7Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV7Qwen2Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    The RWKV7Qwen2 Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    RWKV7QWEN2_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with Llama->RWKV7Qwen2, LLAMA->RWKV7QWEN2
class RWKV7Qwen2ForTokenClassification(RWKV7Qwen2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = RWKV7Qwen2Model(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
The RWKV7Qwen2 Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    RWKV7QWEN2_START_DOCSTRING,
)
# Copied from transformers.models.mistral.modeling_mistral.MistralForQuestionAnswering with Mistral->RWKV7Qwen2, MISTRAL->RWKV7QWEN2
class RWKV7Qwen2ForQuestionAnswering(RWKV7Qwen2PreTrainedModel):
    base_model_prefix = "model"

    # Copied from models.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->RWKV7Qwen2
    def __init__(self, config):
        super().__init__(config)
        self.model = RWKV7Qwen2Model(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(RWKV7QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )