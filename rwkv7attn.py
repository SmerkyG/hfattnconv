import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Callable

from transformers.cache_utils import Cache, DynamicCache, StaticCache

from transformers import AutoConfig, AutoModelForCausalLM

wind_cuda = False
fast_cuda = True

# -----------------------------------------------------------------------------
# RWKV-7 kernel

HEAD_SIZE = 64 #cmd_args.headsz
sequence_length = 1024

from torch.utils.cpp_extension import load

if wind_cuda:
    load(name="wind", sources=['rwkv_cuda_wind/wind_rwkv7.cu', 'rwkv_cuda_wind/wind_rwkv7.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=[f'-D_C_={HEAD_SIZE}',"-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"])

    class WindRWKV7(torch.autograd.Function):
        @staticmethod
        def forward(ctx,w,q,k,v,a,b):
            B,T,H,C = w.shape
            s0 = torch.zeros(B,H,C,C,dtype=w.dtype,device=w.device)
            assert T%16 == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,a,b,s0])
            w,q,k,v,a,b,s0 = [i.contiguous() for i in [w,q,k,v,a,b,s0]]
            y = torch.empty_like(v)
            sT = torch.empty_like(s0)
            s = torch.zeros(B,H,T//16,C,C, dtype=w.dtype,device=w.device)
            torch.ops.wind.forward(w,q,k,v,a,b, s0,y,s,sT)
            ctx.save_for_backward(w,q,k,v,a,b,s)
            return y
        
        @staticmethod
        def backward(ctx,dy):
            w,q,k,v,a,b,s = ctx.saved_tensors
            B,T,H,C = w.shape
            dsT = torch.zeros(B,H,C,C,dtype=dy.dtype,device=dy.device)
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            dy,dsT = [i.contiguous() for i in [dy,dsT]]
            dw,dq,dk,dv,da,db,ds0 = [torch.empty_like(x) for x in [w,q,k,v,a,b,dsT]]
            torch.ops.wind.backward(w,q,k,v,a,b, dy,s,dsT, dw,dq,dk,dv,da,db,ds0)
            return dw,dq,dk,dv,da,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b) -> torch.Tensor:
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//HEAD_SIZE,HEAD_SIZE) for i in [q,w,k,v,a,b]]
        return WindRWKV7.apply(w,q,k,v,a,b).view(B,T,HC)

elif fast_cuda:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    VERSION = 1 if HEAD_SIZE < 128 else 2
    load(name="wind_backstepping", sources=[f'rwkv_cuda_wind/backstepping_f32_{VERSION}.cu', 'rwkv_cuda_wind/backstepping_f32.cpp'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            w,q,k,v,z,b = [i.contiguous() for i in [w,q,k,v,z,b]]
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert dy.dtype == torch.bfloat16
            dy = dy.contiguous()
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b) -> torch.Tensor:
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
else:
    DTYPE = torch.bfloat16
    XTYPE = torch.float
    T = sequence_length
    CHUNK_LEN = 16

    load(name="wkv7g", sources=["rwkv_cuda/wkv7g_op.cpp", f"rwkv_cuda/wkv7g_v1.cu"], is_python_module=False,
                        verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={T}", f"-D_CHUNK_LEN_={CHUNK_LEN}"])
    class WKV_7g(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            with torch.no_grad():
                B, T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                A = T // CHUNK_LEN
                assert HEAD_SIZE == C // H
                assert T % CHUNK_LEN == 0
                assert all(i.dtype == DTYPE for i in [r,w,k,v,a,b])
                r,w,k,v,a,b = [i.contiguous() for i in [r,w,k,v,a,b]]
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                y = torch.empty((B, T, C), device=k.device, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                saa = torch.empty((B, T, H, N), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                sss = torch.empty((B, H, A, N, N), device=k.device, dtype=torch.float, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                torch.ops.wkv7g.forward(B, T, C, H, r, w, k, v, a, b, y, saa, sss)
                ctx.save_for_backward(r, w, k, v, a, b, saa, sss)
                return y
        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                N = HEAD_SIZE
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                A = T // CHUNK_LEN
                assert gy.dtype == DTYPE
                gy = gy.contiguous()
                r, w, k, v, a, b, saa, sss = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.zero_()#.uniform_(-100, 100)
                ga = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gb = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=DTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                zzz = torch.empty((B, H, A-1, N, N), device=gy.device, dtype=XTYPE, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                torch.ops.wkv7g.backward(B, T, C, H, r, w, k, v, a, b, saa, sss, zzz, gy, gr, gw, gk, gv, ga, gb)
                del saa
                del sss
                del zzz
                return (gr, gw, gk, gv, ga, gb)
    def RUN_CUDA_RWKV7g(r, w, k, v, a, b) -> torch.Tensor:
        return WKV_7g.apply(r, w, k, v, a, b)      

class RWKV7Attention(nn.Module):
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

        C = self.hidden_size = config.hidden_size
        H = self.num_heads = config.num_attention_heads
        N = self.head_dim = getattr(config, 'head_dim', self.hidden_size // self.num_heads)
        attention_hidden_size = H * N
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        #self.attention_dropout = config.attention_dropout

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
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

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        # self.receptance = nn.Linear(C, C, bias=config.attention_bias)
        # self.key = nn.Linear(C, C, bias=config.attention_bias)
        # self.value = nn.Linear(C, C, bias=config.attention_bias)
        # self.output = nn.Linear(C, C, bias=config.attention_bias)

        self.receptance = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.key = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.value = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.output = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=getattr(config, 'attention_output_bias', config.attention_bias))
        self.ln_x = nn.GroupNorm(H, C, eps=self.head_dim * 1e-5)

        num_hidden_layers = n_layer = self.config.num_hidden_layers
        n_embd = self.hidden_size
        dim_att = self.num_heads * self.head_dim
        layer_id = self.layer_idx



        module = self

        ratio_0_to_1 = layer_id / (num_hidden_layers - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / num_hidden_layers)  # 1 to ~0

        time_weight = torch.tensor(
            [i / C for i in range(C)],
            dtype=module.x_k.dtype,
            device=module.x_k.device,
        )
        time_weight = time_weight[None, None, :]

        decay_speed = [
            -7.0 + 5.0 * (n / (attention_hidden_size - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
            for n in range(attention_hidden_size)
        ]
        decay_speed = torch.tensor(decay_speed, dtype=module.w0.dtype, device=module.w0.device)

        with torch.no_grad():
            torch.nn.init.zeros_(module.x_r) #.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )
            torch.nn.init.zeros_(module.x_w) #.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
            torch.nn.init.zeros_(module.x_k) #.copy_( 1.0 - (torch.pow(time_weight, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1) )
            torch.nn.init.zeros_(module.x_v) #.copy_( 1.0 - (torch.pow(time_weight, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1) )
            torch.nn.init.zeros_(module.x_a) #.copy_( 1.0 - torch.pow(time_weight, 0.9 * ratio_1_to_almost0) )
            torch.nn.init.zeros_(module.x_g) #.copy_( 1.0 - torch.pow(time_weight, 0.2 * ratio_1_to_almost0) )
            
        with torch.no_grad():
            ratio_0_to_1 = layer_id / (n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, n_embd)
            for i in range(n_embd):
                ddd[0, 0, i] = i / n_embd

            # initialization comes from fitting my RWKV-6 7B runs
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, 0.6 * ratio_1_to_almost0 ** 0.9))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x
                
            module.w0.copy_(decay_speed.reshape(1,1,C) + 0.5) # !!! 0.5 comes from F.softplus !!!
            module.w1.zero_()
            ortho_init(module.w2, 0.1)

            module.a0.zero_()
            module.a1.zero_()
            ortho_init(module.a2, 0.1)

            if layer_id > 0:
                module.v0.copy_(1.0)
                module.v1.zero_()
                ortho_init(module.v2, 0.1)

            module.g1.zero_()
            ortho_init(module.g2, 0.1)

            self.k_k.copy_(0.85) # FIXME - should this be 1.0?
            self.k_a.copy_(1.0)
            self.r_k.zero_()

            module.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(attention_hidden_size**0.5))
            module.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(attention_hidden_size**0.5))
            module.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(attention_hidden_size**0.5))
            module.output.weight.data.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
        v_first: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        x = hidden_states

        input_seq_len = x.size(1)
        if input_seq_len % 16 != 0:
            x = F.pad(x, (0, 0, 0, 16 - input_seq_len%16))
        B, T, C = x.size()
        H = self.num_heads
        N = self.head_dim

        dxprev = F.pad(x, (0, 0, 1, -1)) - x

        #xxx = x + dxprev * self.time_maa_x
        #xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 6, -1).transpose(0, 1)
        #xxx = torch.bmm(xxx, self.time_maa_w2).view(6, B, T, -1)
        #mr, mw, mk, mv, ma, mg = xxx.unbind(dim=0)

        # xr = x + dxprev * (self.time_maa_r + mr)
        # xw = x + dxprev * (self.time_maa_w + mw)
        # xk = x + dxprev * (self.time_maa_k + mk)
        # xv = x + dxprev * (self.time_maa_v + mv)
        # xa = x + dxprev * (self.time_maa_a + ma)
        # xg = x + dxprev * (self.time_maa_g + mg)

        xr = x+dxprev*self.x_r
        xw = x+dxprev*self.x_w
        xk = x+dxprev*self.x_k
        xv = x+dxprev*self.x_v
        xa = x+dxprev*self.x_a
        xg = x+dxprev*self.x_g

        r = self.receptance(xr)
        w = torch.tanh(xw @ self.w1) @ self.w2
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        log_neglog_w = - 0.5 - torch.nn.functional.softplus(-(self.w0 + w)) # FIXME - we had tried 0-softplus before

        # repeat k/v heads if n_kv_heads < n_heads
        k = k.view(B, T, 1, -1, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(B, T, -1)
        v = v.view(B, T, 1, -1, self.head_dim).expand(-1, -1, self.num_key_value_groups, -1, -1).reshape(B, T, -1)

        kk = torch.nn.functional.normalize((k * self.k_k).view(B,T,H,-1), dim=-1, p=2.0).view(B,T,-1)
        k = k * (1 + (a-1) * self.k_a)
        if self.layer_idx == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)

        # FIXME - do we still need something like the code below?
        #mk = torch.sigmoid(self.time_misc_k + (xk @ self.mk_w1) @ self.mk_w2)
        #k = k * torch.clamp(log_neglog_w*mk, max=0).exp()

        x = RUN_CUDA_RWKV7g(r.bfloat16(), log_neglog_w.bfloat16(), k.bfloat16(), v.bfloat16(), -kk.bfloat16(), (kk*a).bfloat16())

        x = self.ln_x(x.view(B * T, H*N)).view(B, T, H*N)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)

        if input_seq_len != T:
            x = x[:, :input_seq_len]

        return x, None, past_key_value

