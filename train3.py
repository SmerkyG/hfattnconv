import os
import torch
from torch import nn, distributed as dist
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, FullyShardedDataParallel as FSDP
#from torch.distributed.fsdp._init_utils

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info, DistributedSampler

from typing import Any, Optional, Tuple, Callable

from dataclasses import dataclass, field, asdict, is_dataclass
from pydoc import locate

from logger import print0 as print

@dataclass(kw_only=True)
class Train_Config:
    wandb: str | None = None
    
    output_dir: str = ''
    train_dataset_path: str = ''
    train_dataset_split:str = 'train'
    train_dataset_column:str = 'text'
    attention_distillation_stage:int = 0
    teacher_model_path: str = ''
    sequence_length: int = 512
    token_count: int = 40_000_000

    max_epochs:int|None = None # not set by user
    max_steps:int = -1  # not set by user

    seed_everything:int = 1337

    lr_decay_type:str = 'cosine'
    lr_init:float = 6e-4
    lr_final:float = 1e-5
    warmup_steps:int = -1
    
    # optimizer:str = 'adamw'
    beta1:float = 0.9
    beta2:float = 0.95
    adam_eps:float = 1e-8

    weight_decay:float = 0.0

    ds_bucket_mb:int = 200

    log_every_n_steps:int = 50

    check_val_every_n_epoch: int = 1
    val_check_interval: int|None = None
    gradient_clip_val:float|None = 1.0

    accelerator:str = 'gpu'
    strategy:str = 'auto'
    devices:int = 1
    num_nodes:int = 1
    precision:str = 'bf16'
    accumulate_grad_batches:int = 1

    custom_fsdp:int = 0

    per_device_train_batch_size:int = 4
    gradient_checkpointing:bool=False
    dataloader_num_workers:int=2 # FIXME - this was 1 in the original code

@dataclass(kw_only=True)
class CLI_Config:
    tokenizer_path: str
    model_path: str
    attn_path: str = 'rwkv6attn.RWKV6Attention'
    attn_classes_path: str = 'transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES' # 'transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES' 
    seed: int = 1337
    train: Train_Config = field(default_factory=lambda: Train_Config())
    eval: bool = False

# cfg = CLI_Config(
#     tokenizer_path="Qwen/Qwen2.5-32B-Instruct",
#     model_path="Qwen/Qwen2.5-32B-Instruct",
#     train=Train_Config(attention_distillation_stage=2, train_dataset_path='robbiegwaldd/dclm-10B'),
# )

# cfg = CLI_Config(
#     tokenizer_path="Qwen/Qwen2.5-32B-Instruct",
#     model_path="/share/qrwkv/out/L64-D5120-qwen2-16384-3/rwkv-final.pth",
#     train=Train_Config(attention_distillation_stage=3, teacher_model_path='Qwen/Qwen2.5-32B-Instruct', train_dataset_path='robbiegwaldd/dclm-10B'),
# )

cfg = CLI_Config(
    tokenizer_path="Qwen/Qwen2.5-72B-Instruct",
    model_path="/share/qrwkv/out/L80-D8192-qwen2-2/rwkv-final.pth",
    train=Train_Config(attention_distillation_stage=3, teacher_model_path='Qwen/Qwen2.5-72B-Instruct', train_dataset_path='robbiegwaldd/dclm-10B'),
)

def tokenize_all(data, tokenizer, block_size : int, crop_n_blocks:int = 999999):
    # temporarily set tokenizer.model_max_length to avoid warnings when tokenizing long strings
    temp_max_length = getattr(tokenizer, 'model_max_length', None)
    tokenizer.model_max_length=int(1e30)

    tokenized = tokenizer(data['text'])['input_ids']

    dt = torch.long
    text_tensors = [torch.tensor(tokens, dtype=dt) for tokens in tokenized]
    eos_tensor = torch.tensor([tokenizer.eos_token_id], dtype=dt)

    output_batch = []
    current = torch.tensor([], dtype=dt)
    for text_tensor in text_tensors:
        text_tensor = text_tensor[0 : crop_n_blocks * block_size]
        while text_tensor.size(0) > 0:
            if current.size(0) > 0:
                current = torch.cat([current, eos_tensor])
            amount_taken = min(text_tensor.size(0), block_size-current.size(0))
            current = torch.cat([current, text_tensor[:amount_taken]], dim=0)
            text_tensor = text_tensor[amount_taken:]
            if current.size(0) == block_size:
                output_batch.append(current)
                current = torch.tensor([], dtype=dt)
                # NOTE - we skip parts that would give only a partial final output

    # reset tokenizer.model_max_length
    tokenizer.model_max_length = temp_max_length

    # NOTE - HF requires the labels be the same as the input_ids, which is essentially an off by one error on their part
    return dict(input_ids=output_batch, labels=output_batch)

class DistributedIterableDatasetWrapper(IterableDataset):
    def __init__(self, wrapped_dataset):
        super().__init__()
        self.wrapped_dataset = wrapped_dataset

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0      
        num_replicas = num_workers * dist.get_world_size()
        rank = dist.get_rank() * num_workers + worker_id

        i = -1
        for sample in iter(self.wrapped_dataset):
            i += 1
            if i % num_replicas == rank:
                yield sample

def _get_loss_logits_preds(batch, model, teacher, attention_distillation_stage):
    #inputs, labels = batch
    inputs = batch

    # remove the labels so that the model doesn't bother computing loss
    labels = inputs.pop("labels").to(device)

    if attention_distillation_stage == 2:
        #inputs['return_dict'] = True
        inputs['output_attentions'] = True
        outputs = model(**inputs)
        distillation_loss = torch.stack(outputs.attentions, dim=0).mean()
        logits = outputs['logits']
        return distillation_loss, None, logits, None #(distillation_loss, outputs) if return_outputs else distillation_loss
    
    elif attention_distillation_stage in (3, 4):
        if attention_distillation_stage == 4:
            teacher_logits = teacher(**inputs).logits
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # memory saving measure, because otherwise cross_entropy tried to allocate everything all at once
        ce_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        for b in range(student_logits.size(0)):
            # NOTE - HF requires the labels be the same as the input_ids, which is essentially an off by one error on their part
            ce_loss = ce_loss + F.cross_entropy(student_logits[b][..., :-1, :].view(-1, student_logits.size(-1)), labels[b][..., 1:].flatten())
        ce_loss = ce_loss / student_logits.size(0)

        if attention_distillation_stage == 4:
            # memory saving measure, because otherwise kl_div tried to allocate everything all at once
            distillation_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
            for b in range(student_logits.size(0)):
                # FIXME - currently only supporting 100% KL loss, no CE loss

                student_log_softmax = F.log_softmax(student_logits[b].view(-1, student_logits.size(-1)), dim=-1)
                teacher_log_softmax = F.log_softmax(teacher_logits[b].view(-1, teacher_logits.size(-1)), dim=-1)
                distillation_loss = distillation_loss + F.kl_div(
                    student_log_softmax,
                    teacher_log_softmax,
                    log_target=True,
                    reduction='batchmean'
                )
            distillation_loss = distillation_loss / student_logits.size(0)
            training_loss = distillation_loss
        else:
            training_loss = ce_loss

        return training_loss, ce_loss, student_logits, None #return (distillation_loss, outputs) if return_outputs else distillation_loss            

    assert attention_distillation_stage == 3
    return None, None, None, None # unreachable code

teacher_ckpt_path = cfg.train.teacher_model_path # "Qwen/Qwen2.5-32B-Instruct"
ckpt_path = cfg.model_path # "/share/qrwkv/out/L64-D5120-qwen2-16384-3/rwkv-final.pth" # "/share/qrwkv/out/L80-D8192-qwen2-2/rwkv-final.pth"
#hf_config_path = "Qwen/Qwen2.5-32B-Instruct"

distillation_stage = cfg.train.attention_distillation_stage

dtype = torch.bfloat16

# in stage 2, use from_pretrained on rank 0 and from_config elsewhere
# in stage 3, use load_state_dict on rank 0 and from_config elsewhere
# in stage 4, use load_state_dict on rank 0 and from_config elsewhere

teacher_model = None

# replace Qwen2RMSNorm with a version that supports reset_parameters for FSDP
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
class Qwen2RMSNorm2(modeling_qwen2.Qwen2RMSNorm):
    def reset_parameters(self):
        with torch.no_grad():
            self.weight.copy_(torch.ones_like(self.weight))
modeling_qwen2.Qwen2RMSNorm = Qwen2RMSNorm2

class Qwen2RotaryEmbedding2(modeling_qwen2.Qwen2RotaryEmbedding):
    def reset_parameters(self):
        pass
        #with torch.no_grad():
        #    self.weight.copy_(torch.ones_like(self.weight))
modeling_qwen2.Qwen2RotaryEmbedding = Qwen2RotaryEmbedding2

device_id = int(os.environ['LOCAL_RANK']) # FIXME - can we just use dist.get_rank()? or is that not the local rank or...
torch.cuda.set_device(device_id)
device = torch.cuda.current_device()

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

def create_and_patch_model(cfg:CLI_Config):
    #print("loading config")
    if cfg.train.attention_distillation_stage == 2:
        hf_config = AutoConfig.from_pretrained(cfg.model_path)
    elif cfg.train.attention_distillation_stage == 3:
        hf_config = AutoConfig.from_pretrained(cfg.train.teacher_model_path)

    # FIXME - hardcoded for now, but it'd be great if we could specify this in data somewhere per model type (or even analyze the weights to see)
    # NOTE - when loading a custom Qwen2RWKV model we don't need to set hf_config.attention_bias and model_config.attention_output_bias, because the model config contains it
    #if 'Qwen/Qwen' in hf_config_path:
    hf_config.attention_bias = True
    hf_config.attention_output_bias = False       

    ReplacementSelfAttentionType = locate(cfg.attn_path)
    assert isinstance(ReplacementSelfAttentionType, Callable)

    # replace attention classes
    attn_classes_dict = locate(cfg.attn_classes_path)
    attn_classes_dict_original_copy:dict = attn_classes_dict.copy()
    assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
    if cfg.train.attention_distillation_stage >= 3:
        for key in list(attn_classes_dict.keys()):
            attn_classes_dict[key] = ReplacementSelfAttentionType
 
    if cfg.train.attention_distillation_stage == 2 and dist.get_rank() == 0:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=hf_config)
    else:
        model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=dtype)

    # reset attention classes for upcoming teacher module's use
    for key, value in attn_classes_dict_original_copy.items():
        attn_classes_dict[key] = value

    # patch model
    if cfg.train.attention_distillation_stage == 2:
        # requires_grad_(False) on entire model, so it acts as teacher
        model.requires_grad_(False)

        # monkeypatch conditionally executed student attention replacements (which do require grad)
        for layer in model.model.layers:
            layer.self_attn = AttentionDistillationWrapper(layer.self_attn, ReplacementSelfAttentionType, hf_config, cfg.train.attention_distillation_stage)

        # student attention replacements do require grad in both stages 1 and 2
        for layer in model.model.layers:
            student_attn = layer.self_attn.student_attn
            student_attn.requires_grad_(True)

    return model

# FIXME - maybe implement init_empty_weights ourselves (to place tensors on meta device) instead of relying on hf transformers to do something unclear
#meta_device = torch.device("meta")
init_on_meta_device = init_empty_weights
#fabric.init_module(empty_init=True)

from contextlib import nullcontext

dist.init_process_group(world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))
try:
    # only load on rank zero to save memory
    if distillation_stage == 2:       
        print("loading student model")
        with (nullcontext if dist.get_rank() == 0 else init_on_meta_device)(): # init on meta device everywhere but rank 0
            # FIXME - maybe load from_config instead of from_pretrained for meta device non-rank0?
            student_model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=dtype)

        # FIXME - patch student_model as desired
        student_model.requires_grad = False
        # go thru layers and move attention to teacher, add RWKV as student

    elif distillation_stage >= 3:
        # retrofit model

        print("instantiating student model")
        with init_on_meta_device(): # always init on meta device since we'll populate it below on rank 0
            student_model:nn.Module = create_and_patch_model(cfg)

        if dist.get_rank() == 0:
            print("loading student state dict")
            state_dict = torch.load(ckpt_path, weights_only=True)
            print("populating student model")
            student_model.load_state_dict(state_dict, assign=True)
            # NOTE - model should not require patching here if the loaded checkpoint was saved in the final format

        if distillation_stage == 3:
            print("loading teacher model")
            if dist.get_rank() == 0:
                teacher_model = AutoModelForCausalLM.from_pretrained(teacher_ckpt_path, torch_dtype=dtype)
            else:
                with init_on_meta_device():
                    #print("loading teacher config")
                    teacher_hf_config = AutoConfig.from_pretrained(teacher_ckpt_path)
                    teacher_model = AutoModelForCausalLM.from_config(teacher_hf_config, torch_dtype=dtype)
            teacher_model.requires_grad = False
            teacher_model.eval()

    # placing into FSDP will shard the model appropriately across all GPUs
    print("placing student model onto FSDP")
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, ModuleWrapPolicy
    from functools import partial
    from transformers import models

    fsdp_auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=int(1e6))
    activation_checkpointing_policy = ModuleWrapPolicy({ models.qwen2.modeling_qwen2.Qwen2DecoderLayer })

    # initializes parameters that are on meta devices
    def init_fn(x: nn.Module):
        if dist.get_rank() == 0:
            return x
        else:
            return x.to_empty(device=device, recurse=False)

    # TODO: enable other policies
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    dp_strategy = ShardingStrategy.FULL_SHARD

    wrapped_student_model = FSDP(student_model, device_id=device, param_init_fn=init_fn, auto_wrap_policy=fsdp_auto_wrap_policy, sync_module_states=True, limit_all_gathers=True, mixed_precision=mp_policy, sharding_strategy=dp_strategy)

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing, checkpoint_wrapper
    apply_activation_checkpointing(wrapped_student_model, checkpoint_wrapper_fn=checkpoint_wrapper, auto_wrap_policy=activation_checkpointing_policy)

    wrapped_teacher_model = None
    if teacher_model is not None:
        print("placing teacher model onto FSDP")
        #wrapped_teacher_model = FSDP(teacher_model, device_id=device, sync_module_states=True) # auto_wrap_policy=..., 
        wrapped_teacher_model = FSDP(teacher_model, device_id=device, param_init_fn=init_fn, auto_wrap_policy=fsdp_auto_wrap_policy, sync_module_states=True, limit_all_gathers=True, mixed_precision=mp_policy, sharding_strategy=dp_strategy)

    print("creating optimizer")
    optim = torch.optim.Adam(wrapped_student_model.parameters(), lr=1e-5)

    print('loading dataset')
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    streaming_from_hf_remote = False
    train_dataset = load_dataset(cfg.train.train_dataset_path, split=cfg.train.train_dataset_split, streaming=streaming_from_hf_remote)
    if not streaming_from_hf_remote:
        train_dataset = train_dataset.to_iterable_dataset()
    train_dataset = train_dataset.map(lambda x: tokenize_all(x, tokenizer, cfg.train.sequence_length, 8), batched=True, remove_columns=train_dataset.column_names)

    train_dataset = train_dataset.take(cfg.train.token_count)
    train_dataset = train_dataset.shuffle(seed=cfg.seed)

    # wrap iterable dataset so training works properly for multi-gpu (instead of just same data copied on each gpu)
    train_dataset = DistributedIterableDatasetWrapper(train_dataset)

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    train_dataloader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=cfg.train.per_device_train_batch_size, num_workers=cfg.train.dataloader_num_workers, persistent_workers=False, drop_last=True)

    print("getting first entry from dataset...")
    step = 0
    for batch in train_dataloader:
        if step == 0:
            print("training...")

        # don't pass labels, because then the model will inefficiently calculate loss and run out of VRAM
        student_outputs = wrapped_student_model.forward(input_ids=batch['input_ids'])
        student_logits = student_outputs.logits

        # # memory saving measure, because otherwise cross_entropy tried to allocate everything all at once
        # ce_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        # for b in range(student_logits.size(0)):
        #     # NOTE - HF requires the labels be the same as the input_ids, which is essentially an off by one error on their part
        #     ce_loss = ce_loss + F.cross_entropy(student_logits[b][..., :-1, :].view(-1, student_logits.size(-1)), labels[b][..., 1:].flatten())
        # ce_loss = ce_loss / student_logits.size(0)
        # loss = ce_loss

        training_loss, ce_loss, student_logits, _ = _get_loss_logits_preds(batch, wrapped_student_model, wrapped_teacher_model, cfg.train.attention_distillation_stage) #student_results.loss
        training_loss.backward()
        # FIXME - add gradient clipping and gradient accumulation support
        optim.step()
        print("step ", step, "ce_loss", float(ce_loss.item()))
        step = step + 1

    print("completed")
except:
    print("error encountered")
    raise
finally:
    print('tearing down process group...')
    dist.destroy_process_group()

print("DONE!!!")
