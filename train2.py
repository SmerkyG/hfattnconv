import os, math, gc, importlib
import torch
import torch.linalg
import torch.utils.checkpoint
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import lightning.pytorch as pl
from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
from lightning.pytorch.strategies import DeepSpeedStrategy

import pickle
import torch.distributed as dist
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info, DistributedSampler
from torch.distributed import get_rank, get_world_size

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_set_full_fp32_param, safe_set_local_fp32_param, safe_set_full_optimizer_state

from torch.optim import lr_scheduler

from logger import print0 as print

import metrics

from dataclasses import dataclass, field

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

class LightningModelWrapper(pl.LightningModule):
    def __init__(self, model:nn.Module, config:CLI_Config, teacher:nn.Module|None=None):
        super().__init__()
        self.model = model
        self.config = config
        self.teacher = teacher
        if teacher is not None:
            teacher.requires_grad_(False)

        self.kl_loss_metric = metrics.Loss()
        self.ce_loss_metric = metrics.Loss()
        self.configured = False

        # # temporarily stash the teacher inside the model itself so it gets wrapped and placed onto the proper devices, even when using FSDP or deepspeed_stage_3
        # # this will be removed from the model when training starts
        # self.model.teacher = teacher

    def forward(self, *args, **kwargs):
        # if self.model.teacher is not None:
        #     self.model.teacher = None

        return self.model.forward(*args, **kwargs)
    
    def configure_optimizers(self):
        train_config = self.config.train

        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if p.requires_grad # (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": train_config.weight_decay, 
            },
        ]
        betas = (train_config.beta1, train_config.beta2)

        from torch.optim import AdamW
        optimizer_class = AdamW # optimizer_kwargs.update({"fused": True})
        #optimizer_class = FusedAdam

        opt = optimizer_class(optim_groups, lr=train_config.lr_init, betas=betas, eps=train_config.adam_eps, weight_decay=train_config.weight_decay) #, bias_correction=True, adam_w_mode=True, amsgrad=False)
        return opt

    def _get_loss_logits_preds(self, batch, batch_idx):
        #inputs, labels = batch
        inputs = batch

        # remove the labels so that the model doesn't bother computing loss
        labels = inputs.pop("labels")

        if self.config.train.attention_distillation_stage == 2:
            #inputs['return_dict'] = True
            inputs['output_attentions'] = True
            outputs = model(**inputs)
            distillation_loss = torch.stack(outputs.attentions, dim=0).mean()
            logits = outputs['logits']
            return distillation_loss, None, logits, None #(distillation_loss, outputs) if return_outputs else distillation_loss
        
        elif self.config.train.attention_distillation_stage == 3:
            teacher_logits = self.teacher(**inputs).logits
            student_outputs = model(**inputs)
            student_logits = student_outputs.logits

            # memory saving measure, because otherwise cross_entropy tried to allocate everything all at once
            ce_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
            for b in range(student_logits.size(0)):
                # NOTE - HF requires the labels be the same as the input_ids, which is essentially an off by one error on their part
                ce_loss = ce_loss + F.cross_entropy(student_logits[b][..., :-1, :].view(-1, student_logits.size(-1)), labels[b][..., 1:].flatten())
            ce_loss = ce_loss / student_logits.size(0)

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

            return distillation_loss, ce_loss, student_logits, None #return (distillation_loss, outputs) if return_outputs else distillation_loss            

        assert self.config.train.attention_distillation_stage == 3
        return None, None, None, None # unreachable code

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        training_loss, reported_loss, logits, preds = self._get_loss_logits_preds(batch, batch_idx)

        self.kl_loss_metric.update(metrics.MetricArgs(inputs, logits, preds, labels, training_loss))
        if reported_loss is not None:
            self.ce_loss_metric.update(metrics.MetricArgs(inputs, logits, preds, labels, reported_loss))
        if self.trainer.is_global_zero:
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
                    self.log("train/loss", self.kl_loss_metric.compute(), prog_bar=True, on_step=True)#, rank_zero_only=True)
                    self.kl_loss_metric.clear()
                    if reported_loss is not None:
                        self.log("train/ce_loss", self.ce_loss_metric.compute(), prog_bar=True, on_step=True)#, rank_zero_only=True)
                        self.kl_loss_metric.clear()

        # self.log("train/loss", training_loss, prog_bar=True, sync_dist=True)
        # if reported_loss is not None:
        #     self.log("train/ce_loss", reported_loss, prog_bar=True, sync_dist=True)

        # FIXME - move learning rate update to lr_scheduler so it occurs only on optimizer steps, not every training step
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            def lerp(a, b, amt): return a * (1-amt) + b * amt

            lr_init = self.config.train.lr_init
            warmup_steps = max(0, self.config.train.warmup_steps)
            if batch_idx < warmup_steps:
                lr = lerp(0.0, lr_init, batch_idx / warmup_steps)
            else:
                t = max(0, min(1, (batch_idx - warmup_steps) / (config.train.max_steps - warmup_steps)))
                lr_final = self.config.train.lr_final
                if self.config.train.lr_decay_type == 'linear':
                    lr = lerp(lr_init, lr_final, t)
                elif self.config.train.lr_decay_type == 'cosine':
                    lr = lerp(lr_init, lr_final, 0.5 - 0.5 * math.cos(t * math.pi))
                else:
                    print("bad lr_decay_type specified")
                    exit()

            self.lr = lr
            for param_group in self.trainer.optimizers[0].param_groups:
                param_group["lr"] = lr

        self.log("train/learning_rate", lr, prog_bar=False)

        return training_loss

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
        num_replicas = num_workers * get_world_size()
        rank = get_rank() * num_workers + worker_id

        i = -1
        for sample in iter(self.wrapped_dataset):
            i += 1
            if i % num_replicas == rank:
                yield sample

if __name__ == "__main__":
    from lightning import Trainer
    from lightning_utilities.core.rank_zero import rank_zero_info, rank_zero_only
    import lightning as pl

    from transformers import AutoModelForCausalLM, AutoTokenizer

    import os, warnings, math, datetime, sys, time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    from datasets import load_dataset

    from configs import parse_cmdline_configs

    from pydoc import locate
    from typing import Callable

    from rwkv6attn import load_and_patch_model_with_attention_replacement

    config, errors = parse_cmdline_configs(sys.argv[1:], base_config_type=CLI_Config)
    config:CLI_Config
    if errors != '':
        print(errors)
        exit()

    train_batch_size = config.train.sequence_length * config.train.per_device_train_batch_size * config.train.devices #torch.cuda.device_count()
    config.train.max_steps = max(1, config.train.token_count // train_batch_size)

    warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
    warnings.filterwarnings("ignore", ".*FutureWarning: `torch.cuda.amp.custom_fwd(args...)` is deprecated.*")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if str(config.train.precision) == "32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    pl.seed_everything(config.train.seed_everything)

    attention_distillation_stage = config.train.attention_distillation_stage
    assert attention_distillation_stage in (0,2,3)

    ReplacementSelfAttnType = locate(config.attn_path)
    assert isinstance(ReplacementSelfAttnType, Callable)

    model = load_and_patch_model_with_attention_replacement(config.model_path, config.attn_classes_path, ReplacementSelfAttnType, attention_distillation_stage)
    model.train()
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    teacher = AutoModelForCausalLM.from_pretrained(config.train.teacher_model_path) if attention_distillation_stage == 3 else None

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    train_dataset = load_dataset(config.train.train_dataset_path, split=config.train.train_dataset_split, streaming=True)
    train_dataset = train_dataset.map(lambda x: tokenize_all(x, tokenizer, config.train.sequence_length, 8), batched=True, remove_columns=train_dataset.column_names)

    train_dataset = train_dataset.take(config.train.token_count)
    train_dataset = train_dataset.shuffle(seed=config.seed)

    # wrap iterable dataset so training works properly for multi-gpu (instead of just same data copied on each gpu)
    train_dataset = DistributedIterableDatasetWrapper(train_dataset)

    wrapper = LightningModelWrapper(model, config, teacher) # delay setting the teacher until after init so deepspeed_stage_3 doesn't break it

    strategy_obj = config.train.strategy

    logger = False
    if config.train.wandb is not None and config.train.wandb != '':
        from lightning.pytorch.loggers import WandbLogger
        logger = WandbLogger(project=config.train.wandb)

    # FIXME - why use_distributed_sampler=False? was this an oversight in the original repo? is this related to replace_sampler_ddp from Bo's code?
    trainer = Trainer(
                        use_distributed_sampler=False, 
                        enable_checkpointing=False,
                        num_sanity_val_steps=0,
                        logger=logger,
                        max_epochs=config.train.max_epochs,
                        max_steps=config.train.max_steps,

                        accelerator=config.train.accelerator, 
                        strategy=strategy_obj, 
                        devices=config.train.devices, 
                        num_nodes=config.train.num_nodes, 
                        precision=config.train.precision,
                        #callbacks=[train_callback(config)], 
                        check_val_every_n_epoch=config.train.check_val_every_n_epoch, 
                        log_every_n_steps=config.train.log_every_n_steps, 
                        accumulate_grad_batches=config.train.accumulate_grad_batches, 
                        gradient_clip_val=config.train.gradient_clip_val, 
                        val_check_interval=config.train.val_check_interval)

    if "deepspeed" in config.train.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = config.train.ds_bucket_mb * 1000 * 1000

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    train_data_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, batch_size=config.train.per_device_train_batch_size, num_workers=config.train.dataloader_num_workers, persistent_workers=False, drop_last=True)

    ds_ckpt_path = None
    #if 'deepspeed_stage_3' in config.train.strategy and config.continued:
    #    ds_ckpt_path = config.train.load_model
    #print("max_steps", config.train.max_steps)
    trainer.fit(wrapper, train_dataloaders=train_data_loader, ckpt_path=ds_ckpt_path) # val_dataloaders=validation_data_loader, 

    if attention_distillation_stage == 2:
        # remove extra teacher attention modules so we can load this model normally in stage 3 and beyond
        for layer in model.model.layers:
            layer.self_attn = layer.self_attn.student_attn
    elif attention_distillation_stage == 3:
        # remove teacher, if any, so it doesn't get saved to the checkpoint
        wrapper.teacher = None

    print("Training complete - saving model")

    if 'deepspeed_stage_3' in config.train.strategy:
        # NOTE - have to get state_dict on all ranks (it only returns the actual dict on rank zero)
        state_dict = trainer.strategy.deepspeed_engine._zero3_consolidated_16bit_state_dict()
        for n in list(state_dict.keys()):
            if n.startswith('model.'):
                state_dict[n[len('model.'):]] = state_dict.pop(n)
    else:
        state_dict = None
    if trainer.global_rank == 0:
        model.save_pretrained(config.train.output_dir, state_dict=state_dict, safe_serialization=True)

    print("Done")
