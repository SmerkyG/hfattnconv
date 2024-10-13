import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, PreTrainedModel
from transformers.trainer_callback import TrainerCallback
from datasets import load_dataset
from configs import parse_cmdline_configs
from dataclasses import dataclass, field
from pydoc import locate

from rwkv6attn import RWKV6Attention, RWKV6AttentionDistillationWrapper

@dataclass(kw_only=True)
class Train_Config:
    output_dir: str = ''
    train_dataset_path: str = ''
    train_dataset_split:str = 'train'
    train_dataset_column:str = 'text'
    attention_distillation_stage:int = 0
    teacher_model_path: str = ''
    sequence_length: int = 512
    token_count: int = 40_000_000
    training_args: TrainingArguments = field(default_factory=lambda: TrainingArguments(output_dir='out', bf16=True, per_device_train_batch_size=4, gradient_checkpointing=False, include_tokens_per_second=True, learning_rate=1e-3, adam_beta1=0.9, adam_beta2=0.95, lr_scheduler_type='cosine', dataloader_num_workers=8))

@dataclass(kw_only=True)
class CLI_Config:
    tokenizer_path: str
    model_path: str
    attn_classes_path: str = 'transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES' # 'transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES' 
    seed: int = 1337
    train: Train_Config = field(default_factory=lambda: Train_Config())
    eval: bool = False

class RemoveTeacherCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, model, **kwargs):
        # remove teacher from model so we don't end up saving it later
        del model.teacher
 
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher:nn.Module, callbacks = None, **kwargs):
        if callbacks is None:
            callbacks = []
        callbacks += [RemoveTeacherCallback()]
        super().__init__(*args, callbacks=callbacks, **kwargs)
        self.teacher = teacher
        teacher.requires_grad_(False)
        # temporarily stash the teacher inside the model itself so it gets wrapped and placed onto the proper devices, even when using FSDP or deepspeed_stage_3
        # this will be removed from the model when training starts, via the RemoveTeacherCallback
        self.model.teacher = teacher

    def compute_loss(self, model, inputs, return_outputs=False):
        # remove the labels so that the model doesn't bother computing loss
        labels = inputs.pop("labels")

        teacher_logits = self.teacher(**inputs).logits
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # memory saving measure, because otherwise cross_entropy tried to allocate everything all at once
        ce_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
        for b in range(student_logits.size(0)):
            # NOTE - HF requires the labels be the same as the input_ids, which is essentially an off by one error on their part
            ce_loss = ce_loss + F.cross_entropy(student_logits[b][..., :-1, :].view(-1, student_logits.size(-1)), labels[b][..., 1:].flatten())
        ce_loss = ce_loss / student_logits.size(0)
        if (self.state.global_step + 0) % 10 == 0:
            self.log(dict(ce_loss = float(ce_loss.item())))
        del ce_loss

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

        outputs = student_outputs
        return (distillation_loss, outputs) if return_outputs else distillation_loss

class AttentionDistillationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # remove the labels so that the model doesn't bother computing loss
        inputs.pop("labels")
        #inputs['return_dict'] = True
        inputs['output_attentions'] = True
        outputs = model(**inputs)
        distillation_loss = torch.stack(outputs.attentions, dim=0).mean()
        return (distillation_loss, outputs) if return_outputs else distillation_loss

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
    return dict(input_ids=output_batch, labels=output_batch) # different size than input_batch

def main():
    config, errors = parse_cmdline_configs(sys.argv[1:], base_config_type=CLI_Config)
    if errors != '':
        print(errors)
        exit()

    # FIXME - would be nicer to get the device count from config.train.training_args.train_batch_size(), but not sure about setup if we instantiate it first
    train_batch_size = config.train.sequence_length * config.train.training_args.per_device_train_batch_size * torch.cuda.device_count()
    config.train.training_args.max_steps = config.train.token_count // train_batch_size

    # workaround for TrainingArguments use of __post_init__
    if not isinstance(config.train.training_args, TrainingArguments):
        config.train.training_args = TrainingArguments(**config.train.training_args)

    training_args = config.train.training_args

    model_config = AutoConfig.from_pretrained(config.model_path)

    # FIXME - hardcoded for now, but it'd be great if we could specify this in data somewhere per model type (or even analyze the weights to see)
    # NOTE - when loading a custom Qwen2RWKV model we don't need to set model_config.attention_bias and model_config.attention_output_bias, because the model config contains it
    if 'Qwen/Qwen' in config.model_path:
        model_config.attention_bias = True
        model_config.attention_output_bias = False

    attention_distillation_stage = config.train.attention_distillation_stage

    # replace attention classes
    attn_classes_dict = locate(config.attn_classes_path)
    attn_classes_dict_original_copy:dict = attn_classes_dict.copy()
    assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
    if attention_distillation_stage == 1:
        pass
    elif attention_distillation_stage == 2:
        # replace all attention classes with RWKV6AttentionDistillationWrapper that will be passed an invocation of the original attention class to wrap
        for key in list(attn_classes_dict.keys()):
            cls = attn_classes_dict[key]
            attn_classes_dict[key] = lambda *args, **kwargs: RWKV6AttentionDistillationWrapper(cls(*args, **kwargs), model_config, attention_distillation_stage)
    elif attention_distillation_stage >= 3:
        for key in list(attn_classes_dict.keys()):
            attn_classes_dict[key] = RWKV6Attention

    model = AutoModelForCausalLM.from_pretrained(config.model_path, config=model_config)

    # reset attention classes for upcoming teacher module's use
    for key, value in attn_classes_dict_original_copy.items():
        attn_classes_dict[key] = value

    # patch model
    model.attention_distillation_stage = attention_distillation_stage
    if attention_distillation_stage in (1, 2):
        # requires_grad_(False) on entire model, so it acts as teacher
        model.requires_grad_(False)

        if attention_distillation_stage == 1:
            # monkeypatch conditionally executed student attention replacements (which do require grad)
            for layer in model.model.layers:
                layer.self_attn = RWKV6AttentionDistillationWrapper(layer.self_attn, model_config, attention_distillation_stage)

        # student attention replacements do require grad in both stages 1 and 2
        for layer in model.model.layers:
            student_attn = layer.self_attn.student_attn
            if attention_distillation_stage == 1:
                student_attn.time_maa_x.requires_grad_(True)
                student_attn.time_maa_r.requires_grad_(True)
                student_attn.time_maa_k.requires_grad_(True)
                #student_attn.time_maa_w.requires_grad_(True) # FIXME!!! we left this out!
                student_attn.time_maa_w1.requires_grad_(True)
                student_attn.time_maa_w2.requires_grad_(True)
                student_attn.time_decay.requires_grad_(True)
                student_attn.time_decay_w1.requires_grad_(True)
                student_attn.time_decay_w2.requires_grad_(True)
                # FIXME - wow we removed q, k here by accident and it.. helped??!?!
            elif attention_distillation_stage == 2:
                student_attn.requires_grad_(True)

            # student_attn.requires_grad_(True)
            # if attention_distillation_stage == 1:
            #     # BUT, we don't want to train certain parameters during stage 1!
            #     student_attn.q_proj.requires_grad_(False) # not training q,k was an odd choice but it somehow helped
            #     student_attn.k_proj.requires_grad_(False)
            #     student_attn.v_proj.requires_grad_(False)
            #     student_attn.o_proj.requires_grad_(False)
            #     #student_attn.time_maa_w.requires_grad_(False) # FIXME!!! we left this out by accident!

            #     student_attn.time_maa_v.requires_grad_(False)
            #     student_attn.ln_x.requires_grad_(False)
                
        TrainerType = AttentionDistillationTrainer
        trainer_kwargs = {}

    elif attention_distillation_stage == 3:
        if model_config.tie_word_embeddings:
            # copy untied embeddings
            model.get_output_embeddings().weight = nn.Parameter(model.get_input_embeddings().weight.clone())
            # untie the embeddings in the config, too
            model_config.tie_word_embeddings = False

        # traditional distillation, so we need to load a second teacher model for inference
        TrainerType = DistillationTrainer
        trainer_kwargs = dict(teacher=AutoModelForCausalLM.from_pretrained(config.train.teacher_model_path))

    elif attention_distillation_stage == 4:
        if model_config.tie_word_embeddings:
            # copy untied embeddings
            model.get_output_embeddings().weight = nn.Parameter(model.get_input_embeddings().weight.clone())
            # untie the embeddings in the config, too
            model_config.tie_word_embeddings = False

        TrainerType = Trainer
        trainer_kwargs = {}

    else:
        TrainerType = Trainer
        trainer_kwargs = {}

    # FIXME - patch model's _init_weights function

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    train_dataset = load_dataset(config.train.train_dataset_path, split=config.train.train_dataset_split, streaming=True)

    train_dataset = train_dataset.map(lambda x: tokenize_all(x, tokenizer, config.train.sequence_length, 8), batched=True, remove_columns=train_dataset.column_names)

    train_dataset = train_dataset.take(config.train.token_count)

    train_dataset = train_dataset.shuffle(seed=config.seed)

    if not config.eval:
        trainer = TrainerType(
            model = model,
            args = training_args,
            train_dataset = train_dataset,
            #eval_dataset = dataset['test'],
            tokenizer = tokenizer,
            #data_collator = data_collator,
            **trainer_kwargs
        )
        
        trainer.train()

        delattr(model, 'attention_distillation_stage')
        if attention_distillation_stage == 2:
            # remove extra teacher attention modules so we can load this model normally in stage 3 and beyond
            for layer in model.model.layers:
                layer.self_attn = layer.self_attn.student_attn

        model.save_pretrained(config.train.output_dir)

    else:
        # hacky quick eval

        model.requires_grad_(False)
        model.eval()

        eval_dataset = train_dataset.take(100)
        device = 'cuda:0'
        model.to(device)
        loss = torch.tensor(0.0, device=device, dtype=torch.float)
        steps = 0
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(eval_dataset, batch_size=8, )
        with torch.no_grad():
            for batch in eval_dataloader:
                #print({k: v.shape for k, v in batch.items()})
                #exit()
                steps += 1
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch, return_dict=True)
                loss = loss + outputs.loss
        print("loss", float(loss) / steps)

        # training_args.set_evaluate(loss_only=True, batch_size=training_args.per_device_train_batch_size, accumulation_steps=1)
        # trainer = TrainerType(
        #     model = model,
        #     args = training_args,
        #     tokenizer = tokenizer,
        #     compute_metrics=lambda pred: {},
        #     preprocess_logits_for_metrics=lambda logits, labels: torch.argmax(logits[0]),
        # )
        # print(trainer.predict(test_dataset=train_dataset.take(1024)).metrics)

if __name__ == "__main__":
    main()
