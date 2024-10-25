import sys, os
import transformers # just for a bugfix for 0.4.2 of lm_eval

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from configs import parse_cmdline_configs
from pydoc import locate

from rwkv6attn import RWKV6Attention

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from dataclasses import dataclass
from typing import Any

@dataclass
class CLI_Config:
    tokenizer_path: str
    model_path: str
    prompt:str = "Hey, are you conscious? Can you talk to me?"
    max_len:int = 30
    attempts:int = 3
    precision: int | str = 'bf16'
    attn_classes_path: str = 'transformers.models.qwen2.modeling_qwen2.QWEN2_ATTENTION_CLASSES' # 'transformers.models.llama.modeling_llama.LLAMA_ATTENTION_CLASSES' 
    seed: int | None = None
    train:Any = None

config, errors = parse_cmdline_configs(sys.argv[1:], CLI_Config)
if errors != '':
    print(errors)
    exit()

match config.precision:
    case 32:
        dtype = torch.float32
    case '32':
        dtype = torch.float32
    case 16:
        dtype = torch.float16
    case '16':
        dtype = torch.float16
    case 'bf16':
        dtype = torch.bfloat16
    case _:
        print("Bad precision type specified")
        exit()

# avoid 1000 huggingface warnings "huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...""
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print(f'Loading model - {config.model_path}')

model_config = AutoConfig.from_pretrained(config.model_path)

if config.model_path.startswith('.'):
    # FIXME - hardcoded for now, but it'd be great if we could specify this in data somewhere per model type (or even analyze the weights to see)
    # NOTE - when loading a custom Qwen2RWKV model we don't need to set model_config.attention_bias and model_config.attention_output_bias, because the model config contains it
    if 'Qwen/Qwen' in config.model_path:
        model_config.attention_bias = True
        model_config.attention_output_bias = False
    
    # replace attention classes
    attn_classes_dict = locate(config.attn_classes_path)
    assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
    for key in list(attn_classes_dict.keys()):
        attn_classes_dict[key] = RWKV6Attention

model = AutoModelForCausalLM.from_pretrained(config.model_path, config=model_config, torch_dtype=dtype, device_map='cuda')

# UNNECESSARY because leave the config saying that they're tied, so it will treat them as tied even though they're both saved
# re-tie embeddings
#model.get_output_embeddings().weight = model.get_input_embeddings().weight

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

#device = 'cuda'
#model = model.to(device=device, dtype=dtype)
model.eval()

if config.seed is None:
    config.seed = 1234 

from transformers import AutoTokenizer, Qwen2ForCausalLM

inputs = tokenizer(config.prompt, return_tensors="pt").to('cuda')

# Generate
for i in range(config.attempts):
    print(f"Attempt {i+1}:")
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=config.max_len, use_cache=False, do_sample=True, temperature=1.0, top_p=1.0)#, typical_p=0.95)#top_p=0.7, repetition_penalty=0.25)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, use_cache=False)[0])
