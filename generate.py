import sys, os

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from configs import parse_cmdline_configs
from pydoc import locate

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class CLI_Config:
    tokenizer_path: str
    model_path: str
    attn_path: str = 'rwkv6attn.RWKV6Attention'
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
    # replace attention classes
    ReplacementSelfAttentionType = locate(config.attn_path)
    assert isinstance(ReplacementSelfAttentionType, Callable)
    attn_classes_dict = locate(config.attn_classes_path)
    assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
    for key in list(attn_classes_dict.keys()):
        attn_classes_dict[key] = ReplacementSelfAttentionType

model = AutoModelForCausalLM.from_pretrained(config.model_path, config=model_config, torch_dtype=dtype, device_map='cuda')

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

#device = 'cuda'
#model = model.to(device=device, dtype=dtype)
model.eval()

if config.seed is None:
    config.seed = 1234 

from transformers import AutoTokenizer, Qwen2ForCausalLM

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": config.prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer(text, return_tensors="pt").to('cuda')

# Generate
for i in range(config.attempts):
    print(f"Attempt {i+1}:")
    generate_ids = model.generate(inputs.input_ids, max_new_tokens=config.max_len, use_cache=False, do_sample=True, temperature=1.0, top_p=1.0)#, typical_p=0.95)#top_p=0.7, repetition_penalty=0.25)
    print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False, use_cache=False)[0])
