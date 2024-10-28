import sys, os
import transformers # just for a bugfix for 0.4.2 of lm_eval

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from configs import parse_cmdline_configs
from pydoc import locate

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class CLI_Config:
    tokenizer_path: str
    model_path: str
    attn_path: str = 'rwkv6attn.RWKV6Attention'
    tasks: str = 'lambada_openai' # arc_challenge, arc_easy, headqa, openbookqa, hellaswag, winogrande, piqa, record, copa, storycloze_2016
    bsz: int|str = 'auto'
    precision: int | str = 'bf16'
    num_fewshot: int = 0
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

# FIXME - hardcoded for now, but it'd be great if we could specify this in data somewhere per model type (or even analyze the weights to see)
# NOTE - when loading a custom Qwen2RWKV model we don't need to set model_config.attention_bias and model_config.attention_output_bias, because the model config contains it
if 'Qwen/Qwen' in config.model_path:
    model_config.attention_bias = True
    model_config.attention_output_bias = False

# replace attention classes
ReplacementSelfAttentionType = locate(config.attn_path)
assert isinstance(ReplacementSelfAttentionType, Callable)
attn_classes_dict = locate(config.attn_classes_path)
assert isinstance(attn_classes_dict, dict), 'could not find attention classes dict at path provided'
for key in list(attn_classes_dict.keys()):
    attn_classes_dict[key] = ReplacementSelfAttentionType

model = AutoModelForCausalLM.from_pretrained(config.model_path, config=model_config, torch_dtype=dtype, device_map='cuda')

# UNNECESSARY because leave the config saying that they're tied, so it will treat them as tied even though they're both saved
# re-tie embeddings
#model.get_output_embeddings().weight = model.get_input_embeddings().weight

tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)

#device = 'cuda'
#model = model.to(device=device, dtype=dtype)
model.eval()

eval_tasks = config.tasks.split(',')

if config.seed is None:
    config.seed = 1234 

adapter = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=config.bsz)
with torch.no_grad():
    with torch.amp.autocast(device_type='cuda', dtype=dtype):
	    results = evaluator.simple_evaluate(
	        model=adapter,
	        tasks=eval_tasks,
	        #provide_description=False,
	        num_fewshot=config.num_fewshot,
	        limit=None,
	        bootstrap_iters=10000,
	        numpy_random_seed = config.seed,
	        torch_random_seed = config.seed,
	        fewshot_random_seed = config.seed,
	    )

print(results['results'])
