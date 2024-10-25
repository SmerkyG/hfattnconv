import sys
import torch
from safetensors.torch import save_file
from huggingface_hub import save_torch_state_dict

if len(sys.argv) != 3:
    print(f"Converts from .pth to .safetensors")
    print("Usage: python convert_to_safetensors.py in_file out_file")
    exit()

in_model_path = sys.argv[1]
out_model_path = sys.argv[2]

print("Loading file...")
state_dict = torch.load(in_model_path, map_location='cpu', weights_only=True)
if out_model_path.endswith('.safetensors'):
    print("Saving file...")
    save_file(state_dict, out_model_path, metadata=dict(format='pt'))
else:
    print("Saving model chunks...")
    save_torch_state_dict(state_dict, out_model_path)
print("Done!")
