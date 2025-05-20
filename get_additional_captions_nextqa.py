import json
import os
import torch
import requests
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria

from typing import List
tricks: List[str] = []
tricks: List[int] = []

# Define the prompt template
def apply_prompt_template(prompt):
    return (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
    )

# Stopping criteria for the model
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids      

# Load model and tokenizer
model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)
tokenizer.padding_side = "left"
tokenizer.eos_token = '<|end|>'

# Load frame data
frame_path = '/home/user16/HT/LLoVi/output/nextqa/gpt4_nextqa_frame_selection.json'
frame_idx = json.load(open(frame_path))

img_path = '/home/user16/HT/VideoTree/nextqa_llovi/'

caption_dict = {}
# Function to sort image file names correctly
def sort_image_list(image_list):
    return sorted(image_list, key=lambda x: int(x.split('.')[0].replace('frame_', '')))

# Process each video and generate captions
if os.path.exists(f'{frame_path[:-5]}_caps.json'):
    new_json = json.load(open(f'{frame_path[:-5]}_caps.json'))
else:
    new_json = deepcopy(frame_idx)

for i, (vid, data) in enumerate(tqdm(frame_idx['data'].items())):
    if new_json['data'][vid]['captions'] != None: continue

    uid = data['uid']
    top_k_indices = sorted(data['pred'])
    sorted_values = data['frames']
    
    frame_list = os.listdir(os.path.join(img_path, uid))
    # frame_list = sort_image_list(frame_list)

    tmp = []
    for j in top_k_indices:
        # Load image
        if uid + str(sorted_values[j]) in caption_dict.keys():
            generated_caption = caption_dict[uid + str(sorted_values[j])]
            tmp.append(generated_caption)
        else:
            image_path = os.path.join(img_path, uid, f'frame_{sorted_values[j]}.jpg')
            image = Image.open(image_path).convert("RGB")

            # Apply prompt template
            prompt = apply_prompt_template("Write a specific description of the image.")

            image_inputs = image_processor([image], image_aspect_ratio="anyres")["pixel_values"].cuda()
            inputs = {"pixel_values": [[image_inputs]]}  # Wrap pixel_values in nested lists
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update({k: v.cuda() for k, v in language_inputs.items()})

            # Generate caption
            with torch.no_grad():
                generated_text = model.generate(
                    **inputs, 
                    image_size=[[image.size]], 
                    pad_token_id=tokenizer.pad_token_id,
                    # do_sample=False, 
                    max_new_tokens=256, 
                    # top_p=None, 
                    # num_beams=1,
                    # repetition_penalty=1.5,
                    # length_penalty=1.0,
                    stopping_criteria=[EosListStoppingCriteria()]
                )

            # Decode output and append
            generated_caption = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0].strip()
            tmp.append(generated_caption)
            if i < 10:
                print(generated_caption)

            caption_dict[uid + str(sorted_values[j])] = generated_caption

    # Save results
    new_json['data'][vid]['captions'] = tmp
    
    # Save intermediate results every 20 iterations
    if i % 20 == 0:
        with open(f'{frame_path[:-5]}_caps.json', 'w') as f:
            json.dump(new_json, f, indent=2)

# Final save
with open(f'{frame_path[:-5]}_caps.json', 'w') as f:
    json.dump(new_json, f, indent=2)

print("Caption generation completed and saved.")
