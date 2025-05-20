import os
import json
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
from transformers import (
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoImageProcessor,
    StoppingCriteria,
)

# === 사용자 설정 ===
model_name_or_path = "Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
frame_path = '/home/user16/HT/LLoVi/output/egoschema/hcqa_selected_frames_full.json'
img_path = '/home/user16/HT/VideoTree/egoschema_frames/'
output_path = frame_path[:-5] + '_caps'
NUM_GPUS = torch.cuda.device_count()

# === 보조 함수 ===
def sort_image_list(image_list):
    return sorted(image_list, key=lambda x: int(x.split('.')[0]))

def apply_prompt_template(prompt):
    return (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
    )

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

# === 워커 함수 ===
def worker(rank, all_keys, frame_idx, img_path, output_path):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # 모델과 토크나이저 초기화
    model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
    tokenizer = model.update_special_tokens(tokenizer)
    tokenizer.padding_side = "left"
    tokenizer.eos_token = '<|end|>'
    image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

    # GPU마다 처리할 데이터 할당
    local_keys = [k for i, k in enumerate(all_keys) if i % NUM_GPUS == rank]
    local_result = {}

    for vid in tqdm(local_keys, desc=f"[GPU {rank}] Processing"):
        data = frame_idx['data'][vid]
        top_k_indices = sorted(data['pred'])
        sorted_values = data['frames']

        frame_list = sort_image_list(os.listdir(os.path.join(img_path, vid)))
        tmp = []

        for j in top_k_indices:
            image_path = os.path.join(img_path, vid, frame_list[sorted_values[j]])
            image = Image.open(image_path).convert("RGB")

            prompt = apply_prompt_template("Write a specific description of a first person view image.")
            image_inputs = image_processor([image], image_aspect_ratio="anyres")["pixel_values"].to(device)
            inputs = {"pixel_values": [[image_inputs]]}
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update({k: v.to(device) for k, v in language_inputs.items()})

            with torch.no_grad():
                generated_text = model.generate(
                    **inputs,
                    image_size=[[image.size]],
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=256,
                    stopping_criteria=[EosListStoppingCriteria()]
                )

            caption = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0].strip()
            tmp.append(caption)

        data['captions'] = tmp
        local_result[vid] = data

    # GPU별 중간 결과 저장
    with open(f'{output_path}_gpu{rank}.json', 'w') as f:
        json.dump(local_result, f, indent=2)

# === 멀티 프로세싱 실행 함수 ===
def run_multi_gpu(frame_idx, img_path, output_path):
    mp.set_start_method("spawn", force=True)
    all_keys = list(frame_idx['data'].keys())
    processes = []

    for rank in range(NUM_GPUS):
        p = mp.Process(target=worker, args=(rank, all_keys, frame_idx, img_path, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# === 결과 병합 ===
def merge_outputs(output_path, final_output_path):
    import glob

    final_json = {'data': {}}
    for fpath in glob.glob(f"{output_path}_gpu*.json"):
        partial = json.load(open(fpath))
        final_json['data'].update(partial)

    with open(final_output_path, 'w') as f:
        json.dump(final_json, f, indent=2)
    print(f"✅ 최종 결과 저장됨: {final_output_path}")

# === 실행 ===
if __name__ == "__main__":
    frame_idx = json.load(open(frame_path))
    run_multi_gpu(frame_idx, img_path, output_path)
    merge_outputs(output_path, f"{output_path}.json")


# import json
# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# import torch
# from PIL import Image
# import random
# from tqdm import tqdm
# import os
# from copy import deepcopy

# frame_path = '/home/user16/HT/VideoTree/new/mdf_new_question_image_top10_with_explanation.json'
# frame_idx = json.load(open(frame_path))

# img_path = '/home/user16/HT/VideoTree/egoschema_frames/'

# model_name = "Salesforce/instructblip-vicuna-7b"

# model = InstructBlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16)#, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16)
# processor = InstructBlipProcessor.from_pretrained(model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device).eval()

# # 프레임 정렬 함수
# def sort_image_list(image_list):
#     return sorted(image_list, key=lambda x: int(x.split('.')[0]))

# output = []
# for i, data in enumerate(tqdm(frame_idx)):
#     vid = data['name']
#     top_k_indices = data['top_k_indices']
#     sorted_values = data['sorted_values']

#     new_json = deepcopy(data)

#     frame_list = os.listdir(os.path.join(img_path, vid))
#     frame_list = sort_image_list(frame_list)

#     tmp = []
#     for j in top_k_indices:
#         image = Image.open(os.path.join(img_path, vid, frame_list[sorted_values[j]])).convert("RGB")

#         prompt = "Write a detailed description of a first person view image."
#         inputs = processor(text=prompt, images=image, return_tensors="pt").to(device="cuda", dtype=torch.bfloat16)

#         # while len(tmp) != 5:
#         with torch.no_grad():
#             outputs = model.generate(
#                     **inputs,
#                     do_sample=False,
#                     # num_beams=5,
#                     max_new_tokens=256,
#                     # top_p=0.9,
#                     # repetition_penalty=1.5,
#                     # length_penalty=1.0,
#                     # temperature=1,
#             )
#         generated_text = processor.batch_decode(outputs, skip_special_tokens=True)#[0].strip()
#         for gen in generated_text:
#             tmp.append(gen.strip())
#             if i < 10:
#                 print(gen)

#     new_json['top_k_captions'] = tmp
#     output.append(new_json)
    
#     if i % 20 == 0:
#         with open(f'{frame_path[:-5]}_caps_blip.json', 'w') as f:
#             json.dump(output, f, indent=2)

# with open(f'{frame_path[:-5]}_caps_blip.json', 'w') as f:
#     json.dump(output, f, indent=2)
