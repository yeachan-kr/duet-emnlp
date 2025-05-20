from pathlib import Path
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, CLIPTokenizer
from datasets import load_dataset

ds = load_dataset("google-research-datasets/conceptual_captions", "unlabeled")

def save_features(features, save_folder):
    """
    Save stacked features to a .pt file in a specified folder.

    Args:
    - features (torch.Tensor): Tensor containing all stacked features
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filepath = os.path.join(save_folder, "features_ms.pt")
    torch.save(features, filepath)


def save_captions(captions, save_folder):
    """
    Save captions to a JSON file in a specified folder.

    Args:
    - captions (list): List of captions to save
    - save_folder (str): Path to the folder where the file should be saved

    Returns:
    - None
    """
    filepath = os.path.join(save_folder, "captions_ms.json")
    with open(filepath, 'w') as f:
        json.dump(captions, f, indent=4)


def load_json(fn):
    with open(fn, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, fn, indent=4):
    with open(fn, 'w') as f:
        json.dump(data, f, indent=indent)


# Model configuration
model_name_or_path = "BAAI/EVA-CLIP-8B"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
print("Loading tokenizer and model...")
tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(
    model_name_or_path, 
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device).eval()


# Feature extraction

def extract_text_features_from_json(json_path, save_folder, batch_size=32):
    os.makedirs(save_folder, exist_ok=True)

    # Load captions from JSON
    print(f"Loading captions from {json_path}...")
    data = load_json(json_path)
    captions = [entry['caption'] for entry in tqdm(data['annotations'])]
    captions = list(set(captions))

    all_captions = []
    all_features = []

    pbar = tqdm(total=len(captions), desc="Extracting features")

    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        # Process captions
        input_ids = tokenizer(batch_captions, return_tensors="pt", padding='longest', truncation=True, max_length=77).input_ids.to(device)

        with torch.no_grad():
            # Extract features
            text_features = model.encode_text(input_ids)

            # # Normalize features
            # text_features /= text_features.norm(dim=-1, keepdim=True)

            # Append features
            all_features.append(text_features)

        all_captions.extend(batch_captions)
        pbar.update(len(batch_captions))
        # break
    pbar.close()

    # Stack all features and save
    all_features = torch.cat(all_features, dim=0).squeeze()
    save_features(all_features, save_folder)

    # Save all captions
    save_captions(all_captions, save_folder)


def extract_text_features_from_ds(ds, save_folder, batch_size=32):
    os.makedirs(save_folder, exist_ok=True)

    # Load captions from JSON
    print(f"Loading captions from HuggingFace...")
    captions = [entry['caption'] for entry in tqdm(ds['train'])]
    captions = list(set(captions))

    all_captions = []
    all_features = []

    pbar = tqdm(total=len(captions), desc="Extracting features")

    for batch_start in range(0, len(captions), batch_size):
        batch_captions = captions[batch_start:batch_start + batch_size]

        # Process captions
        input_ids = tokenizer(batch_captions, return_tensors="pt", padding='longest', truncation=True, max_length=77).input_ids.to(device)

        with torch.no_grad():
            # Extract features
            text_features = model.encode_text(input_ids).cpu()

            # # Normalize features
            # text_features /= text_features.norm(dim=-1, keepdim=True)

            # Append features
            all_features.append(text_features)

        all_captions.extend(batch_captions)
        pbar.update(len(batch_captions))
        # break
    pbar.close()

    # Stack all features and save
    all_features = torch.cat(all_features, dim=0).squeeze()
    save_features(all_features, save_folder)

    # Save all captions
    save_captions(all_captions, save_folder)


if __name__ == "__main__":
    json_path = "/home/user16/HT/data/annotations/captions_train2017.json"  # Path to the JSON file
    save_folder = "/home/user16/HT/VideoTree/"  # Update with your desired save path

    extract_text_features_from_json(json_path, save_folder, batch_size=128)
    # extract_text_features_from_ds(ds, save_folder, batch_size=256)