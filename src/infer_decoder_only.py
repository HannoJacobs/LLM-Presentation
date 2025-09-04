import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.decoder_only import *

BASE_MODEL = False

if BASE_MODEL:
    #### BASE MODEL ####
    #### BASE MODEL ####
    #### BASE MODEL ####
    # MODEL_PATH = "models/base_full_10_1024_8_4_2048.pth"
    MODEL_PATH = "models/base_nano_10_512_4_4_512.pth"
    INFER_TEXTS = [
        "The cat (Felis catus), also referred to as the domestic cat or house cat,",
        "The dog (Canis familiaris or Canis lupus familiaris) is a ",
        "Elephants are the largest living land animals. ",
        "The lion (Panthera leo) is a large cat of the genus Panthera,",
        "The tiger (Panthera tigris) is a large cat and a member of ",
    ]
    TEMP_CONFIGS = [
        (0.5, 5, ""),  # Low creativity + small top_k
        # (0.0, 1, "❄️"),  # Deterministic + top_1 (only best token)
        # (0.5, 5, "🌤️"),  # Low creativity + small top_k
        # (1.0, 20, "🌡️"),  # Balanced + medium top_k
        # (2.0, 50, "🔥"),  # High creativity + large top_k
    ]
else:
    #### FINETUNED MODEL ####
    #### FINETUNED MODEL ####
    #### FINETUNED MODEL ####
    # MODEL_PATH = "models/finetune_full_10_1024_8_4_2048_latest.pth"
    MODEL_PATH = "models/finetune_nano_10_512_4_4_512_latest.pth"
    INFER_TEXTS = [
        "Question: What is the scientific name of the cat?\nAnswer: ",
        "Question: When did the domestication of cats occur?\nAnswer: ",
        "Question: What is the estimated global dog population?\nAnswer: ",
        "Question: What is the largest living land animal?\nAnswer: ",
        "Question: How many living elephant species are currently recognised?\nAnswer: ",
    ]
    TEMP_CONFIGS = [
        (0.5, 5, ""),  # Low creativity + small top_k
        # (0.0, 1, "❄️"),  # Deterministic + top_1 (only best token)
        # (0.5, 5, "🌤️"),  # Low creativity + small top_k
        # (1.0, 20, "🌡️"),  # Balanced + medium top_k
        # (2.0, 50, "🔥"),  # High creativity + large top_k
    ]


def load_model(ckpt_path: str):
    device = DEVICE
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    inv_vocab = {i: w for w, i in vocab.items()}
    model = TransformerModel(vocab).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, inv_vocab


model, vocab, inv_vocab = load_model(MODEL_PATH)

print("\nTemperature + Top-K Effects Demo")
print("=" * 40)

for text in INFER_TEXTS:
    print(f"\nPrompt: {text}")
    print("-" * 30)

    for temp, top_k, emoji in TEMP_CONFIGS:
        generated = infer(
            model, text, vocab, inv_vocab, max_len=20, temperature=temp, top_k=top_k
        )
        print(f"{emoji} {temp:.1f}(k={top_k}): {generated}")

    print()
