"""
Inference Script for Decoder-Only Transformer Models

This script provides inference capabilities for trained decoder-only transformer language models.
It supports both base models (trained on general text) and fine-tuned models (optimized for Q&A).
The script demonstrates text generation with configurable temperature settings and top-k sampling.

Key Features:
- Load pre-trained model checkpoints with vocabularies
- Generate text completions for given prompts
- Support for both base and fine-tuned models
- Temperature-controlled creativity (0.0 = deterministic, higher = more creative)
- Top-k sampling for quality control
- Batch processing of multiple prompts
- Comparison between different model variants

Configuration Options:
- BASE_MODEL: Toggle between base model (True) and fine-tuned model (False)
- DEMO_TEMP: Enable temperature comparison demo with multiple settings
- MODEL_PATH: Path to the model checkpoint file
- INFER_TEXTS: List of prompts to generate completions for

Model Types:
- Base Model: General text generation trained on animal-related content
- Fine-tuned Model: Q&A optimized model for answering animal-related questions

Temperature Settings:
- 0.0: Deterministic generation (always same output)
- 0.5: Balanced creativity and coherence
- 1.0: Moderate creativity
- 2.0: High creativity with more diverse outputs

Usage:
    # Basic inference with base model
    python infer_decoder_only.py

    # Fine-tuned model inference
    # Set BASE_MODEL = False and update MODEL_PATH
    python infer_decoder_only.py

    # Temperature comparison demo
    # Set DEMO_TEMP = True
    python infer_decoder_only.py

Dependencies:
    - torch: Deep learning framework
    - src.decoder_only: Main model implementation and inference functions

Output:
    - Generated text completions for each prompt
    - Model type and configuration information
    - Temperature settings and generation parameters
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.decoder_only import *

BASE_MODEL = True
DEMO_TEMP = False

if BASE_MODEL:
    #### BASE MODEL ####
    #### BASE MODEL ####
    #### BASE MODEL ####
    MODEL_PATH = "models/base_nano_10_512_4_4_512.pth"
    INFER_TEXTS = [
        "The cat (Felis catus), also referred to as the domestic cat or house cat,",
        "The dog (Canis familiaris or Canis lupus familiaris) is a ",
        "Elephants are the largest living land animals. ",
        "The lion (Panthera leo) is a large cat of the genus Panthera,",
        "The tiger (Panthera tigris) is a large cat and a member",
    ]
else:
    #### FINETUNED MODEL ####
    #### FINETUNED MODEL ####
    #### FINETUNED MODEL ####
    # MODEL_PATH = "models/finetune_nano_10_512_4_4_512.pth"
    MODEL_PATH = "models/finetune_nano_10_512_4_4_512_temp_demo.pth"
    INFER_TEXTS = [
        "Question: Where are lions native to?\nAnswer: ",
        "Question: What are the main threats to tiger populations?\nAnswer: ",
        "Question: What is the estimated global dog population?\nAnswer: ",
        "Question: What are the distinctive features of elephants?\nAnswer: ",
        "Question: What roles do dogs perform for humans?\nAnswer: ",
    ]

if DEMO_TEMP:
    TEMP_CONFIGS = [
        (0.0, 1, "❄️"),  # Deterministic + top_1 (only best token)
        (0.5, 5, "🌤️"),  # Low creativity
        (1.0, 20, "🌡️"),  # Balanced
        (2.0, 50, "🔥"),  # High creativity
    ]
else:
    TEMP_CONFIGS = [
        (0.5, 5, ""),  # Low creativity + small top_k
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

for text in INFER_TEXTS:
    print_txt = text
    if not BASE_MODEL:
        print_txt = text[10:-9]
    print(f"\nPrompt: {print_txt}")

    if DEMO_TEMP:
        print("-" * 30)

    for temp, top_k, emoji in TEMP_CONFIGS:
        generated = infer(
            model, text, vocab, inv_vocab, max_len=20, temperature=temp, top_k=top_k
        )
        if DEMO_TEMP:
            print(f"{emoji} {temp:.1f}(k={top_k}): {generated}")
        else:
            print(f"LLM response: {generated}")

    print()
