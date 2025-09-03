"""Fine-tune the decoder-only model on an instruction dataset.

This script:
- Loads a pre-trained checkpoint from models/decoder_only_latest.pth (by default)
- Builds a dataset from instruct_dataset.txt using the saved vocab
- Fine-tunes the model
- Saves a timestamped and a *_latest.pth checkpoint for the finetuned model
"""

import os
import sys
import time
import math
import datetime

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split, Dataset

# Ensure we can import project modules when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.decoder_only import (  # pylint: disable=C0413
    TransformerModel,
    TextDataset,
    collate,
    train_epoch,
    eval_epoch,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    DEVICE,
    BATCH_SIZE as DEFAULT_BATCH_SIZE,
    INPUT_MAX_SEQ_LEN as DEFAULT_SEQ_LEN,
    tokenize,
    encode,
)


# =====================
# Global Configuration
# =====================
# Edit these variables to configure fine-tuning without CLI args
BASE_MODEL_PATH = "models/decoder_only_latest.pth"
DATASET_PATH = "Datasets/finetune_qa.txt"
EPOCHS = 5
BATCH_SIZE = 16  # smaller batch for longer sequences
SEQ_LEN = 128  # max total length (BOS + question + answer + EOS)
LEARNING_RATE = 5e-5
VAL_SPLIT = 0.1
SAVE_PREFIX = "models/decoder_only_qa_finetuned_small"


class QADataset(Dataset):
    """Dataset for QA fine-tuning.

    For each (Question, Answer) pair, constructs a single sequence:
        [<bos>, "question", <q tokens> , "answer", <a tokens>, <eos>]

    Inputs are the sequence without the last token; targets are the sequence
    without the first token. All target positions that correspond to the prompt
    tokens (BOS + "question" + question tokens + "answer") are set to pad_id so
    they are ignored by the loss.
    """

    def __init__(self, text: str, vocab: dict, max_len: int, pad_id: int):
        self.vocab = vocab
        self.max_len = max(4, int(max_len))  # ensure room for BOS/EOS
        self.pad_id = pad_id
        self.inputs: list[list[int]] = []
        self.targets: list[list[int]] = []
        self._num_pairs = 0
        self._avg_input_len = 0.0
        self._avg_answer_len = 0.0

        pairs = self._parse_pairs(text)
        self._num_pairs = len(pairs)

        total_input_len = 0
        total_answer_len = 0

        bos = vocab[BOS_TOKEN]
        eos = vocab[EOS_TOKEN]

        for q, a in pairs:
            # Tokenize
            q_tokens = tokenize(q)
            a_tokens = tokenize(a)

            prompt_tokens = ["question"] + q_tokens + ["answer"]
            # Build id sequence with explicit BOS/EOS
            prompt_ids = encode(prompt_tokens, vocab)
            answer_ids = encode(a_tokens, vocab)

            full_ids = [bos] + prompt_ids + answer_ids + [eos]

            # Build language modeling inputs/targets
            inp = full_ids[:-1]
            tgt = full_ids[1:]

            # Mask loss on prompt positions in targets
            prompt_len = 1 + len(prompt_ids)  # includes BOS

            # If sequence too long, keep the tail (to prioritize answer tokens)
            if len(inp) > self.max_len:
                start = len(inp) - self.max_len
                inp = inp[start:]
                tgt = tgt[start:]
                prompt_len = max(0, prompt_len - start)

            if prompt_len > 0:
                # Set prompt labels to pad to ignore in loss
                for i in range(min(prompt_len, len(tgt))):
                    tgt[i] = self.pad_id

            self.inputs.append(inp)
            self.targets.append(tgt)

            total_input_len += len(inp)
            total_answer_len += len(answer_ids)

        if self._num_pairs > 0:
            self._avg_input_len = total_input_len / self._num_pairs
            self._avg_answer_len = total_answer_len / self._num_pairs

    @staticmethod
    def _parse_pairs(text: str) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        current_q: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("question:"):
                current_q = line.split(":", 1)[1].strip()
            elif low.startswith("answer:"):
                a = line.split(":", 1)[1].strip()
                if current_q is not None:
                    pairs.append((current_q, a))
                    current_q = None
        return pairs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def stats(self) -> dict:
        return {
            "num_pairs": self._num_pairs,
            "avg_input_len": round(self._avg_input_len, 1),
            "avg_answer_len": round(self._avg_answer_len, 1),
            "max_len": self.max_len,
        }


def load_base_checkpoint(ckpt_path: str):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Base model checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if not isinstance(ckpt, dict) or "model_state" not in ckpt or "vocab" not in ckpt:
        raise ValueError(
            "Checkpoint must be a dict containing 'model_state' and 'vocab'"
        )
    vocab = ckpt["vocab"]
    model = TransformerModel(vocab).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
    return model, vocab


def main():
    start_time = time.time()
    print(f"Using device: {DEVICE}")
    print(f"Loading base model from: {BASE_MODEL_PATH}")

    # 1) Load base model and vocab
    model, vocab = load_base_checkpoint(BASE_MODEL_PATH)
    pad_id = vocab[PAD_TOKEN]

    # 2) Load dataset text
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset_text = f.read()
    print(f"Loaded dataset with {len(dataset_text):,} characters from {DATASET_PATH}")

    # 3) Build QA dataset and splits using existing vocab (unknown tokens -> <unk>)
    full_ds = QADataset(dataset_text, vocab, SEQ_LEN, pad_id)
    if len(full_ds) < 2:
        raise ValueError(
            "Dataset is too small after preprocessing. Provide more data or reduce seq_len."
        )

    # Print brief dataset stats
    if hasattr(full_ds, "stats"):
        print("QA dataset stats:", full_ds.stats())

    val_size = max(1, int(VAL_SPLIT * len(full_ds)))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    collate_fn = lambda b: collate(b, pad_id=pad_id)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4) Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    # 5) Fine-tuning loop
    epochs_range = range(1, EPOCHS + 1)
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for ep in epochs_range:
        epoch_start = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_dl, optimizer, loss_criterion, pad_id
        )
        vl_loss, vl_acc = eval_epoch(model, val_dl, loss_criterion, pad_id)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        train_accs.append(tr_acc)
        val_accs.append(vl_acc)

        tr_ppl = math.exp(tr_loss)
        vl_ppl = math.exp(vl_loss)
        elapsed = int(time.time() - epoch_start)
        mm, ss = divmod(elapsed, 60)
        print(
            f"Epoch {ep:02d}/{EPOCHS} │ "
            f"train_loss={tr_loss:.3f} acc={tr_acc:.2%} ppl={tr_ppl:.1f} │ "
            f"val_loss={vl_loss:.3f} acc={vl_acc:.2%} ppl={vl_ppl:.1f} │ "
            f"Time: {mm}m {ss}s"
        )

    # 6) Save finetuned model
    os.makedirs("models", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base = os.path.basename(SAVE_PREFIX)
    dir_prefix = os.path.dirname(SAVE_PREFIX) or "models"
    os.makedirs(dir_prefix, exist_ok=True)

    model_save_path = os.path.join(dir_prefix, f"{base}_{ts}.pth")
    latest_model_path = os.path.join(dir_prefix, f"{base}_latest.pth")
    save_dict = {
        "model_state": model.state_dict(),
        "vocab": vocab,
    }
    torch.save(save_dict, model_save_path)
    torch.save(save_dict, latest_model_path)
    print(f"Finetuned model saved to {model_save_path} and {latest_model_path}")

    total_seconds = int(time.time() - start_time)
    m, s = divmod(total_seconds, 60)
    print(f"Total finetune runtime: {m}m {s}s")


if __name__ == "__main__":
    main()
