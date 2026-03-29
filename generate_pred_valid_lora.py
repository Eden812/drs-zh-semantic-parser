# -*- coding: utf-8 -*-

import os
import sys
import torch
from transformers import MBartForConditionalGeneration
from peft import PeftModel

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR

MODEL_DIR = BASE_DIR / "models" / "mbart-large-50-mlm-zh"
SPT_CKPT = BASE_DIR / "models" / "mlm_spt_mlm_zh.chkpt"
LORA_CKPT = BASE_DIR / "models" / "ft_sft_lora_struct_best"

INPUT_FILE = BASE_DIR / "data" / "sample" / "input.txt"
OUTPUT_FILE = BASE_DIR / "pred_valid_lora_struct.sbn"

MAX_LEN = 256
GEN_MAX_LEN = 128
BATCH_SIZE = 8
NUM_BEAMS = 5


def load_tokenizers():
    print("===== LOAD TOKENIZERS =====")
    tokenizer_zh = MLMTokenizer.from_pretrained(
        MODEL_DIR,
        src_lang="zh_CN",
        local_files_only=True
    )
    tokenizer_drs = MLMTokenizer.from_pretrained(
        MODEL_DIR,
        src_lang="<drs>",
        local_files_only=True
    )

    zh_probe = tokenizer_zh("测试", return_tensors="pt")
    drs_probe = tokenizer_drs("test", return_tensors="pt")

    zh_bos = zh_probe["input_ids"][0][0].item()
    drs_bos = drs_probe["input_ids"][0][0].item()

    print("len(tokenizer_zh) =", len(tokenizer_zh))
    print("mask_token_id     =", tokenizer_zh.mask_token_id)
    print("zh bos(real)      =", zh_bos)
    print("drs bos(real)     =", drs_bos)

    return tokenizer_zh, tokenizer_drs, drs_bos


def load_model():
    print("===== LOAD BASE MODEL =====")
    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    print("===== LOAD SPT CHECKPOINT =====")
    state_dict = torch.load(SPT_CKPT, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    print("===== LOAD LORA CHECKPOINT =====")
    model = PeftModel.from_pretrained(model, LORA_CKPT)

    model.to(device)
    model.eval()
    return model


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def clean_output(text):
    return text.replace("</s>", "").replace("<s>", "").strip()


def generate_batch(model, tokenizer_zh, tokenizer_drs, drs_bos, texts):
    inputs = tokenizer_zh(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=drs_bos,
            max_length=GEN_MAX_LEN,
            num_beams=NUM_BEAMS,
            repetition_penalty=1.1,
            early_stopping=True
        )

    outputs = []
    for ids in output_ids:
        raw = tokenizer_drs.decode(ids, skip_special_tokens=False)
        outputs.append(clean_output(raw))
    return outputs


def main():
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    if not os.path.isfile(SPT_CKPT):
        raise FileNotFoundError(f"SPT_CKPT not found: {SPT_CKPT}")
    if not os.path.isdir(LORA_CKPT):
        raise FileNotFoundError(f"LORA_CKPT dir not found: {LORA_CKPT}")
    if not os.path.isfile(INPUT_FILE):
        raise FileNotFoundError(f"INPUT_FILE not found: {INPUT_FILE}")

    tokenizer_zh, tokenizer_drs, drs_bos = load_tokenizers()
    model = load_model()

    texts = read_lines(INPUT_FILE)
    print(f"===== READ INPUT =====")
    print(f"num_lines = {len(texts)}")
    print(f"input_file = {INPUT_FILE}")
    print(f"output_file = {OUTPUT_FILE}")

    preds = []
    total = len(texts)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_texts = texts[start:end]
        batch_preds = generate_batch(model, tokenizer_zh, tokenizer_drs, drs_bos, batch_texts)
        preds.extend(batch_preds)

        if end % 100 == 0 or end == total:
            print(f"[Info] generated {end}/{total}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for line in preds:
            f.write(line + "\n")

    print("===== DONE =====")
    print(f"saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()