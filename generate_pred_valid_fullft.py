# -*- coding: utf-8 -*-

import os
import torch
from transformers import MBartForConditionalGeneration
from tokenization_mlm import MLMTokenizer

from pathlib import Path
import torch
from transformers import MBartForConditionalGeneration
from tokenization_mlm import MLMTokenizer

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models" / "mbart-large-50-mlm-zh"
CKPT = BASE_DIR / "models" / "ft_sft_mlm_zh.chkpt"

INPUT_FILE = BASE_DIR / "data" / "sample" / "input.txt"
OUTPUT_FILE = BASE_DIR / "pred_valid_fullft.sbn"

device = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def load_tokenizers():
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

    # 用真实编码结果拿合法的 drs bos id
    drs_bos = tokenizer_drs("test", return_tensors="pt")["input_ids"][0][0].item()

    print("[Info] len(tokenizer) =", len(tokenizer_zh))
    print("[Info] mask_token_id =", tokenizer_zh.mask_token_id)
    print("[Info] drs bos(real) =", drs_bos)

    return tokenizer_zh, tokenizer_drs, drs_bos


def load_model():
    model = MBartForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True
    )

    state_dict = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    print("[Info] model loaded from:", CKPT)
    return model


def clean_output(text: str) -> str:
    text = text.replace("</s>", "").replace("<s>", "").strip()
    if text.startswith("<drs>"):
        text = text[len("<drs>"):].strip()
    return text


def generate_one(model, tokenizer_zh, tokenizer_drs, drs_bos, text: str) -> str:
    inputs = tokenizer_zh(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            forced_bos_token_id=drs_bos,
            max_length=128,
            num_beams=5,
            repetition_penalty=1.1,
            early_stopping=True
        )

    raw = tokenizer_drs.decode(output_ids[0], skip_special_tokens=False)
    return clean_output(raw)


def main():
    tokenizer_zh, tokenizer_drs, drs_bos = load_tokenizers()
    model = load_model()

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        src_lines = [line.rstrip("\n") for line in f]

    preds = []
    for i, line in enumerate(src_lines, 1):
        if not line.strip():
            preds.append("")
            continue

        pred = generate_one(model, tokenizer_zh, tokenizer_drs, drs_bos, line)
        preds.append(pred)

        if i % 100 == 0:
            print(f"[Info] processed {i}/{len(src_lines)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(p + "\n")

    print(f"[Done] wrote {len(preds)} lines to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()