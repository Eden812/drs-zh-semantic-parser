# -*- coding: utf-8 -*-

import os
import time
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from transformers import MBartForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

from tokenization_mlm import MLMTokenizer
from utils.dataset_mlm_zh import LMMIterator
from utils.helper import shift_tokens_right
from utils.polynomial_lr_decay import PolynomialLRDecay
from pathlib import Path
device = "cuda" if cuda.is_available() else "cpu"

# 强制离线，避免任何 HF Hub 访问
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    ratio = 100 * trainable_params / all_param if all_param > 0 else 0.0
    print(
        f"[Info] trainable params: {trainable_params:,d} | "
        f"all params: {all_param:,d} | "
        f"trainable%: {ratio:.4f}"
    )


def evaluate(model, valid_loader, tokenizer, loss_fn, step, max_valid_batches=1000):
    loss_list = []
    model.eval()

    with torch.no_grad():
        for j, batch in enumerate(valid_loader):
            if max_valid_batches is not None and j >= max_valid_batches:
                break

            src, tgt = map(lambda x: x.to(device), batch)
            mask = src.ne(tokenizer.pad_token_id).long()

            decoder_input = shift_tokens_right(
                tgt,
                tokenizer.pad_token_id,
                model.config.decoder_start_token_id
            )

            with autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=src,
                    attention_mask=mask,
                    decoder_input_ids=decoder_input
                )
                loss = loss_fn(
                    outputs.logits.view(-1, len(tokenizer)),
                    tgt.view(-1)
                )

            loss_list.append(loss.item())

    model.train()
    avg_loss = float(np.mean(loss_list)) if loss_list else float("inf")
    print("[Info] valid {:05d} | loss {:.4f}".format(step, avg_loss))
    return avg_loss


def save_lora_checkpoint(model, tokenizer, save_dir, opt, step, best_loss):
    os.makedirs(save_dir, exist_ok=True)

    # 保存 LoRA adapter
    model.save_pretrained(save_dir)

    # 保存 tokenizer
    tokenizer.save_pretrained(save_dir)

    # 额外保存训练元信息
    meta = {
        "step": step,
        "best_loss": best_loss,
        "model_dir": opt.model_dir,
        "init_ckpt": opt.init_ckpt,
        "lang": opt.lang,
        "max_len": opt.max_len,
        "max_lr": opt.max_lr,
        "min_lr": opt.min_lr,
        "warmup_steps": opt.warmup_steps,
        "decap_steps": opt.decap_steps,
        "lora_r": opt.lora_r,
        "lora_alpha": opt.lora_alpha,
        "lora_dropout": opt.lora_dropout,
        "lora_target_modules": opt.lora_target_modules,
    }
    with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Info] The checkpoint has been updated: {save_dir}")


def main():
    parser = argparse.ArgumentParser()

    # 基本训练参数
    parser.add_argument("-seed", default=42, type=int, help="random seed")
    parser.add_argument("-stage", default="sft", type=str, help="fine-tuning stage name")
    parser.add_argument("-max_lr", default=3e-5, type=float, help="max learning rate")
    parser.add_argument("-min_lr", default=1e-5, type=float, help="min learning rate")
    parser.add_argument("-max_len", default=256, type=int, help="max sequence length")
    parser.add_argument("-acc_steps", default=8, type=int, help="gradient accumulation steps")
    parser.add_argument("-warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("-decap_steps", default=10000, type=int, help="max decay steps")
    parser.add_argument("-epoch", default=30, type=int, help="max epochs")
    parser.add_argument("-batch_size", default=1, type=int, help="mini batch size")
    parser.add_argument("-patience", default=6, type=int, help="early stopping patience")
    parser.add_argument("-eval_step", default=500, type=int, help="evaluate every x steps")
    parser.add_argument("-log_step", default=50, type=int, help="print log every x steps")
    parser.add_argument(
        "-lang", nargs="+", required=True,
        help="use target langs only, e.g. zh_CN"
    )

    # 模型与初始化
    parser.add_argument(
        "--init_ckpt", default=None, type=str,
        help="full checkpoint to initialize from, e.g. checkpoints/mlm_spt_mlm_zh.chkpt"
    )
    parser.add_argument(
        "--model_dir",
        default=str((Path(__file__).resolve().parent / "models" / "mbart-large-50-mlm-zh")),
        type=str,
        help="expanded base model dir (local path only)"
    )

    # LoRA 参数
    parser.add_argument("--lora_r", default=8, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int, help="LoRA alpha")
    parser.add_argument("--lora_dropout", default=0.1, type=float, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["q_proj", "v_proj"],
        help="target module names for LoRA"
    )

    # checkpoint 保存目录
    parser.add_argument(
        "--save_dir",
        default="checkpoints/ft_sft_lora_best",
        type=str,
        help="directory to save best LoRA adapter"
    )

    opt = parser.parse_args()
    print("[Info]", opt)
    torch.manual_seed(opt.seed)

    model_path = opt.model_dir

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Local model_dir not found: {model_path}")

    # 1) 加载 base model（强制本地）
    base_model = MBartForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True
    )

    # 关闭 gradient checkpointing，避免 LoRA backward 失败
    # base_model.gradient_checkpointing_enable()
    base_model.config.use_cache = False

    # 2) 加载 S-PT 全量 checkpoint（如果提供）
    if opt.init_ckpt is not None:
        if not os.path.isfile(opt.init_ckpt):
            raise FileNotFoundError(f"init_ckpt not found: {opt.init_ckpt}")
        print(f"[Info] loading init checkpoint: {opt.init_ckpt}")
        state_dict = torch.load(opt.init_ckpt, map_location="cpu")
        base_model.load_state_dict(state_dict, strict=True)

    # 3) 挂 LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=opt.lora_r,
        lora_alpha=opt.lora_alpha,
        lora_dropout=opt.lora_dropout,
        target_modules=opt.lora_target_modules,
    )

    model = get_peft_model(base_model, lora_config)
    model = model.to(device).train()

    # tokenizer（强制本地）
    tokenizer = MLMTokenizer.from_pretrained(
        model_path,
        src_lang="en_XX",
        local_files_only=True
    )
    pad_token_id = tokenizer.pad_token_id

    print(f"[Info] model_dir={model_path}")
    print(
        f"[Info] len(tokenizer)={len(tokenizer)} | "
        f"pad_token_id={tokenizer.pad_token_id} | "
        f"mask_token={tokenizer.mask_token} | "
        f"mask_token_id={tokenizer.mask_token_id}"
    )
    print(f"[Info] LoRA target modules = {opt.lora_target_modules}")
    print_trainable_parameters(model)

    # 数据
    train_loader, valid_loader = LMMIterator(opt, pad_token_id).loader

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98),
        eps=1e-9,
        lr=opt.max_lr
    )

    scheduler = PolynomialLRDecay(
        optimizer,
        warmup_steps=opt.warmup_steps,
        max_decay_steps=opt.decap_steps,
        end_learning_rate=opt.min_lr,
        power=2
    )

    scaler = GradScaler("cuda")

    tab = 0
    step = 0
    best_loss = 1e9
    loss_list = []
    start = time.time()

    for epoch in range(opt.epoch):
        for batch in train_loader:
            step += 1
            src, tgt = map(lambda x: x.to(device), batch)

            mask = src.ne(tokenizer.pad_token_id).long()
            decoder_input = shift_tokens_right(
                tgt,
                tokenizer.pad_token_id,
                model.config.decoder_start_token_id
            )

            with autocast("cuda", dtype=torch.float16):
                outputs = model(
                    input_ids=src,
                    attention_mask=mask,
                    decoder_input_ids=decoder_input
                )
                loss = loss_fn(
                    outputs.logits.view(-1, len(tokenizer)),
                    tgt.view(-1)
                )

            loss_list.append(loss.item())

            loss = loss / opt.acc_steps
            scaler.scale(loss).backward()

            if step % opt.acc_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if step % opt.log_step == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    "[Info] steps {:05d} | loss {:.4f} | lr {:.6f} | second {:.2f}".format(
                        step, float(np.mean(loss_list)), lr, time.time() - start
                    )
                )
                loss_list = []
                start = time.time()

            if ((len(train_loader) > opt.eval_step and step % opt.eval_step == 0)
                    or (len(train_loader) < opt.eval_step and step % len(train_loader) == 0)):

                eval_loss = evaluate(
                    model=model,
                    valid_loader=valid_loader,
                    tokenizer=tokenizer,
                    loss_fn=loss_fn,
                    step=step,
                    max_valid_batches=1000
                )

                if best_loss >= eval_loss:
                    best_loss = eval_loss
                    tab = 0
                    save_lora_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        save_dir=opt.save_dir,
                        opt=opt,
                        step=step,
                        best_loss=best_loss
                    )
                else:
                    tab += 1

                if tab == opt.patience:
                    print("[Info] Early stopping triggered.")
                    return


if __name__ == "__main__":
    main()