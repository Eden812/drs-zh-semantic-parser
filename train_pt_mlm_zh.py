# -*- coding: utf-8 -*-

import os
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch import cuda
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_

from tokenization_mlm import MLMTokenizer
from transformers import MBartForConditionalGeneration

from utils.dataset_mlm_zh import token_mask
from utils.dataset_mlm_zh import LMMIterator
from utils.helper import batch_process
from utils.helper import shift_tokens_right
from utils.polynomial_lr_decay import PolynomialLRDecay

device = "cuda" if cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_safe_mask_token_id(tokenizer):
    """
    解决当前 tokenizer.mask_token_id 仍然指向原始 mBART 大词表 id 的问题。
    策略：
      1) 先用 tokenizer.mask_token_id
      2) 再尝试 convert_tokens_to_ids("<mask>")
      3) 如果都越界，回退到 unk_token_id
    """
    vocab_size = len(tokenizer)
    raw_mask_token_id = tokenizer.mask_token_id

    # 1) 直接用 tokenizer 自带
    if raw_mask_token_id is not None and 0 <= raw_mask_token_id < vocab_size:
        return raw_mask_token_id, raw_mask_token_id, False

    # 2) 尝试当前 tokenizer 里重新查 "<mask>"
    converted_mask_token_id = tokenizer.convert_tokens_to_ids("<mask>")
    if converted_mask_token_id is not None and 0 <= converted_mask_token_id < vocab_size:
        return raw_mask_token_id, converted_mask_token_id, False

    # 3) 回退到 unk_token_id
    unk_token_id = tokenizer.unk_token_id
    if unk_token_id is not None and 0 <= unk_token_id < vocab_size:
        return raw_mask_token_id, unk_token_id, True

    raise ValueError(
        f"Invalid mask token and unk token. "
        f"raw_mask_token_id={raw_mask_token_id}, "
        f"converted_mask_token_id={converted_mask_token_id}, "
        f"unk_token_id={unk_token_id}, "
        f"tokenizer_size={vocab_size}"
    )


def assert_ids_in_range(x: torch.Tensor, vocab_size: int, name: str):
    x_min = x.min().item()
    x_max = x.max().item()
    if x_min < 0 or x_max >= vocab_size:
        raise ValueError(
            f"{name} id out of range: min={x_min}, max={x_max}, vocab={vocab_size}"
        )


def evaluate(model, valid_loader, tokenizer, loss_fn, step, stage, mask_token_id, max_valid_batches=1000):
    """Evaluation function for model"""
    loss_list = []
    vocab_size = len(tokenizer)

    with torch.no_grad():
        model.eval()
        for j, batch in enumerate(valid_loader):
            if max_valid_batches is not None and j >= max_valid_batches:
                break

            src, tgt = map(lambda x: x.to(device), batch)

            if stage == "bpt":
                mask = src.ne(tokenizer.pad_token_id).long()
                src = token_mask(
                    src, mask.sum(-1), 0.35,
                    mask_token_id
                )
                mask = src.ne(tokenizer.pad_token_id).long()
            else:
                mask = tgt.ne(tokenizer.pad_token_id).long()
                tgt_mask = token_mask(
                    tgt, mask.sum(-1), 0.35,
                    mask_token_id
                )
                src = batch_process(src, tgt_mask, tokenizer.pad_token_id)
                mask = src.ne(tokenizer.pad_token_id).long()

            assert_ids_in_range(src, vocab_size, "valid/src")
            assert_ids_in_range(tgt, vocab_size, "valid/tgt")

            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id,
                model.config.decoder_start_token_id
            )

            with autocast("cuda", dtype=torch.float16):
                outputs = model(
                    src, mask,
                    decoder_input_ids=decoder_input
                )
                loss = loss_fn(
                    outputs.logits.view(-1, vocab_size),
                    tgt.view(-1)
                )

            loss_list.append(loss.item())

        model.train()

    avg_loss = float(np.mean(loss_list)) if loss_list else float("inf")
    print("[Info] valid {:05d} | loss {:.4f}".format(step, avg_loss))
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", default=42, type=int, help="random seed")
    parser.add_argument("-stage", default="bpt", type=str, help="training stage: bpt or spt")
    parser.add_argument("-max_lr", default=1e-4, type=float, help="max learning rate")
    parser.add_argument("-min_lr", default=1e-5, type=float, help="min learning rate")
    parser.add_argument("-max_len", default=128, type=int, help="max length of sequence")
    parser.add_argument("-acc_steps", default=8, type=int, help="accumulation steps")
    parser.add_argument("-warmup_steps", default=3000, type=int, help="warmup steps")
    parser.add_argument("-decap_steps", default=30000, type=int, help="max decay steps")
    parser.add_argument("-epoch", default=30, type=int, help="max epochs")
    parser.add_argument("-batch_size", default=8, type=int, help="mini batch size")
    parser.add_argument("-patience", default=6, type=int, help="early stopping")
    parser.add_argument("-eval_step", default=1000, type=int, help="eval every x step")
    parser.add_argument("-log_step", default=100, type=int, help="log every x step")
    parser.add_argument(
        "-lang", nargs="+", required=True,
        help="use target langs only, e.g. de_DE it_IT nl_XX zh_CN"
    )
    parser.add_argument(
        "--init_ckpt", default=None, type=str,
        help="optional checkpoint to initialize from; if omitted, use model_dir only"
    )
    parser.add_argument(
        "--model_dir", default="mbart-large-50-mlm-zh", type=str,
        help="expanded base model dir"
    )

    opt = parser.parse_args()
    print("[Info]", opt)
    torch.manual_seed(opt.seed)

    model_path = opt.model_dir

    model = MBartForConditionalGeneration.from_pretrained(model_path)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model = model.to(device).train()

    # spt 默认从 bpt ckpt 继续；也允许手动指定 init_ckpt
    if opt.init_ckpt is not None:
        print(f"[Info] loading init checkpoint: {opt.init_ckpt}")
        model.load_state_dict(torch.load(opt.init_ckpt, map_location="cpu"), strict=True)
    elif opt.stage == "spt":
        default_bpt = "checkpoints/mlm_bpt_mlm_zh.chkpt"
        if os.path.exists(default_bpt):
            print(f"[Info] loading default bpt checkpoint: {default_bpt}")
            model.load_state_dict(torch.load(default_bpt, map_location="cpu"), strict=True)
        else:
            print(f"[Warn] default bpt checkpoint not found: {default_bpt}")

    tokenizer = MLMTokenizer.from_pretrained(model_path, src_lang="en_XX")
    pad_token_id = tokenizer.pad_token_id
    raw_mask_token_id, mask_token_id, used_unk_fallback = get_safe_mask_token_id(tokenizer)
    vocab_size = len(tokenizer)

    print(
        f"[Info] tokenizer_size={vocab_size} "
        f"pad_token_id={pad_token_id} "
        f"mask_token={tokenizer.mask_token} "
        f"mask_token_id(raw)={raw_mask_token_id} "
        f"mask_token_id(used)={mask_token_id} "
        f"used_unk_fallback={used_unk_fallback}"
    )

    train_loader, valid_loader = LMMIterator(opt, pad_token_id).loader

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        betas=(0.9, 0.98), eps=1e-9, lr=opt.max_lr
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

            if opt.stage == "bpt":
                mask = src.ne(tokenizer.pad_token_id).long()
                src = token_mask(
                    src, mask.sum(-1), 0.35,
                    mask_token_id
                )
                mask = src.ne(tokenizer.pad_token_id).long()
            else:
                mask = tgt.ne(tokenizer.pad_token_id).long()
                tgt_mask = token_mask(
                    tgt, mask.sum(-1), 0.35,
                    mask_token_id
                )
                src = batch_process(src, tgt_mask, tokenizer.pad_token_id)
                mask = src.ne(tokenizer.pad_token_id).long()

            assert_ids_in_range(src, vocab_size, "train/src")
            assert_ids_in_range(tgt, vocab_size, "train/tgt")

            decoder_input = shift_tokens_right(
                tgt, tokenizer.pad_token_id,
                model.config.decoder_start_token_id
            )

            with autocast("cuda", dtype=torch.float16):
                outputs = model(
                    src, mask,
                    decoder_input_ids=decoder_input
                )
                loss = loss_fn(
                    outputs.logits.view(-1, vocab_size),
                    tgt.view(-1)
                )

            loss_list.append(loss.item())

            loss = loss / opt.acc_steps
            scaler.scale(loss).backward()

            if step % opt.acc_steps == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1, norm_type=2)
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
                    model, valid_loader,
                    tokenizer, loss_fn, step, opt.stage,
                    mask_token_id=mask_token_id,
                    max_valid_batches=1000
                )

                if best_loss >= eval_loss:
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_path = f"checkpoints/mlm_{opt.stage}_mlm_zh.chkpt"
                    torch.save(model.state_dict(), ckpt_path)
                    print(f"[Info] The checkpoint has been updated: {ckpt_path}")
                    best_loss = eval_loss
                    tab = 0
                else:
                    tab += 1

                if tab == opt.patience:
                    print("[Info] Early stopping triggered.")
                    return


if __name__ == "__main__":
    main()