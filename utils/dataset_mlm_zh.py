# -*- coding: utf-8 -*-

import os
import random
import torch
import torch.utils.data
from tokenization_mlm import MLMTokenizer

random.seed(42)


def token_mask(seq, seq_len, replace_prob=0.35, mask_idx=0):
    if replace_prob == 0:
        return seq

    noise = torch.rand(seq.size(), dtype=torch.float).to(seq.device)
    pos_idx = torch.arange(seq.size(1)).expand_as(seq).to(seq.device)
    token_mask = (0 < pos_idx) & (pos_idx < seq_len.unsqueeze(1) - 1)
    drop_mask = (noise < replace_prob) & token_mask

    x = seq.clone()
    x.masked_fill_(drop_mask, mask_idx)

    return x


class LMMDataset(torch.utils.data.Dataset):
    """Seq2Seq Dataset"""

    def __init__(self, src_inst, tgt_inst):
        self.src_inst = src_inst
        self.tgt_inst = tgt_inst

    def __len__(self):
        return len(self.src_inst)

    def __getitem__(self, idx):
        return self.src_inst[idx], self.tgt_inst[idx]


class LMMIterator(object):
    """
    统一支持 bpt / spt / fft / sft
    数据按你现在真实目录读取：
      data/en_de
      data/en_it
      data/en_nl
      data/en_zh

    约定：
      - DRS 侧: src_lang = "<drs>"
      - 自然语言侧: src_lang = en_XX / de_DE / it_IT / nl_XX / zh_CN
      - model_path 固定为 mbart-large-50-mlm-zh
    """

    PAIR_DIR_TO_LANG = {
        "en_de": "de_DE",
        "en_it": "it_IT",
        "en_nl": "nl_XX",
        "en_zh": "zh_CN",
    }

    def __init__(self, opt, pad_id):
        self.opt = opt
        self.pad_id = pad_id

        self.train_src, self.train_tgt = self.read_insts("train", opt)
        self.valid_src, self.valid_tgt = self.read_insts("valid", opt)

        print("[Info] {} insts from train set".format(len(self.train_src)))
        print("[Info] {} insts from valid set".format(len(self.valid_src)))

        self.loader = self.gen_loader(
            self.train_src, self.train_tgt,
            self.valid_src, self.valid_tgt
        )

    def _select_pair_dirs(self):
        """
        用户传 -lang de_DE it_IT nl_XX zh_CN
        就映射成：
          en_de / en_it / en_nl / en_zh
        不需要单独传 en_XX，因为每个 pair 里英语会自动被读进来。
        """
        want = set(getattr(self.opt, "lang", []) or [])
        candidates = []

        for d, lg in self.PAIR_DIR_TO_LANG.items():
            if lg in want:
                candidates.append(d)

        if len(candidates) == 0:
            for d in self.PAIR_DIR_TO_LANG.keys():
                if os.path.isdir(os.path.join("data", d)):
                    candidates.append(d)

        return candidates

    def _encode_trim(self, tokenizer, text: str):
        ids = tokenizer.encode(
            text.strip(),
            truncation=True,
            max_length=min(self.opt.max_len, 1024)
        )
        return ids

    def _upsample_target(self, n_items: int, stage: str, is_lang_side: bool):
        """
        对齐你之前的经验，同时兼顾 README 的 pretrain 思路。
        参数后续都能改，这里先给一个稳一点的版本。

        - sft: 不上采样
        - fft: 中等上采样
        - spt/bpt: 适当大一些
        """
        if stage == "sft":
            return n_items
        if stage == "fft":
            return 30000
        if stage == "spt":
            return 50000
        if stage == "bpt":
            return 50000
        return 30000

    def read_insts(self, mode, opt):
        src, tgt = [], []
        model_path = "mbart-large-50-mlm-zh"

        pair_dirs = self._select_pair_dirs()
        if len(pair_dirs) == 0:
            raise FileNotFoundError("No pair dirs found under data/.")

        for pair in pair_dirs:
            lang = self.PAIR_DIR_TO_LANG[pair]
            base = os.path.join("data", pair)

            # 英语侧
            en_drs = os.path.join(base, f"{mode}_en_XX.0")
            en_txt = os.path.join(base, f"{mode}_en_XX.1")

            # 目标语言侧
            lg_drs = os.path.join(base, f"{mode}_{lang}.0")
            lg_txt = os.path.join(base, f"{mode}_{lang}.1")

            for p in [en_drs, en_txt, lg_drs, lg_txt]:
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Missing file: {p}")

            tokenizer_drs = MLMTokenizer.from_pretrained(model_path, src_lang="<drs>")
            tokenizer_en = MLMTokenizer.from_pretrained(model_path, src_lang="en_XX")
            tokenizer_lg = MLMTokenizer.from_pretrained(model_path, src_lang=lang)

            # ------------------------------------------------------------------
            # 1) 读取 EN <-> DRS
            # ------------------------------------------------------------------
            src_seq_en, tgt_seq_en = [], []
            with open(en_drs, "r", encoding="utf-8") as f1, open(en_txt, "r", encoding="utf-8") as f2:
                f1_lines = f1.readlines()
                f2_lines = f2.readlines()

                for i in range(min(len(f1_lines), len(f2_lines))):
                    s = self._encode_trim(tokenizer_drs, f1_lines[i])
                    t = self._encode_trim(tokenizer_en, f2_lines[i])
                    src_seq_en.append(s)
                    tgt_seq_en.append(t)

            # ------------------------------------------------------------------
            # 2) 读取 LANG <-> DRS
            # ------------------------------------------------------------------
            src_seq_lg, tgt_seq_lg = [], []
            with open(lg_drs, "r", encoding="utf-8") as f1, open(lg_txt, "r", encoding="utf-8") as f2:
                f1_lines = f1.readlines()
                f2_lines = f2.readlines()

                for i in range(min(len(f1_lines), len(f2_lines))):
                    s = self._encode_trim(tokenizer_drs, f1_lines[i])
                    t = self._encode_trim(tokenizer_lg, f2_lines[i])
                    src_seq_lg.append(s)
                    tgt_seq_lg.append(t)

            # ------------------------------------------------------------------
            # 3) 上采样（只对 train）
            # ------------------------------------------------------------------
            if mode != "valid":
                ups_en = self._upsample_target(len(src_seq_en), opt.stage, is_lang_side=False)
                if len(src_seq_en) < ups_en:
                    times = int(ups_en / len(src_seq_en)) + 1
                    src_seq_en = (src_seq_en * times)[:ups_en]
                    tgt_seq_en = (tgt_seq_en * times)[:ups_en]

                ups_lg = self._upsample_target(len(src_seq_lg), opt.stage, is_lang_side=True)
                if len(src_seq_lg) < ups_lg:
                    times = int(ups_lg / len(src_seq_lg)) + 1
                    src_seq_lg = (src_seq_lg * times)[:ups_lg]
                    tgt_seq_lg = (tgt_seq_lg * times)[:ups_lg]

            # ------------------------------------------------------------------
            # 4) 基础双向数据：DRS <-> EN, DRS <-> LANG
            #    这部分同时服务 fft / sft，也作为 bpt/spt 的底子
            # ------------------------------------------------------------------
            # # DRS -> EN
            # src.extend(src_seq_en)
            # tgt.extend(tgt_seq_en)
            # # EN -> DRS
            # src.extend(tgt_seq_en)
            # tgt.extend(src_seq_en[:len(tgt_seq_en)])
            #
            # # DRS -> LANG
            # src.extend(src_seq_lg)
            # tgt.extend(tgt_seq_lg)
            # # LANG -> DRS
            # src.extend(tgt_seq_lg)
            # tgt.extend(src_seq_lg[:len(tgt_seq_lg)])
            if lang == "zh_CN":
                src.extend(tgt_seq_lg)
                tgt.extend(src_seq_lg[:len(tgt_seq_lg)])

            # ------------------------------------------------------------------
            # 5) S-PT: cross-lingual supervised pre-training
            #    参考 README / 原始 dataset 思路：英语和目标语言之间通过 DRS 形成对齐训练
            # ------------------------------------------------------------------
            if opt.stage == "spt":
                # 让 EN 和 LANG 两边都参与更强的跨语言监督
                src.extend(src_seq_en + src_seq_en + tgt_seq_en + tgt_seq_en +
                           src_seq_lg + src_seq_lg + tgt_seq_lg + tgt_seq_lg)
                tgt.extend(src_seq_lg + tgt_seq_lg + src_seq_lg + tgt_seq_lg +
                           src_seq_en + tgt_seq_en + src_seq_en + tgt_seq_en)

        # ----------------------------------------------------------------------
        # 6) B-PT: basic pre-training
        #    参考原 train_pt.py：bpt 阶段最终返回 src, src.copy()
        #    也就是把所有收集到的序列当成自编码对象
        # ----------------------------------------------------------------------
        if opt.stage == "bpt":
            return src, src.copy()

        return src, tgt

    def gen_loader(self, train_src, train_tgt, valid_src, valid_tgt):
        train_loader = torch.utils.data.DataLoader(
            LMMDataset(src_inst=train_src, tgt_inst=train_tgt),
            num_workers=4,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn,
            shuffle=True
        )

        valid_loader = torch.utils.data.DataLoader(
            LMMDataset(src_inst=valid_src, tgt_inst=valid_tgt),
            num_workers=4,
            batch_size=self.opt.batch_size,
            collate_fn=self.paired_collate_fn
        )

        return train_loader, valid_loader

    def collate_fn(self, insts):
        max_len = max(len(inst) for inst in insts)
        batch_seq = [inst + [self.pad_id] * (max_len - len(inst)) for inst in insts]
        return torch.LongTensor(batch_seq)

    def paired_collate_fn(self, insts):
        src_inst, tgt_inst = list(zip(*insts))
        src_inst = self.collate_fn(src_inst)
        tgt_inst = self.collate_fn(tgt_inst)
        return src_inst, tgt_inst