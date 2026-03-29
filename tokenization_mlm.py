# -*- coding: utf-8 -*-

import os
from contextlib import contextmanager
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import sentencepiece as spm
from transformers.tokenization_utils import AddedToken, BatchEncoding, PreTrainedTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"
VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "mbart-large-50": "sentencepiece.bpe.model",
        "./mbart-large-50": "sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "mbart-large-50": 1024,
    "./mbart-large-50": 1024,
}

# IMPORTANT:
# Keep zh_CN from mBART-50, and add <drs> as an extra language code.
FAIRSEQ_LANGUAGE_CODES = [
    "ar_AR", "cs_CZ", "de_DE", "en_XX", "es_XX", "et_EE", "fi_FI", "fr_XX", "gu_IN", "hi_IN",
    "it_IT", "ja_XX", "kk_KZ", "ko_KR", "lt_LT", "lv_LV", "my_MM", "ne_NP", "nl_XX", "ro_RO",
    "ru_RU", "si_LK", "tr_TR", "vi_VN", "zh_CN", "af_ZA", "az_AZ", "bn_IN", "fa_IR", "he_IL",
    "hr_HR", "id_ID", "ka_GE", "km_KH", "mk_MK", "ml_IN", "mn_MN", "mr_IN", "pl_PL", "ps_AF",
    "pt_XX", "sv_SE", "sw_KE", "ta_IN", "te_IN", "th_TH", "tl_XX", "uk_UA", "ur_PK", "xh_ZA",
    "gl_ES", "sl_SI", "<drs>"
]


class MLMTokenizer(PreTrainedTokenizer):
    """
    mBART-50 tokenizer wrapper that keeps all mBART language codes, adds <drs>,
    and remains compatible with old DRS-LMM code.

    The critical fix versus the original file:
      - load SentencePiece BEFORE super().__init__

    Otherwise transformers may call get_vocab() during parent init, and self.sp_model
    does not exist yet, causing:
        AttributeError: 'MLMTokenizer' object has no attribute 'sp_model'
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        src_lang=None,
        tgt_lang=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        # --- CRITICAL FIX: initialize sentencepiece BEFORE super().__init__ ---
        self.vocab_file = str(vocab_file)
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
        self.sp_model_size = len(self.sp_model)

        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", [])
        kwargs["additional_special_tokens"] += [
            code for code in FAIRSEQ_LANGUAGE_CODES if code not in kwargs["additional_special_tokens"]
        ]

        # Mimic fairseq alignment
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = 1

        self.lang_code_to_id = {
            code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}
        self.fairseq_tokens_to_ids["<mask>"] = self.sp_model_size + len(self.lang_code_to_id) + self.fairseq_offset
        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        if src_lang is None:
            src_lang = "en_XX"
        if src_lang not in self.lang_code_to_id:
            raise ValueError(f"Unsupported src_lang: {src_lang}. Available includes zh_CN and <drs>.")

        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self._src_lang = src_lang
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

    @property
    def vocab_size(self) -> int:
        return self.sp_model_size + len(self.lang_code_to_id) + self.fairseq_offset + 1

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        if new_src_lang not in self.lang_code_to_id:
            raise ValueError(f"Unsupported src_lang: {new_src_lang}")
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)
        self.sp_model_size = len(self.sp_model)

    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return self.sp_model.decode(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return ()
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                fi.write(self.sp_model.serialized_model_proto())

        return (out_vocab_file,)

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
        self, raw_inputs, return_tensors: str, src_lang: Optional[str], tgt_lang: Optional[str], **extra_kwargs
    ):
        if src_lang is None or tgt_lang is None:
            raise ValueError("Translation requires a `src_lang` and a `tgt_lang` for this model")
        self.src_lang = src_lang
        inputs = self(raw_inputs, add_special_tokens=True, return_tensors=return_tensors, **extra_kwargs)
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = [self.eos_token_id]
