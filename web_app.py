# -*- coding: utf-8 -*-
"""
Natural Language -> DRS/SBN -> Semantic Graph Web Demo

Run:
    python web_app.py

浏览器:
    http://127.0.0.1:7860
"""

import os
import re
import sys
import uuid
import shutil
import traceback
from pathlib import Path
from typing import Dict, Tuple

import torch
from flask import Flask, request, render_template_string
from transformers import MBartForConditionalGeneration

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR

MODEL_DIR = Path("/home/yooho/duomotai/code/DRS-pretrained-LMM-main/mbart-large-50-mlm-zh")
CKPT_PATH = Path("/home/yooho/duomotai/code/DRS-pretrained-LMM-main/checkpoints/ft_sft_mlm_zh.chkpt")

SBN_DIR = PROJECT_DIR / "evaluation" / "parsing_smatch" / "sbn"

STATIC_DIR = PROJECT_DIR / "static"
GRAPH_DIR = STATIC_DIR / "graphs"

# ===== 环境变量 =====
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ===== 导入 tokenizer =====
sys.path.append(str(PROJECT_DIR))
from tokenization_mlm import MLMTokenizer  # noqa: E402

# ===== 导入 SBNGraph =====
# 兼容你当前 sbn 工具链的不同导入方式
sys.path.append(str(SBN_DIR))
SBNGraph = None
_sbn_import_error = None

try:
    # 你之前实际用到过这条链路
    from sbn2png import SBNGraph as _SBNGraph  # type: ignore  # noqa: E402
    SBNGraph = _SBNGraph
except Exception as e1:
    try:
        from sbn2penman import SBNGraph as _SBNGraph  # type: ignore  # noqa: E402
        SBNGraph = _SBNGraph
    except Exception as e2:
        _sbn_import_error = f"sbn2png import failed: {e1}; sbn2penman import failed: {e2}"

# ===== Flask =====
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")

# ===== 设备 =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 全局对象 =====
MODEL = None
TOKENIZER_ZH = None
TOKENIZER_EN = None
TOKENIZER_DRS = None
DRS_BOS_ID = None

# ===== 页面模板：单文件，不依赖 templates 目录 =====
HTML = r"""
<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Natural Language → DRS → Semantic Graph</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      margin: 0;
      padding: 24px;
      font-family: Arial, Helvetica, sans-serif;
      background: #f7f7f8;
      color: #111;
    }
    .container {
      max-width: 1100px;
      margin: 0 auto;
    }
    h1 {
      margin: 0 0 16px 0;
      font-size: 28px;
    }
    .card {
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    textarea {
      width: 100%;
      min-height: 110px;
      resize: vertical;
      border: 1px solid #ccc;
      border-radius: 8px;
      padding: 12px;
      font-size: 15px;
      box-sizing: border-box;
    }
    .row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 12px;
    }
    button {
      background: #111;
      color: #fff;
      border: none;
      border-radius: 8px;
      padding: 10px 18px;
      cursor: pointer;
      font-size: 14px;
    }
    button:hover {
      opacity: 0.92;
    }
    .meta {
      font-size: 13px;
      color: #555;
      margin-top: 8px;
    }
    .label {
      font-weight: bold;
      margin-bottom: 8px;
      display: block;
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      background: #f3f4f6;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      font-size: 14px;
      line-height: 1.5;
    }
    img {
      max-width: 100%;
      display: block;
      border: 1px solid #ddd;
      border-radius: 8px;
      background: #fff;
      padding: 8px;
    }
    .error {
      color: #a40000;
      white-space: pre-wrap;
      word-break: break-word;
      background: #fff5f5;
      border: 1px solid #f1bcbc;
      border-radius: 8px;
      padding: 12px;
    }
    .ok {
      color: #0b5;
    }
    .small {
      font-size: 12px;
      color: #666;
    }
    .two-col {
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }
    @media (min-width: 900px) {
      .two-col {
        grid-template-columns: 1fr 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Natural Language → DRS → Semantic Graph</h1>

    <div class="card">
      <form method="post">
        <label class="label" for="text">输入自然语言（中文或英文）</label>
        <textarea id="text" name="text" placeholder="例如：他去了医院。&#10;He went to the hospital.">{{ result.text }}</textarea>
        <div class="row">
          <button type="submit">生成</button>
        </div>
      </form>
      <div class="meta">
        当前设备：{{ result.device }} |
        模型：{{ result.model_name }} |
        Checkpoint：{{ result.ckpt_name }}
      </div>
    </div>

    {% if result.error %}
    <div class="card">
      <div class="label">错误信息</div>
      <div class="error">{{ result.error }}</div>
    </div>
    {% endif %}

    {% if result.text %}
    <div class="two-col">
      <div class="card">
        <div class="label">Natural Language</div>
        <pre>{{ result.text }}</pre>
        <div class="meta">自动识别语言：<span class="ok">{{ result.lang }}</span></div>
      </div>

      <div class="card">
        <div class="label">DRS / SBN</div>
        <pre>{{ result.drs }}</pre>
      </div>
    </div>

    <div class="card">
      <div class="label">Semantic Graph</div>
      {% if result.graph_url %}
        <img src="{{ result.graph_url }}" alt="semantic graph">
        <div class="small">图像路径：{{ result.graph_url }}</div>
      {% else %}
        <div class="error">图生成失败。</div>
      {% endif %}
    </div>
    {% endif %}
  </div>
</body>
</html>
"""


def ensure_dirs() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)


def detect_lang(text: str) -> str:
    """
    简单稳定的中英识别：
    - 只要含中文字符，就判为 zh_CN
    - 否则判为 en_XX
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh_CN"
    return "en_XX"


def clean_model_output(text: str) -> str:
    text = text.replace("</s>", "").replace("<s>", "").strip()
    if text.startswith("<drs>"):
        text = text[len("<drs>"):].strip()
    return text


def load_all() -> None:
    global MODEL, TOKENIZER_ZH, TOKENIZER_EN, TOKENIZER_DRS, DRS_BOS_ID

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"CKPT not found: {CKPT_PATH}")
    if SBNGraph is None:
        raise ImportError(f"Failed to import SBNGraph. Details: {_sbn_import_error}")

    TOKENIZER_ZH = MLMTokenizer.from_pretrained(
        str(MODEL_DIR),
        src_lang="zh_CN",
        local_files_only=True
    )
    TOKENIZER_EN = MLMTokenizer.from_pretrained(
        str(MODEL_DIR),
        src_lang="en_XX",
        local_files_only=True
    )
    TOKENIZER_DRS = MLMTokenizer.from_pretrained(
        str(MODEL_DIR),
        src_lang="<drs>",
        local_files_only=True
    )

    MODEL = MBartForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        local_files_only=True
    )
    state_dict = torch.load(str(CKPT_PATH), map_location="cpu")
    MODEL.load_state_dict(state_dict, strict=True)
    MODEL.to(DEVICE)
    MODEL.eval()

    # 用真实编码结果取合法 bos id，避免旧高位 special id 越界
    DRS_BOS_ID = TOKENIZER_DRS("test", return_tensors="pt")["input_ids"][0][0].item()


def generate_drs(text: str) -> Tuple[str, str]:
    """
    返回: (lang, drs_text)
    """
    if MODEL is None:
        raise RuntimeError("Model is not loaded.")

    lang = detect_lang(text)
    tokenizer_in = TOKENIZER_ZH if lang == "zh_CN" else TOKENIZER_EN

    inputs = tokenizer_in(
        text.strip(),
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            forced_bos_token_id=DRS_BOS_ID,
            max_length=160,
            num_beams=5,
            repetition_penalty=1.1,
            early_stopping=True
        )

    raw = TOKENIZER_DRS.decode(output_ids[0], skip_special_tokens=False)
    drs = clean_model_output(raw)
    return lang, drs


def render_semantic_graph(sbn_text: str) -> str:
    """
    将单行 SBN/DRS 渲染成 png，返回网页静态路径。
    """
    ensure_dirs()

    graph_id = uuid.uuid4().hex
    out_base = GRAPH_DIR / graph_id

    graph = SBNGraph().from_string(sbn_text, is_single_line=True)
    # 你的工具链一般是 graph.to("png", file_name_without_ext)
    graph.to("png", str(out_base))

    # 兼容不同实现的输出命名
    candidates = [
        out_base.with_suffix(".png"),
        Path(str(out_base) + ".png"),
        out_base,  # 极少数实现可能直接输出不带后缀名
    ]

    real_file = None
    for p in candidates:
        if p.exists() and p.is_file():
            real_file = p
            break

    if real_file is None:
        raise FileNotFoundError(
            f"Graph png was not created. Tried: {[str(x) for x in candidates]}"
        )

    # 如果工具链输出不是标准 .png 名，复制成标准名
    final_png = GRAPH_DIR / f"{graph_id}.png"
    if real_file.resolve() != final_png.resolve():
        shutil.copyfile(real_file, final_png)

    return f"/static/graphs/{graph_id}.png"


@app.route("/", methods=["GET", "POST"])
def index():
    result: Dict[str, str] = {
        "text": "",
        "lang": "",
        "drs": "",
        "graph_url": "",
        "error": "",
        "device": DEVICE,
        "model_name": MODEL_DIR.name,
        "ckpt_name": CKPT_PATH.name,
    }

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        result["text"] = text

        if text:
            try:
                lang, drs = generate_drs(text)
                result["lang"] = lang
                result["drs"] = drs
                result["graph_url"] = render_semantic_graph(drs)
            except Exception as e:
                result["error"] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

    return render_template_string(HTML, result=result)


def main():
    ensure_dirs()
    load_all()
    app.run(host="0.0.0.0", port=7860, debug=True)


if __name__ == "__main__":
    main()