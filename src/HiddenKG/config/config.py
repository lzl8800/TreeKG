# -*- coding: utf-8 -*-
"""
HiddenKG/config/config.py
全局/通用配置（隐式KG共享）：工程根目录、输出与日志目录、本地模型资源等。
不要把具体业务流程（训练、推理的超参等）放在这里。
"""
import os
import sys
from pathlib import Path

# ====== API（复用 ExplicitKG 的 APIConfig）======
from ExplicitKG.config.config import APIConfig  # type: ignore

# ====== 工程与路径 ======
# 本文件位于 HiddenKG/config/，向上一层是 HiddenKG/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT
HIDDEN_KG_DIR: Path = PROJECT_ROOT                   # 与既有结构保持一致

# 统一输出目录（与项目结构中的 HiddenKG/output 对应）
OUTPUT_DIR: Path = HIDDEN_KG_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 可选：日志目录（若你有 logging 需求）
LOG_DIR: Path = HIDDEN_KG_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ====== 本地模型/资源路径（按你的目录命名提供便捷常量）======
# 目录结构中有 HiddenKG/bert-base-chinese/
BERT_BASE_DIR: Path = HIDDEN_KG_DIR / "model/bert-base-chinese"
# 常见的权重/词表/配置文件命名（按需使用，存在即用，不存在也不会报错）
BERT_CONFIG_FILE: Path = BERT_BASE_DIR / "config.json"
BERT_VOCAB_FILE: Path = BERT_BASE_DIR / "vocab.txt"
BERT_MODEL_FILE: Path = BERT_BASE_DIR / "pytorch_model.bin"

# ====== 运行期通用开关 ======
RANDOM_SEED: int = int(os.getenv("GLOBAL_RANDOM_SEED", "42"))
# 简单的设备选择：若设置了 CUDA_VISIBLE_DEVICES 则默认用 cuda，否则 cpu
DEVICE: str = os.getenv(
    "DEVICE",
    "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") not in ("", "-1", None) else "cpu",
)

