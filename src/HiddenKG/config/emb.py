# -*- coding: utf-8 -*-
"""
HiddenKG/config/emb.py
Embedding（实体向量化）阶段配置：
- I/O 路径
- 批大小 / 最大长度 / AMP
- 模型资源路径（沿用全局 BERT 资源）
"""
import os
from pathlib import Path

from .config import (
    HIDDEN_KG_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    BERT_BASE_DIR,
    BERT_CONFIG_FILE,
    BERT_VOCAB_FILE,
    BERT_MODEL_FILE,
    DEVICE,  # 复用全局设备选择
)

class EmbConfig:
    # ===== 输入/输出 =====
    # 默认读取 Conv 阶段产出的实体文件：HiddenKG/output/conv_entities.json
    INPUT_FILE: Path = Path(
        os.getenv("EMB_INPUT", str(OUTPUT_DIR / "conv_entities.json"))
    )
    # 默认输出：HiddenKG/output/node_embeddings.pkl
    OUTPUT_FILE: Path = Path(
        os.getenv("EMB_OUTPUT", str(OUTPUT_DIR / "node_embeddings.pkl"))
    )

    # ===== 批处理与序列长度 =====
    BATCH_SIZE: int = int(os.getenv("EMB_BATCH_SIZE", "16"))
    MAX_LENGTH: int = int(os.getenv("EMB_MAX_LENGTH", "512"))

    # ===== AMP（半精度）开关：1=启用, 0=禁用 =====
    USE_AMP: bool = bool(int(os.getenv("EMB_USE_AMP", "1")))

    # ===== 设备（沿用全局 DEVICE，但也允许通过环境覆盖）=====
    DEVICE: str = os.getenv("EMB_DEVICE", DEVICE)

    # ===== BERT 资源（沿用全局，可通过环境覆盖）=====
    BERT_BASE_DIR: Path   = Path(os.getenv("EMB_BERT_DIR",   str(BERT_BASE_DIR)))
    BERT_CONFIG_FILE: Path = Path(os.getenv("EMB_BERT_CFG",  str(BERT_CONFIG_FILE)))
    BERT_VOCAB_FILE: Path  = Path(os.getenv("EMB_BERT_VOC",  str(BERT_VOCAB_FILE)))
    BERT_MODEL_FILE: Path  = Path(os.getenv("EMB_BERT_BIN",  str(BERT_MODEL_FILE)))

    # ===== 其他路径（日志等，可复用全局）=====
    OUTPUT_DIR: Path = OUTPUT_DIR
    LOG_DIR: Path = LOG_DIR

    @staticmethod
    def ensure_paths() -> None:
        EmbConfig.OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        EmbConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)

