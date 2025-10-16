# -*- coding: utf-8 -*-
"""
HiddenKG.config
统一对外导出：
- 全局环境/路径与 APIConfig（来自 config.py）
- Pred：边预测配置（PredConfig，类命名空间）
"""

# —— 全局路径 & APIConfig —— #
from .config import (
    PROJECT_ROOT,
    SRC_DIR,
    HIDDEN_KG_DIR,
    OUTPUT_DIR,
    LOG_DIR,
    BERT_BASE_DIR,
    BERT_CONFIG_FILE,
    BERT_VOCAB_FILE,
    BERT_MODEL_FILE,
    RANDOM_SEED,
    DEVICE,
    APIConfig,
)

# —— 分阶段配置（类命名空间）—— #
from .pred import PredConfig as Pred  # 边预测配置
from .conv import ConvConfig as Conv
from .aggr import AggrConfig as Aggr
from .emb import EmbConfig as Emb
from .dedup import DedupConfig as Dedup


__all__ = [
    # 全局
    "PROJECT_ROOT", "SRC_DIR", "HIDDEN_KG_DIR", "OUTPUT_DIR", "LOG_DIR",
    "BERT_BASE_DIR", "BERT_CONFIG_FILE", "BERT_VOCAB_FILE", "BERT_MODEL_FILE",
    "RANDOM_SEED", "DEVICE", "APIConfig",
    # 分阶段配置（推荐业务使用）
    "Pred",
    "Conv",
    "Aggr",
    "Emb",
    "Dedup",
]
