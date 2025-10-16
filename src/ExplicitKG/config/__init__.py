# -*- coding: utf-8 -*-
"""
统一导出配置对象/常量，便于 from config import ... 的写法。
"""
from .config import (
    PROJECT_ROOT,
    SRC_DIR,
    EXPLICIT_KG_DIR,
    OUTPUT_DIR,
    APIConfig,
)
from .text import TextSegConfig
from .summarize import SummarizeConfig
from .extract import ExtractionConfig

__all__ = [
    # 全局/通用
    "PROJECT_ROOT",
    "SRC_DIR",
    "EXPLICIT_KG_DIR",
    "OUTPUT_DIR",
    "APIConfig",
    # TextSegmentation 专用
    "TextSegConfig",
    # Summarization 专用
    "SummarizeConfig",
    # Extraction 专用
    "ExtractionConfig"
]
