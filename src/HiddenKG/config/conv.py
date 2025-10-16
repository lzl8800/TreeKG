# -*- coding: utf-8 -*-
"""
HiddenKG/config/conv.py
Conv（Contextual-based Convolution）阶段配置：
- I/O 路径
- LLM 调用参数与重试/超时
- 速率限制与节流
- 其他开关（如 DRY_RUN、邻居上限等）
"""
import os
from pathlib import Path
from .config import HIDDEN_KG_DIR, OUTPUT_DIR, APIConfig  # 复用全局输出目录与 APIConfig 字段可用性

class ConvConfig:
    # ===== 路径 & 文件 =====
    OUTPUT_DIR: Path           = OUTPUT_DIR
    EXPLICIT_OUT_DIR: Path = (HIDDEN_KG_DIR.parent / "ExplicitKG" / "output").resolve()
    FILE_TOC_ENT_REL: Path = Path(
        os.getenv("CONV_TOC_ENT_REL",
                  str(EXPLICIT_OUT_DIR / "toc_with_entities_and_relations.json"))
    )
    FILE_EXPLICIT_KG: Path = Path(
        os.getenv("CONV_EXPLICIT_KG",
                  str(EXPLICIT_OUT_DIR / "explicit_kg.json"))
    )

    # 输出（仍写到 HiddenKG/output）
    FILE_CONV_RESULT: Path = OUTPUT_DIR / "conv_entities.json"
    FILE_CONV_PROMPTS: Path = OUTPUT_DIR / "conv_prompts.json"

    @staticmethod
    def ensure_paths() -> None:
        ConvConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ConvConfig.FILE_CONV_PROMPTS.parent.mkdir(parents=True, exist_ok=True)
        ConvConfig.FILE_CONV_RESULT.parent.mkdir(parents=True, exist_ok=True)

    # ===== 邻域控制 =====
    MAX_NEIGHBORS_GLOBAL: int  = int(os.getenv("CONV_MAX_NEIGHBORS_GLOBAL", "30"))

    # ===== LLM / 超时 / 重试（优先环境变量；否则兼容 APIConfig 的字段名） =====
    TEMPERATURE: float         = float(os.getenv("CONV_TEMPERATURE", "0.2"))
    MAX_TOKENS: int            = int(os.getenv("CONV_MAX_TOKENS", "300"))
    API_TIMEOUT: int           = int(os.getenv(
                                "CONV_TIMEOUT_SECS",
                                str(getattr(APIConfig, "TIMEOUT_SECS",
                                    getattr(APIConfig, "TIMEOUT", 120)))
                              ))
    RETRIES: int               = int(os.getenv("CONV_RETRIES", "3"))
    CHAT_COMPLETIONS_PATH: str = os.getenv("CONV_CHAT_COMPLETIONS_PATH", "/chat/completions")

    # ===== 速率限制（可选） =====
    RATE_LIMIT_QPS: float      = float(os.getenv("CONV_RATE_LIMIT_QPS", "0"))
    EXTRA_THROTTLE_SEC: float  = float(os.getenv("CONV_EXTRA_THROTTLE_SEC", "0.0"))
    RETRY_BACKOFF_BASE: float  = float(os.getenv("CONV_BACKOFF_BASE", "1.8"))

    # ===== 其他开关 =====
    DRY_RUN: bool              = bool(int(os.getenv("CONV_DRY_RUN", "0")))

__all__ = [
    "ConvConfig",
]
