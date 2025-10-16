# -*- coding: utf-8 -*-
"""
HiddenKG/config/aggr.py
Aggr（实体聚合）阶段配置：
- I/O 路径（conv 输出作为输入；aggr 结果输出）
- LLM 调用参数（温度、max_tokens）与重试/超时
- 日志与目录保障
"""
import os
from pathlib import Path
from .config import OUTPUT_DIR as _OUTPUT_DIR, APIConfig, HIDDEN_KG_DIR

class AggrConfig:
    # ===== 目录 & 路径 =====
    OUTPUT_DIR: Path = _OUTPUT_DIR
    LOG_DIR: Path    = HIDDEN_KG_DIR / "logs"

    # 输入（conv 结果）：默认 HiddenKG/output/conv_entities.json
    FILE_CONV_RESULT: Path = Path(
        os.getenv("AGGR_CONV_INPUT", str(OUTPUT_DIR / "conv_entities.json"))
    )
    # 输出（aggr 结果）：默认 HiddenKG/output/aggr_entities.json
    FILE_AGGR_RESULT: Path = Path(
        os.getenv("AGGR_OUT", str(OUTPUT_DIR / "aggr_entities.json"))
    )

    @staticmethod
    def ensure_paths() -> None:
        AggrConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        AggrConfig.LOG_DIR.mkdir(parents=True, exist_ok=True)
        AggrConfig.FILE_AGGR_RESULT.parent.mkdir(parents=True, exist_ok=True)

    # ===== LLM 调用参数 =====
    TEMPERATURE: float = float(os.getenv("AGGR_TEMPERATURE", "0.0"))
    MAX_TOKENS: int    = int(os.getenv("AGGR_MAX_TOKENS", "24"))

    # ===== 超时 / 重试（优先环境变量 → 其次 APIConfig 字段）=====
    API_TIMEOUT: int = int(os.getenv(
        "AGGR_TIMEOUT_SECS",
        str(getattr(APIConfig, "TIMEOUT_SECS", getattr(APIConfig, "TIMEOUT", 120)))
    ))
    RETRIES: int     = int(os.getenv(
        "AGGR_RETRIES",
        str(getattr(APIConfig, "RETRIES", 3))
    ))

    # 兼容路径：聊天接口路径（可覆盖）
    CHAT_COMPLETIONS_PATH: str = os.getenv("AGGR_CHAT_COMPLETIONS_PATH", "/chat/completions")

    # 可选：干运行（跳过 LLM，便于调试）
    DRY_RUN: bool = bool(int(os.getenv("AGGR_DRY_RUN", "0")))

