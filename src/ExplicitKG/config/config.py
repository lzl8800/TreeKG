# -*- coding: utf-8 -*-
"""
config/config.py
全局/通用配置（整个显式KG共享）：工程路径、输出路径、API 等。
不要把具体到某一条流程（例如文本分割）的业务参数放这里。
"""
import os
from pathlib import Path

# ====== 工程与路径 ======
# 本文件位于 ExplicitKG/config/，向上一层是 ExplicitKG/
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
SRC_DIR: Path = PROJECT_ROOT
EXPLICIT_KG_DIR: Path = PROJECT_ROOT                  # 与既有结构保持一致
OUTPUT_DIR: Path = EXPLICIT_KG_DIR / "output"         # 统一输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ====== API（全局可复用）======
class APIConfig:
    """
    仅保留必要字段，全部从环境变量读取，避免在仓库里留密钥。
    示例：
      export OPENAI_API_KEY="sk-xxxx"
      export LLM_API_BASE="https://api.suanli.cn/v1"
      export LLM_MODEL_NAME="deepseek-r1:7b"
    """


    API_BASE: str = os.getenv("LLM_API_BASE", "http://192.168.71.130:9996/v1")
    API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gemma-3-it")
    TIMEOUT_SECS: int = int(os.getenv("LLM_TIMEOUT_SECS", "120"))