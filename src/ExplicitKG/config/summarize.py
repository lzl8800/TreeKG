# -*- coding: utf-8 -*-
"""
config/summarize.py
自底向上摘要流程专用配置：输入/输出、并发控制、分段策略、提示词等。
"""
import os
from pathlib import Path
from .config import OUTPUT_DIR
from .text import TextSegConfig  # 复用上一步的 .docx 和 toc 路径

class SummarizeConfig:
    # —— I/O —— #
    DOCX_PATH: Path = TextSegConfig.DOCX_PATH               # 书籍 .docx
    TOC_JSON_PATH: Path = TextSegConfig.TOC_PATH            # 上一步生成的 toc_structure.json
    OUT_JSON_FILENAME: str = "toc_with_summaries.json"
    OUT_JSON_PATH: Path = OUTPUT_DIR / OUT_JSON_FILENAME
    ENCODING: str = "utf-8"

    # —— LLM 调用参数 —— #
    TEMPERATURE: float = float(os.getenv("SUM_TEMP", "0.3"))
    TARGET_SUMMARY_LEN: int = int(os.getenv("SUM_TARGET_LEN", "220"))   # 目标 ~200-300 字
    REQUEST_TIMEOUT: int = int(os.getenv("SUM_TIMEOUT", "120"))         # 单次请求超时秒

    # —— 文本切块策略 —— #
    MAX_CHARS: int = int(os.getenv("SUM_MAX_CHARS", "2000"))            # 每块最大长度（字符）
    CHUNK_OVERLAP: int = int(os.getenv("SUM_CHUNK_OVERLAP", "150"))     # 块间重叠

    # —— 并发控制（线程池）—— #
    MAX_WORKERS: int = int(os.getenv("SUM_MAX_WORKERS", "8"))           # 并发线程数
    RETRY_ATTEMPTS: int = int(os.getenv("SUM_RETRY_ATTEMPTS", "3"))     # 重试次数
    RETRY_BACKOFF_BASE: float = float(os.getenv("SUM_BACKOFF_BASE", "1.8"))  # 指数退避底数(秒)

    # —— 提示词模板 —— #
    LEAF_PROMPT: str = (
        "Role: 你是教材内容提炼专家，擅长提炼关键概念与术语。\n"
        "Task: 为以下小节正文生成 {target_len} 字左右的中文摘要，必须忠于原文，不添加外部知识。\n"
        "Constraints:\n"
        "- 语言简洁，保留学术术语\n"
        "- 说明核心概念/关键术语及其关系或作用\n"
        "- 只输出一段通顺的中文，不要列点编号\n"
        "Input Text:\n"
        "{content}\n"
    )

    AGG_PROMPT: str = (
        "Role: 你是教材内容提炼专家，擅长将多段小节摘要压缩为上级摘要。\n"
        "Task: 基于下列子节点摘要，生成 {target_len} 字左右的中文上级摘要（覆盖要点、突出层次）。\n"
        "Constraints:\n"
        "- 覆盖所有子摘要中的核心概念/术语\n"
        "- 去重与压缩，保持逻辑清晰\n"
        "- 不添加超出原文的信息\n"
        "Child Summaries:\n"
        "{content}\n"
    )
