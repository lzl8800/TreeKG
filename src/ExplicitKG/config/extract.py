# -*- coding: utf-8 -*-
"""
config/extraction.py
实体与关系抽取（Extraction）专用配置：I/O 路径、并发、重试、限速、提示词等。
"""
import os
from pathlib import Path
from .config import OUTPUT_DIR

class ExtractionConfig:
    # —— I/O —— #
    IN_JSON_FILENAME: str = os.getenv("EXT_IN_JSON", "toc_with_summaries.json")
    IN_JSON_PATH: Path = OUTPUT_DIR / IN_JSON_FILENAME

    OUT_JSON_FILENAME: str = os.getenv("EXT_OUT_JSON", "toc_with_entities_and_relations.json")
    OUT_JSON_PATH: Path = OUTPUT_DIR / OUT_JSON_FILENAME

    ENCODING: str = os.getenv("EXT_ENCODING", "utf-8")

    # —— LLM 调用参数 —— #
    TEMPERATURE: float = float(os.getenv("EXT_TEMP", "0.3"))
    REQUEST_TIMEOUT: int = int(os.getenv("EXT_TIMEOUT", "120"))

    # —— 并发与稳健性 —— #
    MAX_WORKERS: int = int(os.getenv("EXT_MAX_WORKERS", "6"))          # 并发线程数
    RETRY_ATTEMPTS: int = int(os.getenv("EXT_RETRY_ATTEMPTS", "3"))    # 每次请求最大重试
    RETRY_BACKOFF_BASE: float = float(os.getenv("EXT_BACKOFF_BASE", "1.8"))  # 指数退避底数(秒)

    # —— 速率限制（可选）—— #
    # 设置每秒允许的最大请求数（整数）。<=0 表示关闭。
    RATE_LIMIT_QPS: float = float(os.getenv("EXT_RATE_LIMIT_QPS", "0"))
    # 额外的固定节流（秒），用于保守节流；0 关闭
    EXTRA_THROTTLE_SEC: float = float(os.getenv("EXT_EXTRA_THROTTLE", "0.0"))

    # —— 提示词模板 —— #
    ENTITY_PROMPT: str = (
        "Role: 你是课程知识图谱专家，擅长实体提取。\n"
        "Task: 从以下摘要中提取教材课程强相关实体，输出JSON格式。\n"
        "Constraints:\n"
        "  1. 实体为名词短语（如“库仑定律”）；\n"
        "  2. 同一实体的不同名称合并到“alias”（如“F=ma”→“牛顿第二定律”的alias）；\n"
        "  3. 实体类型需具体（如“物理定律”，而非“概念”）。\n"
        "Input Summary: {section_summary}\n"
        "Output Template:\n"
        "{{\n"
        '  "entities": [\n'
        "    {{\n"
        '      "name": "实体名",\n'
        '      "alias": ["别名1"],\n'
        '      "type": "实体类型",\n'
        '      "raw_content": "摘要中描述该实体的原文"\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Output must be valid JSON only, without explanation.\n"
    )

    RELATION_PROMPT: str = (
        "Role: 你是课程教学领域专家，擅长判断实体间关系。\n"
        "Task: 基于摘要和实体列表，判断实体间是否存在关系，输出JSON。\n"
        "Input:\n"
        "  摘要：{section_summary}\n"
        "  实体列表：{entity_list}\n"
        "Output Template:\n"
        "{{\n"
        '  "relations": [\n'
        "    {{\n"
        '      "source": "源实体名",\n'
        '      "target": "目标实体名",\n'
        '      "type": "关系类型（如“定义”）",\n'
        '      "description": "关系说明"\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Output must be valid JSON only, without explanation.\n"
    )
