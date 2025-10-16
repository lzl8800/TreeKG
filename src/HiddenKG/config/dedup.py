# -*- coding: utf-8 -*-
"""
HiddenKG/config/dedup_config.py
实体去重（Dedup）阶段配置：
- 完整I/O路径（输入实体、嵌入；输出结果）
- 近邻搜索与过滤规则（KNN、距离阈值、角色过滤）
- LLM调用参数（超时、重试、速率限制）
- 优化开关（快速合并、名称相似度、描述截断等）
"""
import os
from pathlib import Path
from .config import HIDDEN_KG_DIR, OUTPUT_DIR, APIConfig  # 复用全局配置


class DedupConfig:
    # ==================================================
    # 1. 路径与文件配置（输入/输出）
    # ==================================================
    # 输出根目录（复用全局OUTPUT_DIR）
    OUTPUT_DIR: Path = OUTPUT_DIR

    # 输入实体文件（优先聚合实体，其次conv实体）
    FILE_AGGR_ENTITIES: Path = OUTPUT_DIR / "aggr_entities.json"
    FILE_CONV_ENTITIES: Path = OUTPUT_DIR / "conv_entities.json"
    FILE_ENTITIES_IN: Path = Path(
        os.getenv("DEDUP_ENTITIES_IN",
                  str(FILE_AGGR_ENTITIES if FILE_AGGR_ENTITIES.exists() else FILE_CONV_ENTITIES))
    )

    # 输入嵌入文件（节点向量）
    FILE_EMBEDDINGS: Path = Path(
        os.getenv("DEDUP_EMBEDDINGS", str(OUTPUT_DIR / "node_embeddings.pkl"))
    )

    # 输出结果文件
    FILE_DEDUP_RESULT: Path = OUTPUT_DIR / "dedup_result.json"

    @staticmethod
    def ensure_paths() -> None:
        """确保所有输出目录存在"""
        DedupConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DedupConfig.FILE_DEDUP_RESULT.parent.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # 2. KNN近邻搜索配置
    # ==================================================
    KNN_NEIGHBORS: int = int(os.getenv("DEDUP_KNN_NEIGHBORS", "20"))  # 近邻数量（论文默认20）
    DIST_THRESHOLD: float = float(os.getenv("DEDUP_DIST_THRESHOLD", "0.55"))  # L2距离阈值（论文推荐0.55）
    MUTUAL_KNN: bool = bool(int(os.getenv("DEDUP_MUTUAL_KNN", "1")))  # 是否启用互为近邻（i∈N(j)且j∈N(i)）
    RANK_TOP: int = int(os.getenv("DEDUP_RANK_TOP", "12"))  # 仅处理前N名近邻（默认12→实际1~11名）

    # ==================================================
    # 3. 实体过滤规则
    # ==================================================
    # 角色过滤模式（严格/宽松）
    STRICT_SAME_ROLE: bool = bool(int(os.getenv("DEDUP_STRICT_SAME_ROLE", "0")))  # 0=宽松（默认），1=严格

    # 名称相似度过滤
    USE_NAME_JACCARD: bool = bool(int(os.getenv("DEDUP_USE_NAME_JACCARD", "1")))  # 是否启用名称过滤
    NAME_JACCARD_MIN: float = float(os.getenv("DEDUP_NAME_JACCARD_MIN", "0.25"))  # 名称相似度阈值
    NAME_MODE: str = os.getenv("DEDUP_NAME_MODE", "auto")  # 计算模式（auto/token/char/bigram）

    # 配额控制
    PER_NODE_CAP: int = int(os.getenv("DEDUP_PER_NODE_CAP", "30"))  # 每节点最大候选对数（0=无限制）
    MAX_NEIGHBORS_GLOBAL: int = int(os.getenv("DEDUP_MAX_NEIGHBORS_GLOBAL", "30"))  # 全局邻居上限

    # ==================================================
    # 4. LLM调用配置
    # ==================================================
    # 基础参数
    TEMPERATURE: float = float(os.getenv("DEDUP_TEMPERATURE", "0.0"))  # 去重判定用0温度（确定性优先）
    MAX_TOKENS: int = int(os.getenv("DEDUP_MAX_TOKENS", "128"))  # 判定结果最大tokens（无需长文本）
    LLM_WORKERS: int = int(os.getenv("DEDUP_LLM_WORKERS", "6"))  # 并发调用线程数

    # 超时与重试（优先环境变量→APIConfig→默认值）
    API_TIMEOUT: int = int(os.getenv(
        "DEDUP_API_TIMEOUT",
        str(getattr(APIConfig, "TIMEOUT_SECS", getattr(APIConfig, "TIMEOUT", 120)))
    ))
    RETRIES: int = int(os.getenv(
        "DEDUP_RETRIES",
        str(getattr(APIConfig, "RETRIES", 3))
    ))
    CHAT_COMPLETIONS_PATH: str = os.getenv(
        "DEDUP_CHAT_PATH",
        getattr(APIConfig, "CHAT_COMPLETIONS_PATH", "/chat/completions")
    )

    # 速率限制
    RATE_LIMIT_QPS: float = float(os.getenv("DEDUP_RATE_LIMIT_QPS", "0"))  # QPS限制（0=无限制）
    EXTRA_THROTTLE_SEC: float = float(os.getenv("DEDUP_EXTRA_THROTTLE_SEC", "0.0"))  # 额外节流时间
    RETRY_BACKOFF_BASE: float = float(os.getenv("DEDUP_BACKOFF_BASE", "1.8"))  # 重试退避基数（指数增长）

    # ==================================================
    # 5. 其他优化开关
    # ==================================================
    USE_ALIAS_FASTPATH: bool = bool(int(os.getenv("DEDUP_USE_ALIAS_FASTPATH", "0")))  # 别名交集快速合并
    TRUNCATE_DESC: bool = bool(int(os.getenv("DEDUP_TRUNCATE_DESC", "0")))  # 描述截断（降token）
    DESC_MAXLEN: int = int(os.getenv("DEDUP_DESC_MAXLEN", "120"))  # 截断后最大长度
    DRY_RUN: bool = bool(int(os.getenv("DEDUP_DRY_RUN", "0")))  # 干跑模式（不实际调用LLM/写入文件）

