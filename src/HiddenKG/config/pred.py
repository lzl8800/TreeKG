# -*- coding: utf-8 -*-
"""
HiddenKG/config/pred.py
Tree-KG §3.3.5（边预测）配置：
- 路径/文件名
- 算法超超参（含两阶段权重）
- LLM 请求参数（温度/max_tokens/超时/重试/并发）
- 提示模板（system/user）

说明：
- API_BASE/API_KEY/MODEL 仍从 ExplicitKG 的 APIConfig 读取（不在此文件存放密钥）。
- 仅提供 PredConfig 类命名空间，移除旧代码兼容的模块级常量
"""

import os
from pathlib import Path
# 使用相对导入，兼容本地“相对路径运行”的习惯
from .config import OUTPUT_DIR as _OUTPUT_DIR, APIConfig  # 复用全局 OUTPUT_DIR 与 APIConfig


class PredConfig:
    """边预测（§3.3.5）配置命名空间（业务侧常用别名：from HiddenKG.config import Pred as C）"""

    # ====== 路径与文件名 ======
    OUTPUT_DIR: Path = _OUTPUT_DIR
    FILE_AGGR_RESULT: Path = OUTPUT_DIR / "aggr_entities.json"
    FILE_CONV_ENTITIES: Path = OUTPUT_DIR / "conv_entities.json"
    FILE_CONV_PROMPTS: Path = OUTPUT_DIR / "conv_prompts.json"
    FILE_DEDUP_RESULT: Path = OUTPUT_DIR / "dedup_result.json"
    FILE_NODE_EMBEDDINGS: Path = OUTPUT_DIR / "node_embeddings.pkl"
    FILE_PRED_RESULT: Path = OUTPUT_DIR / "pred_result.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ====== 算法默认参数（与 treekg 对齐）======
    STAGE_DEFAULT: int = 1  # 1=初始化连通性；2=跨小节补边
    COS_MIN: float = 0.60  # 语义预筛阈值（cosine）
    ALPHA: float = 0.60  # 语义权重
    BETA_STAGE1: float = 0.0  # AA 权重（Stage1）
    GAMMA_STAGE1: float = 0.4  # CA 权重（Stage1）
    BETA_STAGE2: float = 0.3  # AA 权重（Stage2）
    GAMMA_STAGE2: float = 0.1  # CA 权重（Stage2）
    GRANULARITY_CA: int = 2  # 共同祖先粒度（章/节）
    CROSS_SECTION_ONLY: bool = False  # Stage2：仅跨小节候选

    TOPK: int = 10000  # 进入 LLM 的候选上限
    PER_NODE_CAP: int = 200  # 单实体候选上限
    STRENGTH_MIN: int = 8  # LLM 置信阈值（≥7保留）

    # ====== LLM/并发/容错 ======
    WORKERS: int = int(os.getenv("PRED_WORKERS", "8"))
    # 兼容 TIMEOUT_SECS / TIMEOUT 两种字段名；同时允许用 env 覆盖
    API_TIMEOUT: int = int(os.getenv(
        "LLM_TIMEOUT_SECS",
        str(getattr(APIConfig, "TIMEOUT_SECS",
                    getattr(APIConfig, "TIMEOUT", 120)))
    ))
    API_RETRIES: int = int(os.getenv("LLM_RETRIES", "3"))

    # ====== LLM 请求细节（可按需调参/复现实验）======
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "300"))
    CHAT_COMPLETIONS_PATH: str = os.getenv("LLM_CHAT_COMPLETIONS_PATH", "/chat/completions")

    # ====== 提示模板（可按域、任务风格调整）======
    PROMPT_SYSTEM: str = "你是Physics领域专家，仅输出指定格式JSON，不额外解释。"

    PROMPT_USER_TEMPLATE: str = """
        任务：评估两个Physics实体的关系强度，输出JSON。
        约束：
        1. strength：0-10整数（10=强相关，0=无相关）
        2. is_relevant：strength≥5为true，否则false
        3. type：关系类型（如“概念从属”“计算关联”）
        4. description：10-50字说明关系

        实体信息：
        - 实体U：{u_name}，描述：{u_desc}
        - 实体V：{v_name}，描述：{v_desc}
        - 辅助特征：cos={cos_val:.4f}，AA={aa_val:.4f}，CA={ca_val}

        必须输出的JSON：
        {{
                  "is_relevant": true/false,
          "type": "关系类型",
          "strength": 0-10,
          "description": "关系说明"
        }}
            """.strip()

    # ====== 其他常用开关 ======
    DEBUG_LLM_DEFAULT: bool = bool(int(os.getenv("DEBUG_LLM", "0")))
    COMMIT_DEFAULT: bool = bool(int(os.getenv("PRED_COMMIT_DEFAULT", "0")))


__all__ = ["PredConfig"]
