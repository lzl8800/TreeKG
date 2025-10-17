# -*- coding: utf-8 -*-
"""
Pred.py — Tree-KG §3.3.5 边预测（挖掘隐藏关系 · 语义+结构）
- 读取 HiddenKG/config/config.yaml 并合并 include_files（相对 config 目录解析）
- 路径只存文件名，代码统一拼接 HiddenKG/output
- 详细日志记录到 logs 文件夹，终端只输出必要信息
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import pickle
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, DefaultDict, Any, Optional

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import defaultdict
from pathlib import Path
import yaml


# ========== 配置加载 ==========
def load_merged_config() -> dict:
    """
    读取 HiddenKG/config/config.yaml，并合并 include_files（相对 config.yaml 所在目录解析）
    """
    hidden_dir = Path(__file__).resolve().parent
    cfg_path = hidden_dir / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    merged = dict(base)
    for rel in base.get("include_files", []) or []:
        inc_path = (cfg_path.parent / rel).resolve()
        if not inc_path.exists():
            raise FileNotFoundError(f"未找到 include 文件：{inc_path}")
        with inc_path.open("r", encoding="utf-8") as ff:
            sub = yaml.safe_load(ff) or {}
        merged.update(sub)
    return merged


_CFG = load_merged_config()
API = _CFG["APIConfig"]
PRED = _CFG["PredConfig"]

# 核心路径配置（与 output 同级创建 logs 文件夹）
_HIDDEN_DIR = Path(__file__).resolve().parent
_OUT_DIR = _HIDDEN_DIR / "output"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
# 日志文件夹：与 output 同级，命名为 logs
_LOG_DIR = _HIDDEN_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
# 日志文件：按日期+任务命名，避免覆盖
_LOG_FILE = _LOG_DIR / f"pred_{time.strftime('%Y%m%d_%H%M%S')}.log"

# 数据文件路径（默认值）
DEDUP_PATH_DEFAULT = _OUT_DIR / PRED["DEDUP_NAME"]
EMB_PATH_DEFAULT = _OUT_DIR / PRED["EMBEDDINGS_NAME"]
PRED_OUT_DEFAULT = _OUT_DIR / PRED["RESULT_NAME"]
ENCODING = PRED.get("ENCODING", "utf-8")

# LLM/会话配置
SESSION = requests.Session()
API_BASE = API.get("API_BASE", "")
API_KEY = (API.get("API_KEY") or "").strip()
MODEL_NAME = API.get("MODEL_NAME", "")


# ========== 日志配置核心：分离终端与文件输出 ==========
def setup_logging():
    # 1. 获取根日志器，清除默认配置
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # 根日志器设为 DEBUG，确保所有级别日志都能被捕获
    logger.handlers.clear()

    # 2. 终端处理器：只输出 INFO 及以上（必要信息），格式简化
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # 终端只显示关键步骤
    console_formatter = logging.Formatter("%(message)s")  # 终端仅输出内容，无时间/级别
    console_handler.setFormatter(console_formatter)

    # 3. 文件处理器：记录 DEBUG 及以上（所有详细信息），格式完整
    file_handler = logging.FileHandler(_LOG_FILE, encoding=ENCODING)
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有细节
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    )  # 包含时间、级别、代码位置，便于调试
    file_handler.setFormatter(file_formatter)

    # 4. 添加处理器到日志器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# 初始化日志配置
setup_logging()
logger = logging.getLogger()

# 调试开关（仅影响 LLM 细节日志）
DEBUG_LLM_DEFAULT = bool(PRED.get("DEBUG_LLM_DEFAULT", False))
DEBUG_LLM = False


def _dbg(msg: str) -> None:
    """调试日志：仅在 DEBUG_LLM 开启时记录到文件"""
    if DEBUG_LLM:
        logger.debug(f"[DEBUG_LLM] {msg}")


# --------------------------
# 数据结构定义
# --------------------------
@dataclass
class Neighbor:
    name: str
    snippet: str = ""  # 关系类型：entity_related|undirected|{具体关系}


@dataclass
class Occurrence:
    path: str
    node_id: str
    level: int
    title: str


@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str
    original: str
    updated_description: str
    role: str
    occurrences: List[Occurrence] = field(default_factory=list)
    neighbors: List[Neighbor] = field(default_factory=list)


# --------------------------
# 工具函数
# --------------------------
def clean_path(path: str) -> str:
    """清理路径字符串，去除首尾空格和斜杠"""
    return (path or "").strip().strip("/")


def get_entity_text(ent: EntityItem) -> str:
    """提取实体的描述文本，优先用更新后描述，截断到300字符"""
    txt = (ent.updated_description or ent.original or ent.name or "").strip()
    return txt[:300] if txt else ent.name


def safe_json_loads(txt: str) -> Dict[str, Any]:
    """安全解析JSON字符串，处理格式不完整的情况"""
    if not txt or not isinstance(txt, str):
        return {}
    s = txt.strip()
    # 尝试直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        logger.debug(f"[safe_json_loads] 直接解析失败，尝试提取JSON片段：{s[:100]}...")

    # 尝试提取 {} 包裹的内容
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(s[i:j + 1])
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.debug(f"[safe_json_loads] 提取片段解析失败：{str(e)}，片段：{s[i:j + 1]}")

    # 尝试匹配所有 {} 片段
    for m in re.findall(r"\{.*?\}", s, flags=re.S):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    logger.debug(f"[safe_json_loads] 所有解析方式失败，返回空字典：{s[:100]}...")
    return {}


# --------------------------
# 数据读取函数
# --------------------------
def read_dedup(path: str) -> Tuple[Dict[str, EntityItem], Dict[str, Set[str]]]:
    """读取去重结果文件，构建实体字典和邻接表"""
    logger.info(f"[步骤1/5] 载入去重结果：{os.path.basename(path)}")
    logger.debug(f"[read_dedup] 去重文件完整路径：{os.path.abspath(path)}")

    if not os.path.exists(path):
        logger.error(f"[read_dedup] 去重文件不存在：{path}")
        raise FileNotFoundError(f"去重文件缺失：{path}")

    with open(path, "r", encoding=ENCODING) as f:
        raw = json.load(f)

    ents: Dict[str, EntityItem] = {}
    adj: Dict[str, Set[str]] = defaultdict(set)

    for name, d in raw.get("entities", {}).items():
        # 构建 Occurrences 列表
        occurrences = [
            Occurrence(
                path=o.get("path", ""),
                node_id=o.get("node_id", ""),
                level=o.get("level", 0),
                title=o.get("title", "")
            )
            for o in d.get("occurrences", [])
        ]
        # 构建 Neighbors 列表
        neighbors = [
            Neighbor(
                name=n.get("name", ""),
                snippet=n.get("snippet", "entity_related|undirected")
            )
            for n in d.get("neighbors", [])
        ]
        # 构建 EntityItem
        item = EntityItem(
            name=name,
            alias=d.get("alias", []),
            type=d.get("type", "non-core"),
            original=d.get("original", ""),
            updated_description=d.get("updated_description", d.get("original", "")),
            role=d.get("role", ""),
            occurrences=occurrences,
            neighbors=neighbors
        )
        ents[name] = item
        adj[name].update(n.name for n in neighbors)

    logger.info(f"[步骤1/5] 载入完成：实体 {len(ents)} 个，邻接表覆盖 {len(adj)} 个实体")
    logger.debug(f"[read_dedup] 示例实体：{list(ents.keys())[:3] if len(ents) >= 3 else list(ents.keys())}")
    return ents, adj


def read_embeddings(pkl_path: str) -> Tuple[List[str], np.ndarray]:
    """读取节点嵌入文件，返回实体名列表和归一化后的嵌入矩阵"""
    logger.info(f"[步骤1/5] 载入节点嵌入：{os.path.basename(pkl_path)}")
    logger.debug(f"[read_embeddings] 嵌入文件完整路径：{os.path.abspath(pkl_path)}")

    if not os.path.exists(pkl_path):
        logger.error(f"[read_embeddings] 嵌入文件不存在：{pkl_path}")
        raise FileNotFoundError(f"嵌入文件缺失：{pkl_path}")

    with open(pkl_path, "rb") as f:
        emb_dict = pickle.load(f)  # 格式：{实体名: np.ndarray(嵌入维度,)}

    # 过滤无效嵌入
    valid_emb = {n: v for n, v in emb_dict.items() if isinstance(v, np.ndarray) and v.size > 0}
    if not valid_emb:
        logger.error(f"[read_embeddings] 嵌入文件无有效向量：{pkl_path}")
        raise ValueError("嵌入文件无有效实体向量")

    # 构建实体名列表和嵌入矩阵
    names = list(valid_emb.keys())
    embs = np.vstack([valid_emb[n] for n in names]).astype("float32")

    # 嵌入归一化（L2归一化）
    norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12  # 避免除零
    embs = embs / norm

    logger.info(f"[步骤1/5] 嵌入载入完成：{len(names)} 个实体，维度 {embs.shape[1]}（已归一化）")
    logger.debug(f"[read_embeddings] 嵌入矩阵形状：{embs.shape}，示例实体：{names[:3]}")
    return names, embs


# --------------------------
# 结构特征计算函数
# --------------------------
def adamic_adar(u_name: str, v_name: str, adj: Dict[str, Set[str]]) -> float:
    """计算 Adamic-Adar 指数（衡量两节点的共同邻居重要性）"""
    u_neighbors = adj.get(u_name, set())
    v_neighbors = adj.get(v_name, set())
    common_neighbors = u_neighbors & v_neighbors

    aa_score = 0.0
    for w in common_neighbors:
        w_degree = len(adj.get(w, set()))
        if w_degree > 1:  # 避免 log(1) 为 0
            aa_score += 1.0 / math.log(w_degree)

    aa_score = round(aa_score, 6)
    logger.debug(f"[adamic_adar] {u_name}-{v_name}：共同邻居 {len(common_neighbors)} 个，AA指数 {aa_score}")
    return aa_score


def extract_ancestors(occ: Occurrence, max_levels: int = 10) -> List[str]:
    """从 Occurrence 的 path 中提取祖先路径（按层级拆分）"""
    path = clean_path(occ.path)
    if not path:
        return []
    path_parts = path.split("/")
    # 提取前 max_levels 层的祖先路径
    ancestors = ["/".join(path_parts[:i]) for i in range(1, min(len(path_parts), max_levels) + 1)]
    logger.debug(f"[extract_ancestors] 路径 {path} → 祖先：{ancestors}")
    return ancestors


def common_ancestors(u: EntityItem, v: EntityItem, granularity: int = 2) -> int:
    """计算两实体的共同祖先数量（控制层级粒度）"""
    u_ancestors = set()
    for occ in u.occurrences:
        anc = extract_ancestors(occ)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]  # 控制层级粒度
        u_ancestors.update(anc)

    v_ancestors = set()
    for occ in v.occurrences:
        anc = extract_ancestors(occ)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]
        v_ancestors.update(anc)

    common_count = len(u_ancestors & v_ancestors)
    logger.debug(f"[common_ancestors] {u.name}-{v.name}：共同祖先 {common_count} 个（粒度 {granularity}）")
    return common_count


def is_same_minimal_section(u: EntityItem, v: EntityItem) -> bool:
    """判断两实体是否属于同一最小章节（基于 Occurrence 的 path）"""
    u_paths = {clean_path(o.path) for o in u.occurrences if o.path}
    v_paths = {clean_path(o.path) for o in v.occurrences if o.path}
    is_same = len(u_paths & v_paths) > 0
    logger.debug(
        f"[is_same_minimal_section] {u.name}-{v.name}：是否同章节 {is_same}（u路径数 {len(u_paths)}，v路径数 {len(v_paths)}）")
    return is_same


# --------------------------
# LLM 关系评估函数
# --------------------------
def llm_score_relation(
        u_name: str,
        v_name: str,
        ents: Dict[str, EntityItem],
        cos_val: float,
        aa_val: float,
        ca_val: int
) -> Dict[str, Any]:
    """调用 LLM 评估两实体的关系，返回评估结果"""
    # 1. 检查实体是否存在
    u_ent = ents.get(u_name)
    v_ent = ents.get(v_name)
    if not u_ent or not v_ent:
        msg = f"实体不存在（u={u_name}, v={v_name}）"
        logger.warning(f"[LLM评估] {msg}")
        return {
            "is_relevant": False,
            "type": "",
            "strength": 0,
            "description": msg,
            "raw": "",
            "debug": msg
        }

    # 2. 检查 API 配置
    if not API_BASE or not MODEL_NAME:
        msg = "API 配置缺失（API_BASE/MODEL_NAME 为空）"
        logger.error(f"[LLM评估] {msg}")
        return {
            "is_relevant": False,
            "type": "",
            "strength": 0,
            "description": msg,
            "raw": "",
            "debug": msg
        }

    # 3. 构建 Prompt（从配置读取模板）
    system_prompt = PRED.get("PROMPT_SYSTEM", "你是关系评估助手，需基于实体描述和特征值判断两实体关系")
    user_prompt_template = PRED.get(
        "PROMPT_USER_TEMPLATE",
        "实体1：{u_name}，描述：{u_desc}\n实体2：{v_name}，描述：{v_desc}\n特征值：余弦相似度{cos_val}，AA指数{aa_val}，共同祖先数{ca_val}\n请返回JSON：{\"is_relevant\":是否相关（bool）,\"type\":关系类型（str）,\"strength\":强度（0-10）,\"description\":关系描述（str）}"
    )
    user_prompt = user_prompt_template.format(
        u_name=u_ent.name,
        u_desc=get_entity_text(u_ent),
        v_name=v_ent.name,
        v_desc=get_entity_text(v_ent),
        cos_val=cos_val,
        aa_val=aa_val,
        ca_val=ca_val
    )
    logger.debug(f"[LLM评估] {u_name}-{v_name}：System Prompt：{system_prompt[:100]}...")
    logger.debug(f"[LLM评估] {u_name}-{v_name}：User Prompt：{user_prompt[:200]}...")

    # 4. 构建 API 请求参数
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": float(PRED.get("TEMPERATURE", 0.2)),
        "max_tokens": int(PRED.get("MAX_TOKENS", 256))
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # 5. API 调用配置（重试、超时）
    api_timeout = int(PRED.get("API_TIMEOUT", 120))
    api_retries = int(PRED.get("API_RETRIES", 3))
    chat_path = PRED.get("CHAT_COMPLETIONS_PATH", "/chat/completions")
    api_url = f"{API_BASE}{chat_path}"
    last_error = ""

    # 6. 带重试的 API 调用
    for attempt in range(1, api_retries + 1):
        try:
            _dbg(f"[LLM调用] {u_name}-{v_name}：第{attempt}次请求，URL：{api_url}")
            resp = SESSION.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=api_timeout
            )
            resp.raise_for_status()  # 触发 HTTP 错误（如404、500）
            resp_json = resp.json()
            raw_content = (resp_json.get("choices", [{}])[0]
                           .get("message", {})
                           .get("content", "")).strip()
            logger.debug(f"[LLM调用] {u_name}-{v_name}：原始响应：{raw_content[:300]}...")

            # 7. 解析 LLM 响应（安全解析JSON）
            result = safe_json_loads(raw_content)
            # 补全缺失字段，确保格式统一
            strength = int(result.get("strength", 0))
            strength = max(0, min(10, strength))  # 强度限制在 0-10
            is_relevant = bool(result.get("is_relevant", strength >= 5))  # 强度≥5默认相关
            rel_type = (result.get("type", "") or "未分类").strip()
            description = (result.get("description", f"{u_ent.name}与{v_ent.name}存在关联")).strip()

            logger.info(f"[LLM评估] {u_name}-{v_name}：关系类型「{rel_type}」，强度 {strength}")
            return {
                "is_relevant": is_relevant,
                "type": rel_type,
                "strength": strength,
                "description": description,
                "raw": raw_content[:500],  # 截断长响应，避免日志过大
                "debug": f"成功（第{attempt}次），HTTP状态码：{resp.status_code}"
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning(f"[LLM调用] {u_name}-{v_name}：第{attempt}次失败：{last_error}")
            time.sleep(2 ** attempt)  # 指数退避重试

    # 8. 所有重试失败
    fallback_msg = f"LLM调用失败（{api_retries}次重试）：{last_error}"
    logger.error(f"[LLM评估] {u_name}-{v_name}：{fallback_msg}")
    return {
        "is_relevant": False,
        "type": "",
        "strength": 0,
        "description": fallback_msg,
        "raw": "",
        "debug": fallback_msg
    }


# --------------------------
# 主流程函数
# --------------------------
def run_pred(
        dedup_path: str = str(DEDUP_PATH_DEFAULT),
        emb_pkl: str = str(EMB_PATH_DEFAULT),
        out_dir: str = str(_OUT_DIR),
        cos_min: float = float(PRED.get("COS_MIN", 0.62)),
        stage: int = int(PRED.get("STAGE_DEFAULT", 1)),
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        alpha: float = float(PRED.get("ALPHA", 0.6)),
        cross_section_only: bool = bool(PRED.get("CROSS_SECTION_ONLY", False)),
        topk: int = int(PRED.get("TOPK", 400)),
        per_node_cap: int = int(PRED.get("PER_NODE_CAP", 6)),
        workers: int = int(PRED.get("WORKERS", 6)),
        strength_min: int = int(PRED.get("STRENGTH_MIN", 7)),
        commit: bool = bool(PRED.get("COMMIT_DEFAULT", False)),
        debug_llm: bool = DEBUG_LLM_DEFAULT
) -> None:
    """主流程：边预测全流程执行"""
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    global DEBUG_LLM
    DEBUG_LLM = debug_llm

    # 前置检查：API 配置
    if not API_BASE or not MODEL_NAME:
        logger.error("API 配置不完整（API_BASE 或 MODEL_NAME 缺失），无法执行 LLM 评估")
        raise ValueError("API配置不完整（API_BASE 或 MODEL_NAME 缺失）")

    # 1. 初始化日志与参数
    logger.info("=" * 64)
    logger.info(f"[Tree-KG边预测] 启动（Stage {stage}）")
    logger.info("=" * 64)
    # 补全 beta/gamma 参数（按 stage 选择默认值）
    if beta is None or gamma is None:
        if stage == 1:
            beta = float(PRED.get("BETA_STAGE1", 0.25))
            gamma = float(PRED.get("GAMMA_STAGE1", 0.15))
        else:
            beta = float(PRED.get("BETA_STAGE2", 0.20))
            gamma = float(PRED.get("GAMMA_STAGE2", 0.20))
    logger.info(f"[参数配置] 权重：α={alpha}（余弦）, β={beta}（AA）, γ={gamma}（CA）")
    logger.info(f"[参数配置] 筛选：cos_min={cos_min}, strength_min={strength_min}, topk={topk}")
    logger.debug(f"[参数配置] 完整参数：dedup_path={dedup_path}, emb_pkl={emb_pkl}, out_dir={out_dir}")

    # 2. 数据载入与对齐
    logger.info("\n[步骤1/5] 数据载入与对齐...")
    ents, adj = read_dedup(dedup_path)
    emb_names, embeddings = read_embeddings(emb_pkl)
    # 对齐实体（只保留同时存在于实体字典和嵌入中的实体）
    aligned_names = [name for name in emb_names if name in ents]
    if not aligned_names:
        logger.error("实体与嵌入无交集，无法继续")
        raise ValueError("实体与嵌入无交集，请检查输入文件")
    # 构建实体名到索引的映射
    name2idx = {name: idx for idx, name in enumerate(emb_names)}
    aligned_idx = np.array([name2idx[name] for name in aligned_names], dtype=np.int64)
    aligned_embeddings = embeddings[aligned_idx]
    N = len(aligned_names)
    logger.info(f"[步骤1/5] 对齐完成：{N} 个实体（嵌入{len(emb_names)}个，实体{len(ents)}个）")

    # 3. 候选对预筛（基于余弦相似度）
    logger.info(f"\n[步骤2/5] 候选对预筛（cos_min={cos_min}）...")
    # 计算余弦相似度矩阵（上三角，避免重复）
    cos_matrix = np.dot(aligned_embeddings, aligned_embeddings.T)
    iu, ju = np.triu_indices(N, k=1)  # k=1：排除自身（i=j）
    cos_values = cos_matrix[iu, ju]
    # 筛选相似度高于阈值的候选对
    cos_mask = cos_values > cos_min
    fi, fj, fcos = iu[cos_mask], ju[cos_mask], cos_values[cos_mask]
    logger.info(f"[步骤2/5] 余弦筛选：{len(fi)} 个候选对（总可能{len(iu)}个）")

    # 4. 候选对二次过滤（已存在边 + 跨章节筛选）
    candidates: List[Tuple[int, int, float]] = []
    filtered_existing = 0
    filtered_section = 0
    for idx in tqdm(range(len(fi)), desc="[步骤2/5] 过滤候选对", ncols=80):
        i, j = int(fi[idx]), int(fj[idx])
        u_name, v_name = aligned_names[i], aligned_names[j]
        # 过滤已存在的边（双向检查）
        if v_name in adj.get(u_name, set()) or u_name in adj.get(v_name, set()):
            filtered_existing += 1
            continue
        # Stage2 可选：只保留跨章节候选对
        if stage == 2 and cross_section_only:
            if is_same_minimal_section(ents[u_name], ents[v_name]):
                filtered_section += 1
                continue
        # 保留有效候选对
        candidates.append((i, j, float(round(fcos[idx], 6))))
    logger.info(
        f"[步骤2/5] 最终候选：{len(candidates)} 个（过滤已存在边{filtered_existing}个，过滤同章节{filtered_section}个）")
    if not candidates:
        logger.warning("无有效候选对，提前结束流程")
        return

    # 5. 结构特征计算与综合评分
    logger.info(f"\n[步骤3/5] 结构特征计算与综合评分...")
    gran_ca = int(PRED.get("GRANULARITY_CA", 2))  # CA 特征的层级粒度
    scored_candidates: List[Tuple[float, int, int, float, float, int]] = []
    for (i, j, cos_val) in tqdm(candidates, desc="[步骤3/5] 计算特征", ncols=80):
        u_name, v_name = aligned_names[i], aligned_names[j]
        # 计算结构特征
        aa_val = adamic_adar(u_name, v_name, adj)
        ca_val = common_ancestors(ents[u_name], ents[v_name], granularity=gran_ca)
        # 综合评分（加权求和）
        total_score = alpha * cos_val + beta * aa_val + gamma * ca_val
        total_score = round(total_score, 6)
        scored_candidates.append((total_score, i, j, cos_val, aa_val, ca_val))
    # 按综合评分降序排序
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # 打印Top5候选对（终端和日志都输出）
    logger.info("\n[步骤3/5] 评分Top5候选对：")
    for rank in range(min(5, len(scored_candidates))):
        score, i, j, cos_val, aa_val, ca_val = scored_candidates[rank]
        u_name, v_name = aligned_names[i], aligned_names[j]
        logger.info(f"  Top{rank + 1}: {u_name} ↔ {v_name} | 总分{score}（cos{cos_val} + AA{aa_val} + CA{ca_val}）")

    # 6. 候选对配额控制（TopK + 单节点上限）
    logger.info(f"\n[步骤4/5] 候选对配额控制（topk={topk}, per_node_cap={per_node_cap}）...")
    node_used: DefaultDict[str, int] = defaultdict(int)  # 记录每个节点已用配额
    quota_candidates: List[Tuple[float, int, int, float, float, int]] = []
    for item in scored_candidates:
        _, i, j, _, _, _ = item
        u_name, v_name = aligned_names[i], aligned_names[j]
        # 检查两节点是否都还有配额
        if node_used[u_name] < per_node_cap and node_used[v_name] < per_node_cap:
            quota_candidates.append(item)
            node_used[u_name] += 1
            node_used[v_name] += 1
        # 达到TopK上限则停止
        if topk > 0 and len(quota_candidates) >= topk:
            break
    logger.info(f"[步骤4/5] 配额完成：{len(quota_candidates)} 个候选对进入 LLM 评估")
    if not quota_candidates:
        logger.warning("无候选对进入 LLM 评估，提前结束流程")
        return

    # 7. LLM 批量评估（多线程）
    logger.info(f"\n[步骤5/5] LLM 关系评估（并发{workers}线程，strength≥{strength_min}保留）...")
    final_edges: List[Dict[str, Any]] = []
    llm_total = len(quota_candidates)
    llm_success = 0

    # 多线程执行 LLM 评估
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务到线程池
        future_map = {}
        for item in quota_candidates:
            score, i, j, cos_val, aa_val, ca_val = item
            u_name, v_name = aligned_names[i], aligned_names[j]
            fut = executor.submit(
                llm_score_relation,
                u_name=u_name,
                v_name=v_name,
                ents=ents,
                cos_val=cos_val,
                aa_val=aa_val,
                ca_val=ca_val
            )
            future_map[fut] = (score, i, j, cos_val, aa_val, ca_val)

        # 实时获取任务结果
        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="[步骤5/5] LLM评估进度", ncols=80):
            score, i, j, cos_val, aa_val, ca_val = future_map[fut]
            u_name, v_name = aligned_names[i], aligned_names[j]
            try:
                llm_result = fut.result()
                llm_success += 1
                # 筛选强度达标的边
                if llm_result["is_relevant"] and llm_result["strength"] >= strength_min:
                    final_edges.append({
                        "u": u_name,
                        "v": v_name,
                        "cos": cos_val,
                        "AA": aa_val,
                        "CA": ca_val,
                        "composite_score": score,
                        "llm": {
                            "type": llm_result["type"],
                            "strength": llm_result["strength"],
                            "description": llm_result["description"],
                            "debug_info": llm_result["debug"]
                        }
                    })
            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)[:150]}"
                logger.error(f"[LLM结果处理] {u_name}-{v_name} 失败：{err_msg}")

    # 8. LLM 评估结果汇总
    success_rate = (llm_success / llm_total * 100) if llm_total > 0 else 0.0
    retention_rate = (len(final_edges) / llm_total * 100) if llm_total > 0 else 0.0
    logger.info(f"\n[步骤5/5] LLM 评估汇总：")
    logger.info(f"  总调用：{llm_total} 次 | 成功：{llm_success} 次（{success_rate:.1f}%）")
    logger.info(f"  保留边：{len(final_edges)} 条（{retention_rate:.1f}%）")

    # 9. 输出预测结果（JSON文件）
    logger.info(f"\n[结果输出] 生成预测结果文件...")
    # 确定输出路径
    if out_dir == str(_OUT_DIR):
        out_path = str(PRED_OUT_DEFAULT)
    else:
        out_path = os.path.join(out_dir, os.path.basename(PRED_OUT_DEFAULT))
    # 构建输出内容
    output_data = {
        "meta": {
            "task": "Tree-KG 边预测（§3.3.5）",
            "stage": stage,
            "run_time": {
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                "duration_seconds": round(time.time() - t_start, 2)
            },
            "parameters": {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "cos_min": cos_min, "strength_min": strength_min,
                "topk": topk, "per_node_cap": per_node_cap,
                "workers": workers, "cross_section_only": cross_section_only
            },
            "input_files": {
                "dedup_result": os.path.abspath(dedup_path),
                "node_embeddings": os.path.abspath(emb_pkl)
            },
            "log_file": os.path.abspath(str(_LOG_FILE))
        },
        "summary": {
            "total_aligned_entities": N,
            "total_candidates_before_filter": len(candidates),
            "total_llm_calls": llm_total,
            "total_valid_edges": len(final_edges)
        },
        "predicted_edges": final_edges
    }
    # 写入文件
    with open(out_path, "w", encoding=ENCODING) as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(out_path) / 1024  # 转换为 KB
    logger.info(f"[结果输出] 完成：{os.path.abspath(out_path)}（{file_size:.2f} KB）")

    # 10. 可选：将新边写回去重结果文件（追加无向边）
    if commit and final_edges:
        logger.info(f"\n[结果写回] 追加 {len(final_edges)} 条边到去重文件...")
        # 读取原去重文件
        with open(dedup_path, "r", encoding=ENCODING) as f:
            dedup_data = json.load(f)
        # 追加新边（双向添加邻居）
        added_count = 0
        for edge in final_edges:
            u_name, v_name = edge["u"], edge["v"]
            rel_type = edge["llm"]["type"]
            rel_snippet = f"entity_related|undirected|{rel_type}"
            # 双向添加邻居（确保无重复）
            for a_name, b_name in [(u_name, v_name), (v_name, u_name)]:
                if a_name not in dedup_data.get("entities", {}):
                    logger.debug(f"[结果写回] 实体 {a_name} 不在去重文件中，跳过")
                    continue
                # 获取实体的邻居列表
                neighbors = dedup_data["entities"][a_name].setdefault("neighbors", [])
                existing_neighbors = [n.get("name") for n in neighbors]
                # 避免重复添加
                if b_name not in existing_neighbors:
                    neighbors.append({"name": b_name, "snippet": rel_snippet})
                    added_count += 1
        # 写回文件
        with open(dedup_path, "w", encoding=ENCODING) as f:
            json.dump(dedup_data, f, ensure_ascii=False, indent=2)
        logger.info(f"[结果写回] 完成：追加 {added_count} 个邻居记录（{len(final_edges)} 条无向边）")

    # 11. 流程结束汇总
    total_duration = round(time.time() - t_start, 2)
    logger.info("\n" + "=" * 64)
    logger.info(f"[Tree-KG边预测] 流程结束（Stage {stage}）")
    logger.info(f"  总耗时：{total_duration} 秒")
    logger.info(f"  最终保留边：{len(final_edges)} 条")
    logger.info(f"  详细日志：{os.path.abspath(str(_LOG_FILE))}")
    logger.info("=" * 64)


# --------------------------
# 命令行接口（CLI）
# --------------------------
def main() -> None:
    """命令行入口：解析参数并启动主流程"""
    parser = argparse.ArgumentParser(description="Tree-KG 边预测（§3.3.5）- 挖掘隐藏关系")
    # 输入文件参数
    parser.add_argument("--dedup", type=str, default=str(DEDUP_PATH_DEFAULT),
                        help=f"去重结果文件路径（默认：{DEDUP_PATH_DEFAULT}）")
    parser.add_argument("--emb", type=str, default=str(EMB_PATH_DEFAULT),
                        help=f"节点嵌入文件路径（默认：{EMB_PATH_DEFAULT}）")
    # 输出参数
    parser.add_argument("--out", type=str, default=str(_OUT_DIR),
                        help=f"结果输出目录（默认：{_OUT_DIR}）")
    # 流程控制参数
    parser.add_argument("--stage", type=int, choices=[1, 2], default=int(PRED.get("STAGE_DEFAULT", 1)),
                        help="预测阶段（1：全量候选，2：跨章节候选，默认：1）")
    parser.add_argument("--cos_min", type=float, default=float(PRED.get("COS_MIN", 0.62)),
                        help="余弦相似度筛选阈值（默认：0.62）")
    parser.add_argument("--beta", type=float, default=None,
                        help=f"AA指数权重（默认：Stage1=0.25，Stage2=0.20）")
    parser.add_argument("--gamma", type=float, default=None,
                        help=f"共同祖先权重（默认：Stage1=0.15，Stage2=0.20）")
    parser.add_argument("--alpha", type=float, default=float(PRED.get("ALPHA", 0.6)),
                        help="余弦相似度权重（默认：0.6）")
    parser.add_argument("--cross_section_only", action="store_true",
                        default=bool(PRED.get("CROSS_SECTION_ONLY", False)),
                        help="仅保留跨章节候选对（仅Stage2生效，默认：False）")
    # 配额与并发参数
    parser.add_argument("--topk", type=int, default=int(PRED.get("TOPK", 400)),
                        help="进入LLM评估的候选对数量上限（默认：400）")
    parser.add_argument("--per_node_cap", type=int, default=int(PRED.get("PER_NODE_CAP", 6)),
                        help="单个节点的候选对配额上限（默认：6）")
    parser.add_argument("--workers", type=int, default=int(PRED.get("WORKERS", 6)),
                        help="LLM评估的并发线程数（默认：6）")
    # 结果筛选参数
    parser.add_argument("--strength_min", type=int, default=int(PRED.get("STRENGTH_MIN", 7)),
                        help="LLM强度阈值（≥此值保留，默认：7）")
    parser.add_argument("--commit", action="store_true", default=bool(PRED.get("COMMIT_DEFAULT", False)),
                        help="将预测边写回原去重文件（默认：False）")
    # 调试参数
    parser.add_argument("--debug_llm", action="store_true", default=DEBUG_LLM_DEFAULT,
                        help="开启LLM调试日志（默认：False）")

    # 解析参数
    args = parser.parse_args()
    # 记录CLI参数到日志
    cli_params = [f"{k}={v}" for k, v in vars(args).items()]
    logger.info(f"[CLI] 启动参数：{', '.join(cli_params)}")

    # 启动主流程
    run_pred(
        dedup_path=args.dedup,
        emb_pkl=args.emb,
        out_dir=args.out,
        cos_min=args.cos_min,
        stage=args.stage,
        beta=args.beta,
        gamma=args.gamma,
        alpha=args.alpha,
        cross_section_only=args.cross_section_only,
        topk=args.topk,
        per_node_cap=args.per_node_cap,
        workers=args.workers,
        strength_min=args.strength_min,
        commit=args.commit,
        debug_llm=args.debug_llm
    )


if __name__ == "__main__":
    main()
