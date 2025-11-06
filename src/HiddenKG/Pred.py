from __future__ import annotations

import os
import re
import json
import time
import math
import pickle
import logging
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

# 核心路径/日志
_HIDDEN_DIR = Path(__file__).resolve().parent
_OUT_DIR = _HIDDEN_DIR / "output"
_OUT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_DIR = _HIDDEN_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / f"pred_{time.strftime('%Y%m%d_%H%M%S')}.log"

# 数据文件（默认，按“仅文件名，统一拼接到 output”约定）
DEDUP_PATH_DEFAULT = _OUT_DIR / PRED["DEDUP_NAME"]
EMB_PATH_DEFAULT = _OUT_DIR / PRED["EMBEDDINGS_NAME"]
PRED_OUT_DEFAULT = _OUT_DIR / PRED["RESULT_NAME"]
ENCODING = PRED.get("ENCODING", "utf-8")

# LLM/会话配置
SESSION = requests.Session()
API_BASE = API.get("API_BASE", "")
API_KEY = (API.get("API_KEY") or "").strip()
MODEL_NAME = API.get("MODEL_NAME", "")

# ========== 日志：终端简洁，文件详细 ==========
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    fh = logging.FileHandler(_LOG_FILE, encoding=ENCODING)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"))

    logger.addHandler(ch)
    logger.addHandler(fh)

setup_logging()
logger = logging.getLogger()

DEBUG_LLM_DEFAULT = bool(PRED.get("DEBUG_LLM_DEFAULT", False))
DEBUG_LLM = DEBUG_LLM_DEFAULT

def _dbg(msg: str) -> None:
    if DEBUG_LLM:
        logger.debug(f"[DEBUG_LLM] {msg}")

# --------------------------
# 数据结构
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
def clean_path(path: str) -> str:
    return (path or "").strip().strip("/")

def get_entity_text(ent: EntityItem) -> str:
    txt = (ent.updated_description or ent.original or ent.name or "").strip()
    return txt[:300] if txt else ent.name

def safe_json_loads(txt: str) -> Dict[str, Any]:
    """容错 JSON 解析：优先严格，失败后剪出第一个 {...} 再试"""
    if not txt or not isinstance(txt, str):
        return {}
    s = txt.strip()
    # 严格
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        logger.debug(f"[safe_json_loads] 严格解析失败，尝试截取片段")

    # 截取第一段 {...}
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.debug(f"[safe_json_loads] 片段解析失败: {e}")

    return {}

# --------------------------
# 数据读取
def read_dedup(path: str) -> Tuple[Dict[str, EntityItem], Dict[str, Set[str]]]:
    logger.info(f"[步骤1/5] 载入去重结果：{os.path.basename(path)}")
    logger.debug(f"[read_dedup] 路径：{os.path.abspath(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"去重文件缺失：{path}")

    with open(path, "r", encoding=ENCODING) as f:
        raw = json.load(f)

    ents: Dict[str, EntityItem] = {}
    adj: Dict[str, Set[str]] = defaultdict(set)

    for name, d in raw.get("entities", {}).items():
        occurrences = [
            Occurrence(
                path=o.get("path", ""),
                node_id=o.get("node_id", ""),
                level=o.get("level", 0),
                title=o.get("title", "")
            ) for o in d.get("occurrences", [])
        ]
        neighbors = [
            Neighbor(
                name=n.get("name", ""),
                snippet=n.get("snippet", "entity_related|undirected")
            ) for n in d.get("neighbors", [])
        ]
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

    logger.info(f"[步骤1/5] 载入完成：实体 {len(ents)} 个")
    return ents, adj

def read_embeddings(pkl_path: str) -> Tuple[List[str], np.ndarray]:
    logger.info(f"[步骤1/5] 载入节点嵌入：{os.path.basename(pkl_path)}")
    logger.debug(f"[read_embeddings] 路径：{os.path.abspath(pkl_path)}")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"嵌入文件缺失：{pkl_path}")

    with open(pkl_path, "rb") as f:
        emb_dict = pickle.load(f)

    valid_emb = {n: v for n, v in emb_dict.items() if isinstance(v, np.ndarray) and v.size > 0}
    if not valid_emb:
        raise ValueError("嵌入文件无有效实体向量")

    names = list(valid_emb.keys())
    embs = np.vstack([valid_emb[n] for n in names]).astype("float32")

    norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norm

    logger.info(f"[步骤1/5] 嵌入载入完成：{len(names)} 个实体，维度 {embs.shape[1]}")
    return names, embs

# --------------------------
# 结构特征
def adamic_adar(u_name: str, v_name: str, adj: Dict[str, Set[str]]) -> float:
    u_neighbors = adj.get(u_name, set())
    v_neighbors = adj.get(v_name, set())
    common_neighbors = u_neighbors & v_neighbors

    aa_score = 0.0
    for w in common_neighbors:
        w_degree = len(adj.get(w, set()))
        if w_degree > 1:
            aa_score += 1.0 / math.log(w_degree)

    return round(aa_score, 6)

def extract_ancestors(occ: Occurrence, max_levels: int = 10) -> List[str]:
    path = clean_path(occ.path)
    if not path:
        return []
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, min(len(parts), max_levels) + 1)]

def common_ancestors(u: EntityItem, v: EntityItem, granularity: int = 2) -> int:
    u_anc, v_anc = set(), set()
    for occ in u.occurrences:
        anc = extract_ancestors(occ)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]
        u_anc.update(anc)
    for occ in v.occurrences:
        anc = extract_ancestors(occ)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]
        v_anc.update(anc)
    return len(u_anc & v_anc)

def is_same_minimal_section(u: EntityItem, v: EntityItem) -> bool:
    u_paths = {clean_path(o.path) for o in u.occurrences if o.path}
    v_paths = {clean_path(o.path) for o in v.occurrences if o.path}
    return len(u_paths & v_paths) > 0

# --------------------------
# LLM 关系评估
def llm_score_relation(
        u_name: str,
        v_name: str,
        ents: Dict[str, EntityItem],
        cos_val: float,
        aa_val: float,
        ca_val: int
) -> Dict[str, Any]:
    u_ent = ents.get(u_name)
    v_ent = ents.get(v_name)
    if not u_ent or not v_ent:
        msg = f"实体不存在（u={u_name}, v={v_name}）"
        logger.warning(f"[LLM评估] {msg}")
        return {"is_relevant": False, "type": "", "strength": 0, "description": msg, "raw": "", "debug": msg}

    if not API_BASE or not MODEL_NAME:
        msg = "API 配置缺失（API_BASE/MODEL_NAME 为空）"
        logger.error(f"[LLM评估] {msg}")
        return {"is_relevant": False, "type": "", "strength": 0, "description": msg, "raw": "", "debug": msg}

    system_prompt = PRED.get("PROMPT_SYSTEM", "你是关系评估助手，需基于实体描述和特征值判断两实体关系")
    user_prompt_template = PRED.get(
        "PROMPT_USER_TEMPLATE",
        "实体1：{u_name}，描述：{u_desc}\n实体2：{v_name}，描述：{v_desc}\n特征值：余弦相似度{cos_val}，AA指数{aa_val}，共同祖先数{ca_val}\n请返回JSON：{\"is_relevant\":是否相关（bool）,\"type\":关系类型（str）,\"strength\":强度（0-10）,\"description\":关系描述（str）}"
    )
    user_prompt = user_prompt_template.format(
        u_name=u_ent.name, u_desc=get_entity_text(u_ent),
        v_name=v_ent.name, v_desc=get_entity_text(v_ent),
        cos_val=cos_val, aa_val=aa_val, ca_val=ca_val
    )
    _dbg(f"[LLM] URL={API_BASE}{PRED.get('CHAT_COMPLETIONS_PATH','/chat/completions')} model={MODEL_NAME}")

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

    api_timeout = int(PRED.get("API_TIMEOUT", 120))
    api_retries = int(PRED.get("API_RETRIES", 3))
    chat_path = PRED.get("CHAT_COMPLETIONS_PATH", "/chat/completions")
    api_url = f"{API_BASE}{chat_path}"
    last_error = ""

    for attempt in range(1, api_retries + 1):
        try:
            resp = SESSION.post(api_url, headers=headers, json=payload, timeout=api_timeout)
            status = resp.status_code
            resp.raise_for_status()
            resp_json = resp.json()
            raw_content = (resp_json.get("choices", [{}])[0]
                           .get("message", {})
                           .get("content", "")).strip()
            raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()

            result = safe_json_loads(raw_content)
            if not result:
                logger.debug(f"[LLM解析失败] 片段: {raw_content[:300]}")
                return {
                    "is_relevant": False, "type": "", "strength": 0,
                    "description": "LLM返回非JSON或格式不合规",
                    "raw": raw_content[:500], "debug": "parse_error"
                }

            strength = int(result.get("strength", 0))
            strength = max(0, min(10, strength))
            is_relevant = bool(result.get("is_relevant", strength >= 5))
            rel_type = (result.get("type", "") or "未分类").strip()
            description = (result.get("description", f"{u_ent.name}与{v_ent.name}存在关联")).strip()

            logger.info(f"[LLM评估] {u_name}-{v_name}：关系类型「{rel_type}」，强度 {strength}")
            return {
                "is_relevant": is_relevant,
                "type": rel_type,
                "strength": strength,
                "description": description,
                "raw": raw_content[:500],
                "debug": f"成功（第{attempt}次），HTTP {status}"
            }
        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            logger.warning(f"[LLM调用] {u_name}-{v_name}：第{attempt}次失败：{last_error}")
            time.sleep(2 ** attempt)

    fallback_msg = f"LLM调用失败（{api_retries}次重试）：{last_error}"
    logger.error(f"[LLM评估] {u_name}-{v_name}：{fallback_msg}")
    return {"is_relevant": False, "type": "", "strength": 0, "description": fallback_msg, "raw": "", "debug": fallback_msg}

# --------------------------
# 主流程
def run_pred() -> None:
    """
    纯配置驱动：所有参数从 PredConfig 读取；不接收命令行参数
    """
    t_start = time.time()

    # ---- 从配置读取所有参数 ----
    dedup_path = str(DEDUP_PATH_DEFAULT)
    emb_pkl = str(EMB_PATH_DEFAULT)
    out_dir = str(_OUT_DIR)

    cos_min = float(PRED.get("COS_MIN", 0.62))
    stage = int(PRED.get("STAGE_DEFAULT", 1))
    alpha = float(PRED.get("ALPHA", 0.6))
    cross_section_only = bool(PRED.get("CROSS_SECTION_ONLY", False))
    topk = int(PRED.get("TOPK", 400))
    per_node_cap = int(PRED.get("PER_NODE_CAP", 6))
    workers = int(PRED.get("WORKERS", 6))
    strength_min = int(PRED.get("STRENGTH_MIN", 7))

    # beta/gamma 默认按 stage 切换
    if stage == 1:
        beta = float(PRED.get("BETA_STAGE1", 0.25))
        gamma = float(PRED.get("GAMMA_STAGE1", 0.15))
    else:
        beta = float(PRED.get("BETA_STAGE2", 0.20))
        gamma = float(PRED.get("GAMMA_STAGE2", 0.20))

    # ---- 检查 API 配置 ----
    if not API_BASE or not MODEL_NAME:
        logger.error("API 配置不完整（API_BASE 或 MODEL_NAME 缺失），无法执行 LLM 评估")
        raise ValueError("API配置不完整（API_BASE 或 MODEL_NAME 缺失）")

    # ---- 日志头 ----
    logger.info("=" * 64)
    logger.info(f"[Tree-KG边预测] 启动（Stage {stage}）")
    logger.info("=" * 64)
    logger.info(f"[参数配置] 权重：α={alpha}（余弦）, β={beta}（AA）, γ={gamma}（CA）")
    logger.info(f"[参数配置] 筛选：cos_min={cos_min}, strength_min={strength_min}, topk={topk}")
    logger.info(f"[参数配置] per_node_cap={per_node_cap}, workers={workers}, cross_section_only={cross_section_only}")
    logger.info(f"[路径配置] dedup={dedup_path} | emb={emb_pkl} | out_dir={out_dir}")

    # 1. 数据载入与对齐
    logger.info("\n[步骤1/5] 数据载入与对齐...")
    ents, adj = read_dedup(dedup_path)
    emb_names, embeddings = read_embeddings(emb_pkl)

    aligned_names = [name for name in emb_names if name in ents]
    if not aligned_names:
        raise ValueError("实体与嵌入无交集，请检查输入文件")

    name2idx = {name: idx for idx, name in enumerate(emb_names)}
    aligned_idx = np.array([name2idx[name] for name in aligned_names], dtype=np.int64)
    aligned_embeddings = embeddings[aligned_idx]
    N = len(aligned_names)
    logger.info(f"[步骤1/5] 对齐完成：{N} 个实体")

    # 2. 候选对预筛（余弦）
    logger.info(f"\n[步骤2/5] 候选对预筛（cos_min={cos_min}）...")
    cos_matrix = np.dot(aligned_embeddings, aligned_embeddings.T)
    iu, ju = np.triu_indices(N, k=1)
    cos_values = cos_matrix[iu, ju]
    cos_mask = cos_values > cos_min
    fi, fj, fcos = iu[cos_mask], ju[cos_mask], cos_values[cos_mask]
    logger.info(f"[步骤2/5] 余弦筛选：{len(fi)} 个候选对")

    # 3. 二次过滤（已存在边 + 跨章节）
    candidates: List[Tuple[int, int, float]] = []
    filtered_existing = 0
    filtered_section = 0
    for idx in tqdm(range(len(fi)), desc="[步骤2/5] 过滤候选对", ncols=80):
        i, j = int(fi[idx]), int(fj[idx])
        u_name, v_name = aligned_names[i], aligned_names[j]
        if v_name in adj.get(u_name, set()) or u_name in adj.get(v_name, set()):
            filtered_existing += 1
            continue
        if stage == 2 and cross_section_only and is_same_minimal_section(ents[u_name], ents[v_name]):
            filtered_section += 1
            continue
        candidates.append((i, j, float(round(fcos[idx], 6))))
    logger.info(f"[步骤2/5] 最终候选：{len(candidates)} 个（已存在边{filtered_existing}，同章节{filtered_section}）")
    if not candidates:
        logger.warning("无有效候选对，提前结束流程")
        _write_edges(out_dir, [])
        _tail_log(t_start)
        return

    # 4. 结构特征 + 综合评分
    logger.info(f"\n[步骤3/5] 结构特征计算与综合评分...")
    gran_ca = int(PRED.get("GRANULARITY_CA", 2))
    scored_candidates: List[Tuple[float, int, int, float, float, int]] = []
    for (i, j, cos_val) in tqdm(candidates, desc="[步骤3/5] 计算特征", ncols=80):
        u_name, v_name = aligned_names[i], aligned_names[j]
        aa_val = adamic_adar(u_name, v_name, adj)
        ca_val = common_ancestors(ents[u_name], ents[v_name], granularity=gran_ca)
        total_score = round(alpha * cos_val + beta * aa_val + gamma * ca_val, 6)
        scored_candidates.append((total_score, i, j, cos_val, aa_val, ca_val))
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    logger.info("\n[步骤3/5] 评分Top5候选对：")
    for rank in range(min(5, len(scored_candidates))):
        score, i, j, cos_val, aa_val, ca_val = scored_candidates[rank]
        u_name, v_name = aligned_names[i], aligned_names[j]
        logger.info(f"  Top{rank + 1}: {u_name} ↔ {v_name} | 总分{score}（cos{cos_val} + AA{aa_val} + CA{ca_val}）")

    # 5. 配额控制（TopK + per_node）
    logger.info(f"\n[步骤4/5] 候选对配额控制（topk={topk}, per_node_cap={per_node_cap}）...")
    node_used: DefaultDict[str, int] = defaultdict(int)
    quota_candidates: List[Tuple[float, int, int, float, float, int]] = []
    for item in scored_candidates:
        _, i, j, _, _, _ = item
        u_name, v_name = aligned_names[i], aligned_names[j]
        if node_used[u_name] < per_node_cap and node_used[v_name] < per_node_cap:
            quota_candidates.append(item)
            node_used[u_name] += 1
            node_used[v_name] += 1
        if topk > 0 and len(quota_candidates) >= topk:
            break
    logger.info(f"[步骤4/5] 配额完成：{len(quota_candidates)} 个进入 LLM 评估")
    if not quota_candidates:
        logger.warning("无候选对进入 LLM 评估，提前结束流程")
        _write_edges(out_dir, [])
        _tail_log(t_start)
        return

    # 6. LLM 评估
    logger.info(f"\n[步骤5/5] LLM 关系评估（并发{workers}线程，strength≥{strength_min}保留）...")
    final_edges: List[Dict[str, Any]] = []
    llm_total = len(quota_candidates)
    llm_success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for item in quota_candidates:
            score, i, j, cos_val, aa_val, ca_val = item
            u_name, v_name = aligned_names[i], aligned_names[j]
            fut = executor.submit(llm_score_relation, u_name, v_name, ents, cos_val, aa_val, ca_val)
            future_map[fut] = (score, i, j, cos_val, aa_val, ca_val)

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="[步骤5/5] LLM评估进度", ncols=80):
            score, i, j, cos_val, aa_val, ca_val = future_map[fut]
            u_name, v_name = aligned_names[i], aligned_names[j]
            try:
                llm_result = fut.result()
                llm_success += 1
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

    success_rate = (llm_success / llm_total * 100) if llm_total > 0 else 0.0
    retention_rate = (len(final_edges) / llm_total * 100) if llm_total > 0 else 0.0
    logger.info(f"\n[步骤5/5] LLM 评估汇总：")
    logger.info(f"  总调用：{llm_total} 次 | 成功：{llm_success} 次（{success_rate:.1f}%）")
    logger.info(f"  保留边：{len(final_edges)} 条（{retention_rate:.1f}%）")

    # 输出：仅边列表
    out_path = str(PRED_OUT_DEFAULT)  # 统一输出到 HiddenKG/output
    _write_edges_to_file(out_path, final_edges)
    logger.info(f"[结果输出] 完成：{os.path.abspath(out_path)}（仅边，共 {len(final_edges)} 条）")

    _tail_log(t_start)

def _tail_log(t_start: float) -> None:
    total_duration = round(time.time() - t_start, 2)
    logger.info("\n" + "=" * 64)
    logger.info(f"[Tree-KG边预测] 流程结束，耗时 {total_duration} 秒")
    logger.info(f"详细日志：{os.path.abspath(str(_LOG_FILE))}")
    logger.info("=" * 64)

def _write_edges(out_dir: str, edges: List[Dict[str, Any]]) -> None:
    out_path = str(PRED_OUT_DEFAULT) if out_dir == str(_OUT_DIR) else os.path.join(out_dir, os.path.basename(PRED_OUT_DEFAULT))
    _write_edges_to_file(out_path, edges)

def _write_edges_to_file(out_path: str, edges: List[Dict[str, Any]]) -> None:
    with open(out_path, "w", encoding=ENCODING) as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_pred()
