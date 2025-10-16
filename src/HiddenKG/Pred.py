# -*- coding: utf-8 -*-
"""
Pred.py — Tree-KG §3.3.5 边预测（挖掘隐藏关系 · 语义+结构）

流程概览：
1) 读去重后的实体与已存在边（dedup_result.json）、读节点嵌入（node_embeddings.pkl）
2) 余弦相似度预筛（> Pred.COS_MIN），过滤已有边；Stage2 可选“仅跨小节”
3) 结构特征：AA（Adamic–Adar）、CA（共同祖先计数）
4) 综合评分：score = α·cos + β·AA + γ·CA（按 Stage 默认权重）
5) 候选配额（global topk、per-node cap），并发调用 LLM 进行强相关判定
6) 输出 pred_result.json；可选把新边写回 dedup_result.json

注意：
- 默认路径/参数来自 HiddenKG/config/pred.py 的 PredConfig（经 __init__.py 以 Pred 暴露）
- API_BASE/API_KEY/MODEL 由 HiddenKG/config/config.py 中的 APIConfig 提供
- 若后端服务不需要鉴权，API_KEY 留空即可（本代码不会发送空 Authorization 头）
"""

import os
import re
import json
import time
import math
import pickle
import logging
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, DefaultDict, Any

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from collections import defaultdict

# === 统一配置导入 ===
from HiddenKG.config import APIConfig, Pred

API_BASE = APIConfig.API_BASE
API_KEY = APIConfig.API_KEY
MODEL_NAME = APIConfig.MODEL_NAME

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
SESSION = requests.Session()

# LLM 调试开关（运行时由 Pred.DEBUG_LLM_DEFAULT/命令行控制）
DEBUG_LLM = False


def _dbg(msg: str) -> None:
    if DEBUG_LLM:
        print(f"[DEBUG_LLM] {msg}")


# --------------------------
# 数据结构（与 treekg KG Schema 对齐）
# --------------------------
@dataclass
class Neighbor:
    name: str
    snippet: str = ""  # 关系类型：entity_related|undirected|{具体关系}


@dataclass
class Occurrence:
    path: str  # 层级路径（如 "1/1.1/1.1.1"）
    node_id: str
    level: int
    title: str


@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str  # core/non-core
    original: str
    updated_description: str
    role: str
    occurrences: List[Occurrence] = field(default_factory=list)
    neighbors: List[Neighbor] = field(default_factory=list)


# --------------------------
# 工具函数
# --------------------------
def clean_path(path: str) -> str:
    """清理路径字符串，移除首尾空格和斜杠"""
    return (path or "").strip().strip("/")


# --------------------------
# 数据读取
# --------------------------
def read_dedup(path: str) -> Tuple[Dict[str, EntityItem], Dict[str, Set[str]]]:
    """读取去重后实体数据，构建实体字典与邻接表"""
    logging.info(f"[数据载入] 读取去重结果：{os.path.basename(path)}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"去重文件缺失：{path}")

    with open(path, "r", encoding="utf-8") as f:
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
            )
            for o in d.get("occurrences", [])
        ]
        neighbors = [
            Neighbor(
                name=n.get("name", ""),
                snippet=n.get("snippet", "entity_related|undirected")
            )
            for n in d.get("neighbors", [])
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

    logging.info(f"[数据载入] 实体 {len(ents)} 个，邻接表覆盖 {len(adj)} 个实体")
    return ents, adj


def read_embeddings(pkl_path: str) -> Tuple[List[str], np.ndarray]:
    """读取节点嵌入，返回（实体名列表，L2 归一化向量矩阵）"""
    logging.info(f"[数据载入] 读取节点嵌入：{os.path.basename(pkl_path)}")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"嵌入文件缺失：{pkl_path}")

    with open(pkl_path, "rb") as f:
        emb_dict = pickle.load(f)  # {name: np.ndarray(D,)}

    names = [n for n, v in emb_dict.items() if isinstance(v, np.ndarray) and v.size > 0]
    if not names:
        raise ValueError("嵌入文件无有效实体向量")

    embs = np.vstack([emb_dict[n] for n in names]).astype("float32")
    norm = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norm  # 归一化（余弦 = 点积）

    logging.info(f"[数据载入] 嵌入维度：{embs.shape[0]} × {embs.shape[1]}（已归一化）")
    return names, embs


# --------------------------
# 结构特征
# --------------------------
def adamic_adar(u_name: str, v_name: str, adj: Dict[str, Set[str]]) -> float:
    """Adamic–Adar：AA(u,v) = Σ_{w∈N(u)∩N(v)} 1/log|N(w)|"""
    u_neighbors = adj.get(u_name, set())
    v_neighbors = adj.get(v_name, set())
    commons = u_neighbors & v_neighbors

    aa = 0.0
    for w in commons:
        deg = len(adj.get(w, set()))
        if deg > 1:  # 避免 log(1)=0
            aa += 1.0 / math.log(deg)
    return round(aa, 6)


def extract_ancestors(occ: Occurrence, max_levels: int = 10) -> List[str]:
    """从 Occurrence.path 提取分层前缀路径列表"""
    path = clean_path(occ.path)
    if not path:
        return []
    parts = path.split("/")
    return ["/".join(parts[:i]) for i in range(1, min(len(parts), max_levels) + 1)]


def common_ancestors(u: EntityItem, v: EntityItem, granularity: int = 2) -> int:
    """共同祖先计数。granularity=2 表示章/节两级"""
    ua, va = set(), set()
    for o in u.occurrences:
        anc = extract_ancestors(o)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]
        ua.update(anc)
    for o in v.occurrences:
        anc = extract_ancestors(o)
        if granularity > 0 and anc:
            anc = anc[:min(len(anc), granularity)]
        va.update(anc)
    return len(ua & va)


def is_same_minimal_section(u: EntityItem, v: EntityItem) -> bool:
    """是否处于同一最小结构单元（比较完整 path 是否有交集）"""
    up = {clean_path(o.path) for o in u.occurrences if o.path}
    vp = {clean_path(o.path) for o in v.occurrences if o.path}
    return len(up & vp) > 0


# --------------------------
# LLM 评估
# --------------------------
def get_entity_text(ent: EntityItem) -> str:
    """优先使用 conv 增强后的描述；若为空，用实体名兜底；限制长度以适配上下文"""
    txt = (ent.updated_description or ent.original or ent.name or "").strip()
    return txt[:300] if txt else ent.name


def safe_json_loads(txt: str) -> Dict[str, Any]:
    """稳健解析 LLM JSON 响应：直接解析 → 外层截取 → 正则候选"""
    if not txt or not isinstance(txt, str):
        return {}
    s = txt.strip()
    # 1) 直接解析
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # 2) 外层花括号截取
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            obj = json.loads(s[i:j + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    # 3) 正则多候选
    for m in re.findall(r"\{.*?\}", s, flags=re.S):
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def llm_score_relation(
        u_name: str,
        v_name: str,
        ents: Dict[str, EntityItem],
        cos_val: float,
        aa_val: float,
        ca_val: int
) -> Dict[str, Any]:
    """
    LLM 评估实体对关系强度：
    返回 {"is_relevant": bool, "type": str, "strength": int, "description": str, "raw": str, "debug": str}
    """
    u_ent = ents.get(u_name)
    v_ent = ents.get(v_name)
    if not u_ent or not v_ent:
        msg = f"实体不存在（u={u_name}, v={v_name}）"
        return {"is_relevant": False, "type": "", "strength": 0, "description": msg, "raw": "", "debug": msg}

    if not API_BASE:
        msg = "API 配置缺失（API_BASE 为空）"
        logging.warning(f"[LLM评估] {msg}")
        return {"is_relevant": False, "type": "", "strength": 0, "description": msg, "raw": "", "debug": msg}

    system_prompt = Pred.PROMPT_SYSTEM
    user_prompt = Pred.PROMPT_USER_TEMPLATE.format(
        u_name=u_ent.name, u_desc=get_entity_text(u_ent),
        v_name=v_ent.name, v_desc=get_entity_text(v_ent),
        cos_val=cos_val, aa_val=aa_val, ca_val=ca_val
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": Pred.TEMPERATURE,
        "max_tokens": Pred.MAX_TOKENS
    }

    # 按需加 Authorization；API_KEY 为空则不加头
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    last_error = ""
    for attempt in range(1, Pred.API_RETRIES + 1):
        try:
            logging.debug(f"[LLM调用] 请求（{attempt}/{Pred.API_RETRIES}）：{u_ent.name}-{v_ent.name}")
            resp = SESSION.post(
                f"{API_BASE}{Pred.CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=Pred.API_TIMEOUT
            )
            resp.raise_for_status()
            raw_content = (resp.json().get("choices", [{}])[0]
                           .get("message", {})
                           .get("content", "")).strip()
            obj = safe_json_loads(raw_content)

            strength = int(obj.get("strength", 0))
            strength = max(0, min(10, strength))
            is_relevant = bool(obj.get("is_relevant", strength >= 5))
            rel_type = (obj.get("type", "") or "未分类").strip()
            description = (obj.get("description", f"{u_ent.name}与{v_ent.name}相关")).strip()

            return {
                "is_relevant": is_relevant,
                "type": rel_type,
                "strength": strength,
                "description": description,
                "raw": raw_content[:500],
                "debug": f"OK#{attempt}/{Pred.API_RETRIES} status={resp.status_code}"
            }

        except Exception as e:
            last_error = f"{type(e).__name__}: {str(e)[:200]}"
            logging.warning(f"[LLM调用] 失败（{attempt}/{Pred.API_RETRIES}）：{last_error}")
            _dbg(last_error)
            time.sleep(2 ** attempt)  # 指数退避

    fallback = f"所有 {Pred.API_RETRIES} 次尝试失败：{last_error}"
    return {"is_relevant": False, "type": "", "strength": 0, "description": fallback, "raw": "", "debug": fallback}


# --------------------------
# 主流程（§3.3.5）
# --------------------------
def run_pred(
        dedup_path: str = str(Pred.FILE_DEDUP_RESULT),
        emb_pkl: str = str(Pred.FILE_NODE_EMBEDDINGS),
        out_dir: str = str(Pred.OUTPUT_DIR),
        cos_min: float = Pred.COS_MIN,
        stage: int = Pred.STAGE_DEFAULT,  # 1=初始化连通性；2=跨小节补边
        beta: float = None,  # AA 权重（默认按 stage）
        gamma: float = None,  # CA 权重（默认按 stage）
        alpha: float = Pred.ALPHA,  # 语义权重
        cross_section_only: bool = Pred.CROSS_SECTION_ONLY,  # Stage2：仅跨小节
        topk: int = Pred.TOPK,
        per_node_cap: int = Pred.PER_NODE_CAP,
        workers: int = Pred.WORKERS,
        strength_min: int = Pred.STRENGTH_MIN,
        commit: bool = Pred.COMMIT_DEFAULT,  # 可在配置中设默认写回
        debug_llm: bool = Pred.DEBUG_LLM_DEFAULT
) -> None:
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    global DEBUG_LLM
    DEBUG_LLM = debug_llm

    # 提前校验API配置
    if not API_BASE or not MODEL_NAME:
        raise ValueError("API配置不完整（API_BASE或MODEL_NAME缺失）")

    logging.info("=" * 64)
    logging.info(f"[Tree-KG边预测] 开始（Stage {stage}）")
    logging.info("=" * 64)

    # 1) 权重初始化
    if beta is None or gamma is None:
        if stage == 1:
            beta, gamma = Pred.BETA_STAGE1, Pred.GAMMA_STAGE1
        else:
            beta, gamma = Pred.BETA_STAGE2, Pred.GAMMA_STAGE2
    logging.info(f"[权重配置] α={alpha}（cos）, β={beta}（AA）, γ={gamma}（CA）")

    # 2) 数据载入与对齐
    logging.info("\n[步骤1/5] 载入实体与嵌入...")
    ents, adj = read_dedup(dedup_path)
    emb_names, embeddings = read_embeddings(emb_pkl)
    aligned_names = [n for n in emb_names if n in ents]
    if not aligned_names:
        raise ValueError("实体与嵌入无交集，请检查输入文件")

    name2idx = {n: i for i, n in enumerate(emb_names)}
    aligned_idx = np.array([name2idx[n] for n in aligned_names], dtype=np.int64)
    aligned_embeddings = embeddings[aligned_idx]
    N = len(aligned_names)
    logging.info(f"[数据对齐] 完成：{N} 个实体（嵌入{len(emb_names)}，实体{len(ents)}）")

    # 3) 候选预筛（余弦）
    logging.info(f"\n[步骤2/5] 候选预筛（cos_min={cos_min}）...")
    cos_matrix = np.dot(aligned_embeddings, aligned_embeddings.T)  # 点积即余弦
    iu, ju = np.triu_indices(N, k=1)
    cos_values = cos_matrix[iu, ju]
    cos_mask = cos_values > cos_min
    fi, fj, fcos = iu[cos_mask], ju[cos_mask], cos_values[cos_mask]
    logging.info(f"[余弦筛选] 候选数：{len(fi)} 对（上三角总数{len(iu)}对）")

    # 过滤已存在边 + 可选：Stage2 仅跨小节
    candidates: List[Tuple[int, int, float]] = []
    filtered_existing = 0
    for idx in tqdm(range(len(fi)), desc="过滤边/跨小节", ncols=90):
        i, j = int(fi[idx]), int(fj[idx])
        u_name, v_name = aligned_names[i], aligned_names[j]
        # 已存在无向边
        if v_name in adj.get(u_name, set()) or u_name in adj.get(v_name, set()):
            filtered_existing += 1
            continue
        if stage == 2 and cross_section_only and is_same_minimal_section(ents[u_name], ents[v_name]):
            continue
        candidates.append((i, j, float(round(fcos[idx], 6))))
    logging.info(f"[候选筛选] 最终候选：{len(candidates)} 对（过滤已存在边 {filtered_existing} 对）")
    if not candidates:
        logging.warning("无候选对，提前结束")
        return

    # 4) 计算 AA/CA 与综合评分
    logging.info(f"\n[步骤3/5] 计算 AA/CA 与综合评分...")
    scored: List[Tuple[float, int, int, float, float, int]] = []
    for (i, j, cval) in tqdm(candidates, desc="计算特征/评分", ncols=90):
        u_name, v_name = aligned_names[i], aligned_names[j]
        aa_val = adamic_adar(u_name, v_name, adj)
        ca_val = common_ancestors(ents[u_name], ents[v_name], granularity=Pred.GRANULARITY_CA)
        score = alpha * cval + beta * aa_val + gamma * ca_val
        scored.append((round(score, 6), i, j, cval, aa_val, ca_val))
    scored.sort(key=lambda x: x[0], reverse=True)

    logging.info("[评分排序] 前 5 名候选：")
    for r in range(min(5, len(scored))):
        s, i, j, c, aa, ca = scored[r]
        logging.info(f"  Top{r + 1}: {aligned_names[i]} - {aligned_names[j]}  score={s}, cos={c}, AA={aa}, CA={ca}")

    # 5) 候选配额（per-node cap / global topk）
    logging.info(f"\n[步骤4/5] 候选配额（topk={topk}, per_node_cap={per_node_cap}）...")
    node_used: DefaultDict[str, int] = defaultdict(int)
    quota: List[Tuple[float, int, int, float, float, int]] = []
    for item in scored:
        _, i, j, _, _, _ = item
        u, v = aligned_names[i], aligned_names[j]
        if node_used[u] < per_node_cap and node_used[v] < per_node_cap:
            quota.append(item)
            node_used[u] += 1
            node_used[v] += 1
    if topk > 0 and len(quota) > topk:
        quota = quota[:topk]
    logging.info(f"[配额控制] 进入 LLM 评估：{len(quota)} 对")
    if not quota:
        logging.warning("无候选进入 LLM，提前结束")
        return

    # 6) LLM 评估与筛选
    logging.info(f"\n[步骤5/5] LLM评估（strength≥{strength_min} 保留，并发 {workers}）...")
    final_edges: List[Dict[str, Any]] = []
    llm_total = 0
    llm_success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        for item in quota:
            score, i, j, cosv, aav, cav = item
            u, v = aligned_names[i], aligned_names[j]
            fut = executor.submit(
                llm_score_relation,
                u_name=u, v_name=v, ents=ents, cos_val=cosv, aa_val=aav, ca_val=cav
            )
            future_map[fut] = (score, i, j, cosv, aav, cav)

        for fut in tqdm(as_completed(future_map), total=len(future_map), desc="LLM进度", ncols=90):
            llm_total += 1
            score, i, j, cosv, aav, cav = future_map[fut]
            u, v = aligned_names[i], aligned_names[j]
            try:
                res = fut.result()
                llm_success += 1
                if res.get("is_relevant") and int(res.get("strength", 0)) >= strength_min:
                    final_edges.append({
                        "u": u, "v": v,
                        "cos": cosv, "AA": aav, "CA": cav, "score": score,
                        "llm": {
                            "type": res.get("type", "未分类"),
                            "strength": int(res.get("strength", 0)),
                            "description": res.get("description", ""),
                            "debug": res.get("debug", "")
                        }
                    })
                    logging.info(f"[LLM结果] 保留边：{u}-{v} → strength={res.get('strength')}")
            except Exception as e:
                logging.error(f"[LLM结果] 处理 {u}-{v} 失败：{str(e)[:200]}")

    success_rate = (llm_success / llm_total * 100) if llm_total else 0.0
    retention_rate = (len(final_edges) / llm_total * 100) if llm_total else 0.0
    logging.info(f"\n[LLM汇总] 总调用 {llm_total} 次，成功 {llm_success} 次（{success_rate:.2f}%），"
                 f"保留边 {len(final_edges)} 条（{retention_rate:.2f}%）")

    # 7) 输出结果
    # 兼容自定义输出目录场景
    if out_dir == str(Pred.OUTPUT_DIR):
        out_path = str(Pred.FILE_PRED_RESULT)
    else:
        out_path = os.path.join(out_dir, os.path.basename(Pred.FILE_PRED_RESULT))

    pred_out = {
        "meta": {
            "task": "Tree-KG 边预测（§3.3.5）",
            "stage": stage,
            "params": {
                "alpha": alpha, "beta": beta, "gamma": gamma,
                "cos_min": cos_min, "strength_min": strength_min
            },
            "input": {"dedup": os.path.abspath(dedup_path), "emb": os.path.abspath(emb_pkl)},
            "time": {
                "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start)),
                "duration": round(time.time() - t_start, 2)
            }
        },
        "summary": {
            "entities": len(aligned_names),
            "candidates_total": len(candidates),
            "llm_calls": llm_total,
            "edges_added": len(final_edges)
        },
        "edges": final_edges
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pred_out, f, ensure_ascii=False, indent=2)
    logging.info(f"[结果输出] 完成：{out_path}（{os.path.getsize(out_path) / 1024:.2f} KB）")

    # 8) 可选：写回 dedup_result.json（追加无向边）
    if commit and final_edges:
        logging.info(f"\n[结果写回] 追加 {len(final_edges)} 条边到 {dedup_path}...")
        with open(dedup_path, "r", encoding="utf-8") as f:
            dedup_data = json.load(f)

        added = 0
        for e in final_edges:
            u, v = e["u"], e["v"]
            rel_snippet = f"entity_related|undirected|{e['llm']['type']}"
            for a, b in [(u, v), (v, u)]:
                if a in dedup_data.get("entities", {}):
                    neighbors = dedup_data["entities"][a].setdefault("neighbors", [])
                    names = [n.get("name") for n in neighbors]
                    if b not in names:
                        neighbors.append({"name": b, "snippet": rel_snippet})
                        added += 1

        with open(dedup_path, "w", encoding="utf-8") as f:
            json.dump(dedup_data, f, ensure_ascii=False, indent=2)
        logging.info(f"[结果写回] 完成：追加 {added} 个邻居（{len(final_edges)} 条无向边）")

    logging.info("\n" + "=" * 64)
    logging.info(
        f"[Tree-KG边预测] 结束（Stage {stage}） | 总耗时：{round(time.time() - t_start, 2)} 秒 | 保留边：{len(final_edges)} 条")
    logging.info("=" * 64)


# --------------------------
# 命令行接口
# --------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Tree-KG 边预测（§3.3.5）")
    # 输入输出
    parser.add_argument("--dedup", type=str, default=str(Pred.FILE_DEDUP_RESULT), help="去重结果路径（JSON）")
    parser.add_argument("--emb", type=str, default=str(Pred.FILE_NODE_EMBEDDINGS), help="节点嵌入路径（PKL）")
    parser.add_argument("--out", type=str, default=str(Pred.OUTPUT_DIR), help="输出目录")
    # 核心参数（论文配置）
    parser.add_argument("--stage", type=int, choices=[1, 2], default=Pred.STAGE_DEFAULT, help="阶段（1/2）")
    parser.add_argument("--cos_min", type=float, default=Pred.COS_MIN, help="余弦预筛阈值")
    parser.add_argument("--beta", type=float, default=None, help="AA 权重（默认按 stage）")
    parser.add_argument("--gamma", type=float, default=None, help="CA 权重（默认按 stage）")
    parser.add_argument("--alpha", type=float, default=Pred.ALPHA, help="语义权重（默认 0.6）")
    # 筛选配置
    parser.add_argument("--cross_section_only", action="store_true", default=Pred.CROSS_SECTION_ONLY,
                        help="Stage2：仅跨小节候选")
    parser.add_argument("--topk", type=int, default=Pred.TOPK, help="LLM 候选上限")
    parser.add_argument("--per_node_cap", type=int, default=Pred.PER_NODE_CAP, help="单实体候选上限")
    # LLM/系统
    parser.add_argument("--workers", type=int, default=Pred.WORKERS, help="LLM 并发数")
    parser.add_argument("--strength_min", type=int, default=Pred.STRENGTH_MIN, help="保留阈值（≥7 推荐）")
    parser.add_argument("--commit", action="store_true", default=Pred.COMMIT_DEFAULT,
                        help="将新增边写回 dedup_result.json")
    parser.add_argument("--debug_llm", action="store_true", default=Pred.DEBUG_LLM_DEFAULT, help="打印 LLM 调试信息")

    args = parser.parse_args()
    logging.info("[CLI参数] " + ", ".join([f"--{k}={v}" for k, v in vars(args).items()]))

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
