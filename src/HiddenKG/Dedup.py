# -*- coding: utf-8 -*-
"""
Dedup.py — Tree-KG §3.3.4 实体去重（优化·单文件输出·含中文友好名称相似·诊断）

总体流程（论文框架保留）：
  KNN(top-k, L2) → 距离阈值 + 角色过滤 → LLM 判定（按距离升序） → 并查集合并

工程优化（默认开启，兼顾召回与成本）：
  - 显式向量 L2 归一化（阈值语义稳定）
  - 互为近邻 mutual-kNN（i∈N(j) 且 j∈N(i)）
  - 同角色支持“严格/宽松”两种模式（默认宽松：两边都有且不同才过滤）
  - 限制近邻排名 rank_top（默认前11名：1..11）
  - 每节点候选配额 per_node_cap（默认30）
  - 名称 Jaccard 轻量过滤（默认开，阈值0.25；新增 name_mode：token|char|bigram|auto）
  - “拒绝原因计数器”诊断，精确看到各规则的挡量

输出：仅 1 个文件 HiddenKG/output/dedup_result.json
"""

import os
import re
import json
import time
import pickle
import logging
import argparse
from typing import Dict, List, Tuple, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from pathlib import Path

# === 导入配置（核心：从 dedup_config 读取所有参数） ===
from HiddenKG.config import APIConfig
from HiddenKG.config import Dedup as DedupConfig

# 初始化路径（确保输出目录存在）
DedupConfig.ensure_paths()

# === 全局变量与日志配置 ===
# 日志
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("Dedup")

# 会话保持（复用连接）
SESSION = requests.Session()

# === 导入抽离的模块 ===
from Dedup.data_structures import Occurrence, Neighbor, EntityItem
from Dedup.knn import knn_topk
from Dedup.llm import llm_is_same, _fallback_is_same, _normalize_name
from Dedup.name_similarity import _name_jaccard_mode


# =========================
# I/O 工具函数
# =========================
def read_entities(infile: str) -> Dict[str, EntityItem]:
    """从JSON文件读取实体数据并转换为EntityItem对象"""
    with open(infile, "r", encoding="utf-8") as f:
        raw = json.load(f)
    entities: Dict[str, EntityItem] = {}
    for name, d in raw.items():
        entities[name] = EntityItem(
            name=name,
            alias=d.get("alias", []),
            type=d.get("type", ""),
            original=d.get("original", ""),
            updated_description=d.get("updated_description", ""),
            role=d.get("role", d.get("local_role", "")),
            occurrences=[Occurrence(**o) for o in d.get("occurrences", [])],
            neighbors=[Neighbor(** n) for n in d.get("neighbors", [])],
        )
    logger.info(f"载入实体 {len(entities)} 个：{os.path.basename(infile)}")
    return entities


def read_embeddings(pkl_path: str) -> Tuple[List[str], np.ndarray]:
    """从PKL文件读取实体嵌入向量"""
    with open(pkl_path, "rb") as f:
        emb_dict = pickle.load(f)  # {name: np.ndarray(D,)}
    names = list(emb_dict.keys())
    embs = np.vstack([emb_dict[n] for n in names]).astype("float32")
    logger.info(f"载入嵌入 {embs.shape[0]} x {embs.shape[1]}：{os.path.basename(pkl_path)}")
    return names, embs


# =========================
# 并查集数据结构（用于实体聚类）
# =========================
class DSU:
    def __init__(self, n: int):
        self.fa = list(range(n))  # 父节点
        self.sz = [1] * n  # 集合大小

    def find(self, x: int) -> int:
        """路径压缩查找根节点"""
        while self.fa[x] != x:
            self.fa[x] = self.fa[self.fa[x]]  # 压缩路径
            x = self.fa[x]
        return x

    def union(self, a: int, b: int) -> bool:
        """按大小合并两个集合"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # 已在同一集合
        # 小集合合并到大连合
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.fa[rb] = ra
        self.sz[ra] += self.sz[rb]
        return True


# =========================
# 主流程函数
# =========================
def run_dedup(
    entities_in: str = None,
    emb_pkl: str = None,
    out_dir: str = str(DedupConfig.OUTPUT_DIR),
    # 从配置文件读取默认参数，支持外部传入覆盖
    k_neighbors: int = DedupConfig.KNN_NEIGHBORS,
    dist_thresh: float = DedupConfig.DIST_THRESHOLD,
    workers: int = DedupConfig.LLM_WORKERS,
    mutual_knn: bool = DedupConfig.MUTUAL_KNN,
    rank_top: int = DedupConfig.RANK_TOP,
    per_node_cap: int = DedupConfig.PER_NODE_CAP,
    strict_same_role: bool = DedupConfig.STRICT_SAME_ROLE,
    use_name_jaccard: bool = DedupConfig.USE_NAME_JACCARD,
    name_jaccard_min: float = DedupConfig.NAME_JACCARD_MIN,
    name_mode: str = DedupConfig.NAME_MODE,
    use_alias_fastpath: bool = DedupConfig.USE_ALIAS_FASTPATH,
    truncate_desc: bool = DedupConfig.TRUNCATE_DESC,
    desc_maxlen: int = DedupConfig.DESC_MAXLEN,
):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- 1. 加载实体与嵌入文件 ----------------
    logger.info("📘 [1/5] 加载实体与嵌入文件...")
    # 输入实体路径（优先外部传入，其次配置文件默认）
    ent_path = entities_in or str(DedupConfig.FILE_ENTITIES_IN)
    if not os.path.exists(ent_path):
        raise FileNotFoundError(f"实体文件不存在：{ent_path}")
    entities = read_entities(ent_path)

    # 输入嵌入路径（优先外部传入，其次配置文件默认）
    emb_path = emb_pkl or str(DedupConfig.FILE_EMBEDDINGS)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"嵌入文件不存在：{emb_path}")
    names_all, embs_all = read_embeddings(emb_path)
    name2idx = {n: i for i, n in enumerate(names_all)}

    # 对齐实体（仅保留实体文件和嵌入文件共有的实体）
    names = [n for n in names_all if n in entities]
    if not names:
        raise ValueError("实体与嵌入文件无交集，请检查输入文件")
    idx_keep = np.array([name2idx[n] for n in names], dtype=np.int64)
    embs = embs_all[idx_keep].astype("float32")

    # 显式L2归一化（确保距离阈值语义稳定）
    embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

    logger.info(f"✅ 对齐后参与去重的实体数：{len(names)}")

    # ---------------- 2. 近邻检索（KNN） ----------------
    logger.info("📘 [2/5] 开始近邻检索（FAISS/Sklearn）...")
    # 检索topk+1（含自身）
    dists, idxs = knn_topk(embs, topk=k_neighbors + 1)

    # 提取实体属性用于过滤
    ents: List[EntityItem] = [entities[n] for n in names]
    roles = [e.role or "" for e in ents]  # 角色列表
    dsu = DSU(len(names))  # 初始化并查集

    # 构建互近邻表（每个实体的近邻集合，不含自身）
    N = len(names)
    nbr_sets = [set(idxs[i, 1:]) for i in range(N)]

    # ---------------- 3. 候选对过滤（多级规则） ----------------
    logger.info("📘 [3/5] 收集候选对（距离阈值 + 角色 + 名称Jaccard + 互近邻 + 排名/配额）...")
    candidates: List[Tuple[int, int, float]] = []  # 最终候选对 (i, j, dist)
    cnt_pairs = 0  # 原始近邻对数
    used = [0] * N  # 记录每个实体已使用的候选配额

    # 拒绝原因计数器（诊断用）
    rej_dist = rej_role = rej_jac = rej_mutual = rej_cap = 0

    # 限制近邻排名范围（至少取前2名）
    rank_limit = max(2, int(rank_top))
    for i in range(N):
        # 遍历当前实体的近邻（跳过自身，取前rank_limit-1名）
        for rank in range(1, min(idxs.shape[1], rank_limit)):
            j = idxs[i, rank]
            if j <= i or j >= N:  # 避免重复对（i<j）和越界
                continue
            cnt_pairs += 1
            dist = float(dists[i, rank])

            # 1. 距离阈值过滤
            if dist >= dist_thresh:
                rej_dist += 1
                continue

            # 2. 角色过滤（严格/宽松模式）
            if strict_same_role:
                if roles[i] != roles[j]:
                    rej_role += 1
                    continue
            else:
                # 宽松模式：两边都有角色且不同才过滤
                if roles[i] and roles[j] and roles[i] != roles[j]:
                    rej_role += 1
                    continue

            # 3. 名称Jaccard相似度过滤
            if use_name_jaccard:
                jac_sim = _name_jaccard_mode(ents[i].name, ents[j].name, name_mode)
                if jac_sim < name_jaccard_min:
                    rej_jac += 1
                    continue

            # 4. 互近邻过滤
            if mutual_knn and (i not in nbr_sets[j]):
                rej_mutual += 1
                continue

            # 5. 每节点候选配额过滤
            if per_node_cap > 0 and (used[i] >= per_node_cap or used[j] >= per_node_cap):
                rej_cap += 1
                continue

            # 所有过滤通过，加入候选对
            candidates.append((i, j, dist))
            used[i] += 1
            used[j] += 1

    # 输出过滤诊断信息
    logger.info(f"📌 候选对（通过初筛）：{len(candidates)} / 原始对数：{cnt_pairs}")
    logger.info(f"[诊断] 过滤统计: 距离={rej_dist} 角色={rej_role} 名称Jaccard={rej_jac} 互近邻={rej_mutual} 配额={rej_cap}")
    # 输出距离分位数（辅助调整阈值）
    if candidates:
        kept_d = np.array([d for (_, _, d) in candidates], dtype=np.float32)
        qs = np.percentile(kept_d, [5, 10, 25, 50, 75, 90, 95])
        logger.info(f"[诊断] 初筛距离分位数 p5..p95: {np.round(qs, 4)}")

    # ---------------- 3.5 可选：别名交集快速合并 ----------------
    rep_fast_merge = 0
    kept_for_llm: List[Tuple[int, int, float]] = candidates
    if use_alias_fastpath:
        logger.info("📘 [3.5] 快速通道：别名/名称交集直接合并（可选）...")
        def alias_intersect(a: EntityItem, b: EntityItem) -> bool:
            """检查两个实体的名称/别名是否有交集"""
            alias_a = {_normalize_name(x) for x in ([a.name] + a.alias)}
            alias_b = {_normalize_name(x) for x in ([b.name] + b.alias)}
            return len(alias_a & alias_b) > 0

        tmp_kept: List[Tuple[int, int, float]] = []
        for (i, j, dist) in candidates:
            if alias_intersect(ents[i], ents[j]):
                if dsu.union(i, j):
                    rep_fast_merge += 1
            else:
                tmp_kept.append((i, j, dist))
        kept_for_llm = tmp_kept
        logger.info(f"✅ 快速合并完成：{rep_fast_merge} 条；进入 LLM 的对数：{len(kept_for_llm)}")
    else:
        logger.info(f"📘 [3.5] 跳过快速通道；进入 LLM 的对数：{len(kept_for_llm)}")

    # ---------------- 4. LLM 并行判定 ----------------
    logger.info(f"📘 [4/5] LLM 并行判定（并发={workers}；按距离升序）...")
    # 按距离升序排序（优先处理更可能相似的对）
    kept_for_llm.sort(key=lambda x: x[2])
    llm_cache: Dict[Tuple[str, str], Tuple[bool, str]] = {}  # 缓存LLM结果避免重复调用
    cnt_llm_calls = 0  # 实际LLM调用次数
    cnt_fallback = 0  # 兜底逻辑调用次数
    cnt_merged = rep_fast_merge  # 合并总数（初始为快速合并数）
    merge_decisions: List[Tuple[str, str, bool, float, str]] = []  # 判定记录

    def judge_pair(i: int, j: int, dist: float) -> Tuple[int, int, bool, float, str, bool]:
        """单对实体判定（封装为线程任务）"""
        ent_i, ent_j = ents[i], ents[j]
        # 生成缓存键（排序确保(a,b)和(b,a)为同一键）
        key = tuple(sorted((ent_i.name, ent_j.name)))
        if key in llm_cache:
            # 命中缓存
            is_same, reason = llm_cache[key]
            return (i, j, is_same, dist, reason, False)
        else:
            # 调用LLM判定
            is_same, reason, used_fallback = llm_is_same(
                ent_i, ent_j,
                truncate_desc=truncate_desc,
                desc_maxlen=desc_maxlen
            )
            llm_cache[key] = (is_same, reason)
            return (i, j, is_same, dist, reason, used_fallback)

    # 并行执行LLM判定
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futures = [ex.submit(judge_pair, i, j, dist) for (i, j, dist) in kept_for_llm]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM判定中", ncols=80):
            i, j, is_same, dist, reason, used_fallback = fut.result()
            # 更新统计
            if used_fallback:
                cnt_fallback += 1
            else:
                cnt_llm_calls += 1
            # 记录判定结果
            merge_decisions.append((ents[i].name, ents[j].name, is_same, dist, reason))
            # 合并实体（并查集）
            if is_same and dsu.union(i, j):
                cnt_merged += 1

    # ---------------- 5. 构建等价类并合并实体 ----------------
    logger.info("📘 [5/5] 构建等价类、合并并写出单一结果文件...")
    # 构建聚类（等价类）
    clusters: Dict[int, List[int]] = {}
    for x in range(len(names)):
        root = dsu.find(x)
        clusters.setdefault(root, []).append(x)

    # 辅助函数：计算描述长度（用于选择代表实体）
    def desc_len(e: EntityItem) -> int:
        return len((e.updated_description or e.original or "").strip())

    # 辅助函数：合并两个实体（保留更完整的信息）
    def merge_two(base: EntityItem, other: EntityItem) -> None:
        # 合并别名（去重，排除自身名称）
        merged_alias = list(dict.fromkeys(base.alias + other.alias + [other.name]))
        base.alias = [a for a in merged_alias if a != base.name]

        # 合并出现位置（去重）
        occ_seen = {(o.path, o.node_id) for o in base.occurrences}
        for o in other.occurrences:
            key = (o.path, o.node_id)
            if key not in occ_seen:
                base.occurrences.append(o)
                occ_seen.add(key)

        # 合并邻居（去重，排除自环）
        nb_seen = {(n.name, n.snippet) for n in base.neighbors}
        for n in other.neighbors:
            if n.name == base.name:
                continue  # 跳过自环
            key = (n.name, n.snippet)
            if key not in nb_seen:
                base.neighbors.append(n)
                nb_seen.add(key)

        # 择优保留描述（取更长的）
        if desc_len(other) > desc_len(base):
            base.updated_description = other.updated_description or other.original

    # 生成合并映射与最终实体
    rep_map: Dict[str, str] = {}  # 子实体→主实体映射
    new_entities: Dict[str, EntityItem] = {}

    for root, members in clusters.items():
        if len(members) == 1:
            # 单实体聚类，直接保留
            e = ents[members[0]]
            new_entities[e.name] = e
            continue
        # 按描述长度排序，选最长描述的实体作为代表
        members_sorted = sorted(members, key=lambda idx: desc_len(ents[idx]), reverse=True)
        rep_idx = members_sorted[0]
        rep = ents[rep_idx]
        rep_name = rep.name
        # 合并其他实体到代表实体
        for idx in members_sorted[1:]:
            child = ents[idx]
            if child.name == rep_name:
                continue
            merge_two(rep, child)
            rep_map[child.name] = rep_name
        new_entities[rep_name] = rep

    # 邻居重定向（更新为合并后的实体名称）
    redirect = rep_map.copy()
    for parent_ent in new_entities.values():
        new_neighbors: List[Neighbor] = []
        seen = set()
        for nb in parent_ent.neighbors:
            # 重定向邻居名称
            tgt = redirect.get(nb.name, nb.name)
            if tgt == parent_ent.name:
                continue  # 跳过自环
            key = (tgt, nb.snippet)
            if key in seen:
                continue  # 去重
            seen.add(key)
            new_neighbors.append(Neighbor(name=tgt, snippet=nb.snippet))
        parent_ent.neighbors = new_neighbors

    # ---------------- 输出结果 ----------------
    # 整理输出数据
    out_entities = {
        e.name: {
            "name": e.name,
            "alias": e.alias,
            "type": e.type,
            "original": e.original,
            "updated_description": e.updated_description,
            "role": e.role,
            "occurrences": [o.__dict__ for o in e.occurrences],
            "neighbors": [n.__dict__ for n in e.neighbors],
        }
        for e in new_entities.values()
    }
    # 聚类名称列表
    clusters_names = [[names[i] for i in members] for members in clusters.values()]
    # 判定记录列表
    decisions_list = [
        {
            "a": a_name,
            "b": b_name,
            "is_same": same,
            "dist": round(float(dist), 6),
            "reason": reason
        }
        for a_name, b_name, same, dist, reason in merge_decisions
    ]

    # 汇总元数据与统计信息
    t1 = time.time()
    result = {
        "meta": {
            "entities_in": os.path.abspath(ent_path),
            "emb_pkl": os.path.abspath(emb_path),
            "k_neighbors": k_neighbors,
            "dist_thresh": dist_thresh,
            "workers": workers,
            "mutual_knn": mutual_knn,
            "rank_top": rank_top,
            "per_node_cap": per_node_cap,
            "strict_same_role": strict_same_role,
            "use_name_jaccard": use_name_jaccard,
            "name_jaccard_min": name_jaccard_min,
            "name_mode": name_mode,
            "use_alias_fastpath": use_alias_fastpath,
            "truncate_desc": truncate_desc,
            "desc_maxlen": desc_maxlen,
            "api_model": getattr(APIConfig, "MODEL_NAME", "unknown"),
            "api_timeout": DedupConfig.API_TIMEOUT,
            "api_retries": DedupConfig.RETRIES,
            "duration_sec": round(t1 - t0, 2),
        },
        "summary": {
            "pairs_raw": cnt_pairs,
            "pairs_after_filter": len(candidates),
            "pairs_to_llm": len(kept_for_llm),
            "llm_calls": cnt_llm_calls,
            "fallback_calls": cnt_fallback,
            "merged_edges": cnt_merged,
            "entities_before": len(entities),
            "entities_after": len(new_entities),
        },
        "entities": out_entities,
        "clusters": clusters_names,
        "merge_map": rep_map,
        "decisions": decisions_list,
    }

    # 写入输出文件
    out_path = os.path.join(out_dir, "dedup_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 输出运行摘要
    logger.info("\n—— 去重任务摘要 ——")
    logger.info(f"候选对原始数量：{cnt_pairs}")
    logger.info(f"进入 LLM 判定的数量：{len(kept_for_llm)}（已并行处理）")
    logger.info(f"LLM 调用：{cnt_llm_calls} 次（fallback：{cnt_fallback} 次）")
    logger.info(f"成功合并的实体对：{cnt_merged}")
    logger.info(f"实体数量变化：{len(entities)} -> {len(new_entities)}")
    logger.info(f"总耗时：{t1 - t0:.2f} 秒")
    logger.info(f"结果文件：{os.path.abspath(out_path)}")


# =========================
# 命令行接口（CLI）
# =========================
def main():
    parser = argparse.ArgumentParser(description="Tree-KG 实体去重（优化·单文件输出·中文友好名称相似·含诊断）")
    # 输入输出路径
    parser.add_argument("--entities", type=str, default="", help=f"实体JSON路径（默认：{DedupConfig.FILE_ENTITIES_IN}）")
    parser.add_argument("--emb", type=str, default="", help=f"嵌入PKL路径（默认：{DedupConfig.FILE_EMBEDDINGS}）")
    parser.add_argument("--out", type=str, default=str(DedupConfig.OUTPUT_DIR), help=f"输出目录（默认：{DedupConfig.OUTPUT_DIR}）")

    # KNN参数
    parser.add_argument("--k", type=int, default=DedupConfig.KNN_NEIGHBORS, help=f"KNN近邻个数（默认：{DedupConfig.KNN_NEIGHBORS}）")
    parser.add_argument("--th", type=float, default=DedupConfig.DIST_THRESHOLD, help=f"L2距离阈值（默认：{DedupConfig.DIST_THRESHOLD}）")

    # 并发与LLM参数
    parser.add_argument("--workers", type=int, default=DedupConfig.LLM_WORKERS, help=f"LLM并发线程数（默认：{DedupConfig.LLM_WORKERS}）")

    # 优化开关
    parser.add_argument("--mutual_knn", dest="mutual_knn", action="store_true", help=f"启用互为近邻（默认：{DedupConfig.MUTUAL_KNN}）")
    parser.add_argument("--no_mutual_knn", dest="mutual_knn", action="store_false")
    parser.set_defaults(mutual_knn=DedupConfig.MUTUAL_KNN)

    parser.add_argument("--rank_top", type=int, default=DedupConfig.RANK_TOP, help=f"近邻排名上限（默认：{DedupConfig.RANK_TOP}）")
    parser.add_argument("--per_node_cap", type=int, default=DedupConfig.PER_NODE_CAP, help=f"每节点候选配额（默认：{DedupConfig.PER_NODE_CAP}）")

    parser.add_argument("--strict_same_role", dest="strict_same_role", action="store_true", help=f"严格角色匹配（默认：{DedupConfig.STRICT_SAME_ROLE}）")
    parser.add_argument("--no_strict_same_role", dest="strict_same_role", action="store_false")
    parser.set_defaults(strict_same_role=DedupConfig.STRICT_SAME_ROLE)

    parser.add_argument("--use_name_jaccard", dest="use_name_jaccard", action="store_true", help=f"启用名称Jaccard过滤（默认：{DedupConfig.USE_NAME_JACCARD}）")
    parser.add_argument("--no_use_name_jaccard", dest="use_name_jaccard", action="store_false")
    parser.set_defaults(use_name_jaccard=DedupConfig.USE_NAME_JACCARD)

    parser.add_argument("--name_jaccard_min", type=float, default=DedupConfig.NAME_JACCARD_MIN, help=f"名称Jaccard阈值（默认：{DedupConfig.NAME_JACCARD_MIN}）")
    parser.add_argument("--name_mode", type=str, default=DedupConfig.NAME_MODE, choices=["token", "char", "bigram", "auto"], help=f"名称相似度模式（默认：{DedupConfig.NAME_MODE}）")

    parser.add_argument("--use_alias_fastpath", action="store_true", default=DedupConfig.USE_ALIAS_FASTPATH, help=f"启用别名快速合并（默认：{DedupConfig.USE_ALIAS_FASTPATH}）")
    parser.add_argument("--truncate_desc", action="store_true", default=DedupConfig.TRUNCATE_DESC, help=f"启用描述截断（默认：{DedupConfig.TRUNCATE_DESC}）")
    parser.add_argument("--desc_maxlen", type=int, default=DedupConfig.DESC_MAXLEN, help=f"描述截断长度（默认：{DedupConfig.DESC_MAXLEN}）")

    args = parser.parse_args()

    # 执行去重任务
    run_dedup(
        entities_in=args.entities or None,
        emb_pkl=args.emb or None,
        out_dir=args.out,
        k_neighbors=args.k,
        dist_thresh=args.th,
        workers=args.workers,
        mutual_knn=args.mutual_knn,
        rank_top=args.rank_top,
        per_node_cap=args.per_node_cap,
        strict_same_role=args.strict_same_role,
        use_name_jaccard=args.use_name_jaccard,
        name_jaccard_min=args.name_jaccard_min,
        name_mode=args.name_mode,
        use_alias_fastpath=args.use_alias_fastpath,
        truncate_desc=args.truncate_desc,
        desc_maxlen=args.desc_maxlen,
    )


if __name__ == "__main__":
    main()