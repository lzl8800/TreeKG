# -*- coding: utf-8 -*-
"""
Assemble.py — 汇总 HiddenKG/ExplicitKG 结果为最终 KG(JSON)
- 读取 HiddenKG/output/{dedup_result.json, pred_result.json}
- 读取 ExplicitKG/output/{toc_graph.json}
- 输出 HiddenKG/output/final_kg.json
- 终端只打印必要信息；详细日志写入 HiddenKG/log/assemble-*.log
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import sys
import logging
from datetime import datetime
import yaml

# =============== 日志：终端精简 + 文件详细 ===============
def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"{name.lower()}-{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"))

    logger.addHandler(sh)
    logger.addHandler(fh)

    for noisy in ("urllib3", "requests", "transformers", "openai", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug(f"log file -> {log_path}")
    return logger

# =============== 配置读取（支持 include_files） ===============
def load_yaml_with_includes(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    merged = dict(base)
    incs = base.get("include_files") or []
    for rel in incs:
        inc_path = (cfg_path.parent / rel).resolve()
        if not inc_path.exists():
            raise FileNotFoundError(f"未找到 include 文件：{inc_path}")
        with inc_path.open("r", encoding="utf-8") as ff:
            sub = yaml.safe_load(ff) or {}
        merged.update(sub)
    return merged

# =============== 读取 / 解析 ===============
def read_entities_map(infile: Path) -> Dict[str, dict]:
    with infile.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # 支持两种：dedup_result.json 的 {"entities": {...}} 或 conv/aggr 的平面
    if isinstance(data, dict) and "entities" in data:
        return data["entities"] or {}
    return data or {}

def read_json(infile: Path) -> dict:
    with infile.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_pred_edges(infile: Path) -> List[dict]:
    with infile.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("edges", [])

# =============== TOC 工具 ===============
def _toc_node_name(n: dict) -> str:
    # 兼容 'name' 或 'title'
    return (n.get("name") or n.get("title") or "").strip()

def _build_toc_title_index(toc_nodes: List[dict]) -> Dict[str, dict]:
    # 用 title 优先，其次 name，构建查找表
    idx = {}
    for n in toc_nodes:
        key = (n.get("title") or n.get("name") or "").strip()
        if key:
            idx[key] = n
    return idx

# =============== 主处理逻辑 ===============
def extract_core_noncore_and_edges(
    entities: Dict[str, Dict[str, Any]], toc: dict, logger: logging.Logger
) -> Tuple[List[dict], List[dict], int, int, int, int]:
    toc_nodes = toc.get("nodes", []) or []
    toc_edges = toc.get("edges", []) or []
    toc_node_count = len(toc_nodes)
    toc_edge_count = len(toc_edges)

    toc_title_index = _build_toc_title_index(toc_nodes)

    core_nodes: List[dict] = []
    noncore_nodes: List[dict] = []
    edges: List[dict] = []

    # 用集合去重边 (source, target, relationship, relation_type)
    edge_set = set()

    # 实体名 -> 角色
    name2role = {k: (v.get("role") or "") for k, v in entities.items()}

    def add_edge(src: str, tgt: str, rel: str, rtype: str):
        key = (src, tgt, rel, rtype)
        if not src or not tgt or src == tgt:
            return
        if key in edge_set:
            return
        edges.append({
            "source": src,
            "target": tgt,
            "relationship": rel,
            "relation_type": rtype
        })
        edge_set.add(key)

    # 节点集合（用 name 去重）
    node_seen = set()

    # 组装实体节点
    for name, ent in entities.items():
        role = ent.get("role") or ""
        node = {
            "name": name,
            "type": ent.get("type", ""),
            "description": (ent.get("updated_description") or ent.get("original") or "").strip()
        }
        if name in node_seen:
            continue
        node_seen.add(name)
        if role == "core":
            core_nodes.append(node)
        else:
            noncore_nodes.append(node)

        # 连接 TOC：按 occurrences.title 匹配到 TOC 节点 title/name
        for occ in (ent.get("occurrences") or []):
            occ_title = (occ.get("title") or "").strip()
            if not occ_title:
                continue
            toc_node = toc_title_index.get(occ_title)
            if toc_node:
                toc_name = _toc_node_name(toc_node)
                if toc_name:
                    add_edge(toc_name, name, "toc-core", "所属小节")

        # 实体-实体边：基于 neighbors（只在两端实体都存在时生成）
        for nb in (ent.get("neighbors") or []):
            tgt = (nb.get("name") or "").strip()
            snippet = (nb.get("snippet") or "").strip()
            if not tgt or tgt not in entities:
                continue
            rel_type = snippet.split("|")[0] if "|" in snippet else (snippet or "related")
            # 根据对端是否 core，给关系大类：core-core / core-non-core / non-core-non-core
            tgt_role = name2role.get(tgt, "")
            if role == "core" and tgt_role == "core":
                rel = "core-core"
            elif role == "core" and tgt_role != "core":
                rel = "core-non-core"
            elif role != "core" and tgt_role == "core":
                rel = "non-core-core"
            else:
                rel = "non-core-non-core"
            add_edge(name, tgt, rel, rel_type)

    core_count = len(core_nodes)
    non_core_count = len(noncore_nodes)

    return core_nodes + noncore_nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count

def convert_to_final_kg(
    entities: Dict[str, Dict[str, Any]],
    toc: dict,
    pred_edges: List[dict],
    out_path: Path,
    logger: logging.Logger
) -> None:
    # 先从实体+TOC 生成基础节点/边
    nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count = extract_core_noncore_and_edges(
        entities, toc, logger
    )

    # 合并 TOC 节点（避免重复，按 name/title 去重）
    existing_names = {n["name"] for n in nodes}
    final_toc_nodes: List[dict] = []
    for n in (toc.get("nodes") or []):
        nm = _toc_node_name(n)
        if nm and nm not in existing_names:
            final_toc_nodes.append({"name": nm, "type": "toc", "description": n.get("title") or nm})

    # 合并 Pred 结果边（仅当两端在节点集中）
    node_name_set = {n["name"] for n in nodes} | {n["name"] for n in final_toc_nodes}
    edge_set = {(e["source"], e["target"], e["relationship"], e["relation_type"]) for e in edges}

    def add_edge(src: str, tgt: str, relationship: str, relation_type: str):
        key = (src, tgt, relationship, relation_type)
        if src and tgt and src != tgt and key not in edge_set:
            edges.append({"source": src, "target": tgt, "relationship": relationship, "relation_type": relation_type})
            edge_set.add(key)

    for e in pred_edges or []:
        u = (e.get("u") or "").strip()
        v = (e.get("v") or "").strip()
        if not u or not v:
            continue
        if u not in node_name_set or v not in node_name_set:
            # 只提示一次精简信息；细节进日志
            logger.debug(f"[Pred] 跳过边（节点缺失）: {u} - {v}")
            continue
        relationship = (e.get("llm", {}) or {}).get("type", "").strip() or "predicted"
        description = (e.get("llm", {}) or {}).get("description", "").strip()
        add_edge(u, v, "pred", relationship or "predicted")
        # 可选：把描述放进 relation_type 或另立字段，这里放 relation_type（你原逻辑）
        if description:
            edges[-1]["relation_type"] = description

    # 组装输出
    kg = {"nodes": nodes + final_toc_nodes, "edges": edges}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    # 终端必要信息
    print(f"✅ 最终知识图谱：{out_path.resolve()}")
    print(f"- TOC 节点：{toc_node_count}，TOC 边：{toc_edge_count}")
    print(f"- 核心节点：{core_count}，非核心节点：{non_core_count}")
    print(f"- 总节点：{len(kg['nodes'])}，总边：{len(kg['edges'])}")

    # 详细写入日志
    logger.info(f"写出：{out_path.resolve()}")
    logger.debug(f"节点示例（前3）：{kg['nodes'][:3]}")
    logger.debug(f"边示例（前3）：{kg['edges'][:3]}")

# =============== 主入口 ===============
def main():
    script_dir = Path(__file__).resolve().parent
    logger = setup_logging("Assemble", script_dir / "log")

    # 读取 HiddenKG/config/config.yaml（合并 include）
    hidden_cfg_path = script_dir / "config" / "config.yaml"
    hcfg = load_yaml_with_includes(hidden_cfg_path)

    # 读取 ExplicitKG/config/config.yaml（合并 include）
    explicit_cfg_path = script_dir.parent / "ExplicitKG" / "config" / "config.yaml"
    ecfg = load_yaml_with_includes(explicit_cfg_path)

    # 目录
    def _resolve_out_dir(cfg_value: str, fallback: Path) -> Path:
        v = (cfg_value or "").strip()
        # 占位符或目录不存在，就用兜底
        if (not v) or ("path/to" in v) or (not Path(v).exists()):
            return fallback.resolve()
        return Path(v).resolve()

    hidden_out_dir = _resolve_out_dir(hcfg.get("OUTPUT_DIR"), script_dir / "output")
    explicit_out_dir = _resolve_out_dir(ecfg.get("OUTPUT_DIR"), script_dir.parent / "ExplicitKG" / "output")
    logger.debug(f"HiddenKG/output = {hidden_out_dir}")
    logger.debug(f"ExplicitKG/output = {explicit_out_dir}")

    # 文件名（只保留 NAME，在此拼目录）
    dedup_name = (hcfg.get("DedupConfig", {}) or {}).get("RESULT_NAME", "dedup_result.json")
    pred_name = (hcfg.get("PredConfig", {}) or {}).get("RESULT_NAME", "pred_result.json")
    # TOC 图文件名（Explicit 侧生成的图），常见：toc_graph.json
    toc_graph_name = ecfg.get("TOC_GRAPH_NAME", "toc_graph.json")

    dedup_path = hidden_out_dir / dedup_name
    pred_path = hidden_out_dir / pred_name
    toc_path = explicit_out_dir / toc_graph_name
    final_path = hidden_out_dir / "final_kg.json"  # 最终只写一个

    # 加载数据
    if not dedup_path.exists():
        raise FileNotFoundError(f"未找到去重文件：{dedup_path}")
    if not toc_path.exists():
        raise FileNotFoundError(f"未找到 TOC 图文件：{toc_path}")
    if not pred_path.exists():
        # 没有 pred 也能产出 KG，只是少了预测边
        logger.warning(f"未找到 Pred 结果：{pred_path}（将不合入预测边）")

    entities_map = read_entities_map(dedup_path)
    toc_data = read_json(toc_path)
    pred_edges = read_pred_edges(pred_path) if pred_path.exists() else []

    logger.info(f"载入：entities={len(entities_map)}  toc_nodes={len(toc_data.get('nodes', []))}  pred_edges={len(pred_edges)}")

    convert_to_final_kg(entities_map, toc_data, pred_edges, final_path, logger)

if __name__ == "__main__":
    main()
