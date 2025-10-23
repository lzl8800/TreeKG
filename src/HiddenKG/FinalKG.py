# -*- coding: utf-8 -*-
"""
 汇总 HiddenKG/ExplicitKG 结果为最终 KG(JSON)
- 仅保留：TOC自身边、TOC→core边、core→non-core边、pred预测边
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import sys
import logging
from datetime import datetime
import yaml

# =============== 日志配置 ===============
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

# =============== 配置读取 ===============
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

# =============== 数据读取 ===============
def read_entities_map(infile: Path) -> Dict[str, dict]:
    with infile.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "entities" in data:
        return data["entities"] or {}
    return data or {}

def read_json(infile: Path) -> dict:
    with infile.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_pred_edges(infile: Path) -> List[dict]:
    with infile.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# =============== TOC工具函数 ===============
def _toc_node_name(n: dict) -> str:
    return (n.get("name") or n.get("title") or "").strip()

def _build_toc_title_index(toc_nodes: List[dict]) -> Dict[str, dict]:
    idx = {}
    for n in toc_nodes:
        key = (n.get("title") or n.get("name") or "").strip()
        if key:
            idx[key] = n
    return idx

def _edge_end_name_from_any(x: Any) -> str:
    """
    兼容两种 TOC 边端点：
    1) 字符串：直接是节点名
    2) 字典：包含 name/title 的节点对象
    其它类型：返回空串
    """
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        return (x.get("name") or x.get("title") or "").strip()
    return ""

# =============== 核心逻辑：提取节点和边 ===============
def extract_core_noncore_and_edges(
    entities: Dict[str, Dict[str, Any]], toc: dict, logger: logging.Logger
) -> Tuple[List[dict], List[dict], int, int, int, int]:
    toc_nodes = toc.get("nodes", []) or []
    toc_edges = toc.get("edges", []) or []  # TOC自身的边（如章节层级）
    toc_node_count = len(toc_nodes)
    toc_edge_count = len(toc_edges)

    toc_title_index = _build_toc_title_index(toc_nodes)

    core_nodes: List[dict] = []
    noncore_nodes: List[dict] = []
    edges: List[dict] = []

    # 1) 添加 TOC 自身的边（如"第1章→1.1节"），兼容两种格式
    edge_set = set()
    added_toc_edges = 0
    for e in toc_edges:
        # 优先读取 "source"/"target"（字符串），否则回退到 "source_node"/"target_node"（字典）
        src = _edge_end_name_from_any(e.get("source")) or _edge_end_name_from_any(e.get("source_node"))
        tgt = _edge_end_name_from_any(e.get("target")) or _edge_end_name_from_any(e.get("target_node"))

        rel = "toc→toc"  # 明确标识TOC内部边
        rtype = (e.get("relation_type") or "章节层级").strip()

        if src and tgt and src != tgt:
            key = (src, tgt, rel, rtype)
            if key not in edge_set:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relationship": rel,
                    "relation_type": rtype
                })
                edge_set.add(key)
                added_toc_edges += 1

    logger.info(f"[Assemble] TOC 自身边加入：{added_toc_edges} 条（原始 {len(toc_edges)} 条）")

    # 实体角色映射
    name2role = {k: (v.get("role") or "") for k, v in entities.items()}

    def add_edge(src: str, tgt: str, rel: str, rtype: str):
        """添加边并去重"""
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

    # 节点去重集合
    node_seen = set()

    # 2) 处理实体节点及关联边
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

        # 3) 只保留 TOC→core 边（非 core 实体不关联 TOC）
        if role == "core":
            for occ in (ent.get("occurrences") or []):
                occ_title = (occ.get("title") or "").strip()
                if not occ_title:
                    continue
                toc_node = toc_title_index.get(occ_title)
                if toc_node:
                    toc_name = _toc_node_name(toc_node)
                    if toc_name:
                        add_edge(toc_name, name, "toc→core", "章节包含核心实体")

        # 4) 只保留 core→non-core 边（过滤其他实体间边）
        for nb in (ent.get("neighbors") or []):
            tgt = (nb.get("name") or "").strip()
            snippet = (nb.get("snippet") or "").strip()
            if not tgt or tgt not in entities:
                continue
            # 仅保留 core→non-core 的层级边（根据角色和 snippet 判断）
            src_role = name2role.get(name, "")
            tgt_role = name2role.get(tgt, "")
            if src_role == "core" and tgt_role == "non-core" and "has_subordinate" in snippet:
                add_edge(name, tgt, "core→non-core", "核心实体包含非核心实体")

    core_count = len(core_nodes)
    non_core_count = len(noncore_nodes)

    return core_nodes + noncore_nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count

# =============== 合并pred边并生成最终KG ===============
def convert_to_final_kg(
    entities: Dict[str, Dict[str, Any]],
    toc: dict,
    pred_edges: List[dict],
    out_path: Path,
    logger: logging.Logger
) -> None:
    # 提取基础节点和边（含TOC自身边、TOC→core边、core→non-core边）
    nodes, edges, toc_node_count, toc_edge_count, core_count, non_core_count = extract_core_noncore_and_edges(
        entities, toc, logger
    )

    # 合并 TOC 节点（避免重复）
    existing_names = {n["name"] for n in nodes}
    final_toc_nodes: List[dict] = []
    for n in (toc.get("nodes") or []):
        nm = _toc_node_name(n)
        if nm and nm not in existing_names:
            final_toc_nodes.append({"name": nm, "type": "toc", "description": n.get("title") or nm})

    # 5) 合并 pred 预测边（仅保留两端节点存在的边，携带 llm 信息）
    node_name_set = {n["name"] for n in nodes} | {n["name"] for n in final_toc_nodes}
    edge_set = {(e["source"], e["target"], e["relationship"], e["relation_type"]) for e in edges}

    def add_pred_edge(src: str, tgt: str, relationship: str, relation_type: str, description: str):
        # 用包含 description 的 key 去重，避免相同类型不同描述被错误合并
        key = (src, tgt, relationship, relation_type, description[:100])
        if src and tgt and src != tgt and key not in edge_set:
            edges.append({
                "source": src,
                "target": tgt,
                "relationship": relationship,
                "relation_type": relation_type,
                "description": description
            })
            edge_set.add(key)

    for e in pred_edges or []:
        u = (e.get("u") or "").strip()
        v = (e.get("v") or "").strip()
        if not u or not v:
            continue
        if u not in node_name_set or v not in node_name_set:
            logger.debug(f"[Pred] 跳过边（节点缺失）: {u} - {v}")
            continue
        llm_info = e.get("llm", {}) or {}
        rel_type = (llm_info.get("type") or "未分类").strip()
        rel_desc = (llm_info.get("description") or f"{u}与{v}存在关联").strip()
        add_pred_edge(u, v, "pred", rel_type, rel_desc)

    # 生成最终 KG
    kg = {"nodes": nodes + final_toc_nodes, "edges": edges}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    # 输出统计信息
    print(f"✅ 最终知识图谱：{out_path.resolve()}")
    print(f"- TOC 节点：{toc_node_count}，TOC 自身边（原始）{toc_edge_count}")
    print(f"- 核心节点：{core_count}，非核心节点：{non_core_count}")
    print(f"- 总节点：{len(kg['nodes'])}，总边：{len(kg['edges'])}")
    logger.info(
        "边类型分布："
        f"TOC→TOC {sum(1 for e in edges if e['relationship']=='toc→toc')} 条，"
        f"TOC→core {sum(1 for e in edges if e['relationship']=='toc→core')} 条，"
        f"core→non-core {sum(1 for e in edges if e['relationship']=='core→non-core')} 条，"
        f"pred {sum(1 for e in edges if e['relationship']=='pred')} 条"
    )

# =============== 主入口 ===============
def main():
    script_dir = Path(__file__).resolve().parent
    logger = setup_logging("Assemble", script_dir / "logs")

    # 读取配置
    hidden_cfg_path = script_dir / "config" / "config.yaml"
    hcfg = load_yaml_with_includes(hidden_cfg_path)
    explicit_cfg_path = script_dir.parent / "ExplicitKG" / "config" / "config.yaml"
    ecfg = load_yaml_with_includes(explicit_cfg_path)

    # 解析输出目录
    def _resolve_out_dir(cfg_value: str, fallback: Path) -> Path:
        v = (cfg_value or "").strip()
        if (not v) or ("path/to" in v) or (not Path(v).exists()):
            return fallback.resolve()
        return Path(v).resolve()
    hidden_out_dir = _resolve_out_dir(hcfg.get("OUTPUT_DIR"), script_dir / "output")
    explicit_out_dir = _resolve_out_dir(ecfg.get("OUTPUT_DIR"), script_dir.parent / "ExplicitKG" / "output")

    # 解析文件路径
    dedup_name = (hcfg.get("DedupConfig", {}) or {}).get("RESULT_NAME", "dedup_result.json")
    pred_name = (hcfg.get("PredConfig", {}) or {}).get("RESULT_NAME", "pred_result.json")
    toc_graph_name = ecfg.get("TOC_GRAPH_NAME", "toc_graph.json")
    dedup_path = hidden_out_dir / dedup_name
    pred_path = hidden_out_dir / pred_name
    toc_path = explicit_out_dir / toc_graph_name
    final_path = hidden_out_dir / "final_kg.json"

    # 校验输入文件
    if not dedup_path.exists():
        raise FileNotFoundError(f"未找到去重文件：{dedup_path}")
    if not toc_path.exists():
        raise FileNotFoundError(f"未找到 TOC 图文件：{toc_path}")
    if not pred_path.exists():
        logger.warning(f"未找到 Pred 结果：{pred_path}（将不合入预测边）")

    # 加载数据并生成KG
    entities_map = read_entities_map(dedup_path)
    toc_data = read_json(toc_path)
    pred_edges = read_pred_edges(pred_path) if pred_path.exists() else []
    logger.info(f"载入：entities={len(entities_map)}  toc_nodes={len(toc_data.get('nodes', []))}  pred_edges={len(pred_edges)}")
    convert_to_final_kg(entities_map, toc_data, pred_edges, final_path, logger)

if __name__ == "__main__":
    main()
