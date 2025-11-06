from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any
import json
from collections import Counter
import re

# === 目录锚点 ===
HERE = Path(__file__).resolve().parent      # .../src/HiddenKG
SRC  = HERE.parent                           # .../src
ROOT = SRC.parent                            # .../TreeKG (如需)

# === 固定输入/输出目录（都在 src/*/output 下） ===
EXPLICIT_OUT = SRC / "ExplicitKG" / "output"        # 输入：toc_graph.json
HIDDEN_OUT   = HERE / "output"                      # 输入：dedup_result.json / pred_result.json；输出：final_kg.json

DEDUP_FILE = HIDDEN_OUT / "dedup_result.json"
PRED_FILE  = HIDDEN_OUT / "pred_result.json"
TOC_FILE   = EXPLICIT_OUT / "toc_graph.json"
FINAL_FILE = HIDDEN_OUT / "final_kg.json"           # 输出：src/HiddenKG/output/final_kg.json

REL7 = {
    "prerequisite","part-of","applies-to","example-of","synonym","contrasts-with","related"
}
_re_type = re.compile(r"type=([a-z\-]+)", re.I)

def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_entities_map(p: Path) -> Dict[str, dict]:
    data = read_json(p)
    return data.get("entities", {}) if isinstance(data, dict) else {}

def build_toc_index(toc_nodes: List[dict]) -> Dict[str, dict]:
    idx = {}
    for n in toc_nodes:
        title = (n.get("title") or n.get("name") or "").strip()
        if title:
            idx[title] = n
    return idx

def _infer_nb_rel_type(nb: dict) -> str | None:
    """
    从邻居项推断 core->noncore 的关系类型：
    1) 优先 nb['type']（Aggr/Dedup 已写入的标准标签）
    2) 无则从 snippet 中解析 rel|...|type=xxx
    3) 若含 has_subordinate → 映射为 part-of
    4) 其他返回 None（不写边）
    """
    t = (nb.get("type") or "").strip().lower()
    if t in REL7:
        return t

    sn = (nb.get("snippet") or "").strip().lower()
    m = _re_type.search(sn)
    if m:
        cand = m.group(1).strip().lower()
        if cand in REL7:
            return cand

    if "has_subordinate" in sn:
        return "part-of"

    return None

def assemble():
    # 输入检查（都在 src/*/output 下）
    if not DEDUP_FILE.exists():
        raise FileNotFoundError(f"missing {DEDUP_FILE}")
    if not TOC_FILE.exists():
        raise FileNotFoundError(f"missing {TOC_FILE}")

    entities = load_entities_map(DEDUP_FILE)
    toc = read_json(TOC_FILE)
    toc_nodes = toc.get("nodes", []) or []
    toc_edges = toc.get("edges", []) or []
    pred_edges = read_json(PRED_FILE) if PRED_FILE.exists() else []

    toc_title_index = build_toc_index(toc_nodes)

    nodes: List[dict] = []
    edges: List[dict] = []

    # 1) TOC 节点 —— level 标注为 "toc"
    final_toc_nodes: List[dict] = []
    for n in toc_nodes:
        title = (n.get("title") or n.get("name") or "").strip()
        if not title:
            continue
        final_toc_nodes.append({
            "name": title,
            "type": "toc",
            "description": title,
            "level": "toc"
        })

    # 2) 实体节点 —— level 标注为 "core" / "noncore"
    role_map: Dict[str, str] = {}
    for name, ent in entities.items():
        role_raw = (ent.get("role") or "").strip().lower()
        # 统一成 core / noncore 两类
        if role_raw.replace("-", "") == "noncore":
            role_norm = "noncore"
        elif role_raw == "core":
            role_norm = "core"
        else:
            # 未知角色的，统一当作 noncore；也可根据你数据实际改为 core
            role_norm = "noncore"
        role_map[name] = role_norm

        nodes.append({
            "name": name,
            "type": ent.get("type", ""),
            "description": (ent.get("original") or "").strip(),
            "level": role_norm  # <<<<<< 使用 "core" / "noncore"
        })

    # 3) TOC 自身边（toc->toc）
    edge_set = set()
    def add_edge(src: str, tgt: str, typ: str, desc: str | None = None):
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, typ, desc or "")
        if key in edge_set:
            return
        item = {"source": src, "target": tgt, "type": typ}
        if desc:
            item["description"] = desc
        edges.append(item)
        edge_set.add(key)

    for e in toc_edges:
        s = (e.get("source") or "").strip()
        t = (e.get("target") or "").strip()
        if s and t:
            add_edge(s, t, "toc->toc")

    # 4) TOC→core（toc->core）：occurrences 里的 title 连接，仅连到 core
    for name, ent in entities.items():
        if role_map.get(name) != "core":
            continue
        for occ in (ent.get("occurrences") or []):
            title = (occ.get("title") or "").strip()
            if title and title in toc_title_index:
                add_edge(title, name, "toc->core")

    # 5) core→noncore：从 neighbors 读取“类型”（优先 nb.type → snippet 解析 → has_subordinate→part-of）
    for pname, pent in entities.items():
        if role_map.get(pname) != "core":
            continue
        for nb in (pent.get("neighbors") or []):
            cname = (nb.get("name") or "").strip()
            if not cname or cname not in entities:
                continue
            if role_map.get(cname) != "noncore":
                continue
            rel_type = _infer_nb_rel_type(nb)
            if not rel_type:
                continue
            add_edge(pname, cname, rel_type, desc="core→non-core")

    # 6) 合并 pred 边（type 使用 llm.type）
    all_node_names = {n["name"] for n in nodes} | {n["name"] for n in final_toc_nodes}
    for e in pred_edges or []:
        u = (e.get("u") or "").strip()
        v = (e.get("v") or "").strip()
        if not u or not v:
            continue
        if u not in all_node_names or v not in all_node_names:
            continue
        llm = e.get("llm", {}) or {}
        typ = (llm.get("type") or "pred").strip()
        desc = (llm.get("description") or f"{u}与{v}存在关联").strip()
        add_edge(u, v, typ, desc)

    # 7) 写出
    kg = {"nodes": nodes + final_toc_nodes, "edges": edges}
    FINAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_FILE.open("w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False, indent=2)

    # 统计
    print(f"✅ 输出: {FINAL_FILE.resolve()}")
    print(f"- TOC 节点: {len(toc_nodes)}")
    core_cnt = sum(1 for _, e in entities.items() if (e.get('role') or '').strip().lower() == 'core')
    noncore_cnt = sum(1 for _, e in entities.items() if (e.get('role') or '').strip().lower().replace('-', '') == 'noncore')
    print(f"- 核心节点: {core_cnt}，非核心节点: {noncore_cnt}")
    print(f"- 总节点: {len(kg['nodes'])}，总边: {len(kg['edges'])}")
    by_type = Counter(e["type"] for e in edges)
    for k, v in by_type.items():
        print(f"  · {k}: {v}")

if __name__ == "__main__":
    assemble()
