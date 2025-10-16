# -*- coding: utf-8 -*-
"""
HiddenKG 第一步：Contextual-based Convolution（conv）

输出：
  - HiddenKG/output/conv_entities.json  （仅此一个结果文件）

日志：
  - HiddenKG/logs/conv.log  （与 Conv.py 同级的 logs 目录）

特性：
  1) 融合两类邻居：同小节共现 + 显式KG (ExplicitKG/output/explicit_kg.json)
  2) 结构化 Prompt：邻居含关系与方向，面向中文、200–300字
  3) LLM 调用仅校验 API_BASE；API_KEY 可为空（本地网关可无鉴权）
  4) 支持限速(QPS)、固定节流、指数退避重试
  5) 仅输出一个结果文件；详细日志落盘，控制台简要进度

依赖配置（均已封装在 HiddenKG/config）：
  - APIConfig：API_BASE、API_KEY、MODEL_NAME 等
  - Conv：FILE_TOC_ENT_REL、FILE_EXPLICIT_KG、FILE_CONV_RESULT、温度/超时/重试/上限等
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, DefaultDict, Optional
import requests
import os
from pathlib import Path
import shutil
from collections import defaultdict
import re
import threading

# === 导入配置（不使用别名，命名空间清晰） ===
from HiddenKG.config import APIConfig, Conv

# ========== 日志配置 ==========
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "conv.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Conv")

# ========== 数据结构 ==========
@dataclass
class Occurrence:
    path: str
    node_id: str
    level: int
    title: str

@dataclass
class Neighbor:
    name: str
    snippet: str = ""  # 关系描述（可能含关系类型与方向：type|in/out）

@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str
    original: str
    occurrences: List[Occurrence]
    neighbors: List[Neighbor]
    updated_description: str = ""

# ========== 读取显式KG邻居 ==========
def load_graph_neighbors(file_path: Path) -> Dict[str, List[Neighbor]]:
    """
    从 explicit_kg.json 构建邻接映射：
      - 对每条有向边 src->dst，生成 src: (dst, type|out) 与 dst: (src, type|in)
    """
    if not file_path.exists():
        logger.warning(f"显式KG文件缺失：{file_path}")
        return {}

    with file_path.open("r", encoding="utf-8") as f:
        kg = json.load(f)

    edges = kg.get("edges") or kg.get("E") or []
    nodes = kg.get("nodes") or kg.get("V") or []
    id2name = {n.get("id", n.get("name")): n.get("name", n.get("title")) for n in nodes}

    adj: DefaultDict[str, List[Neighbor]] = defaultdict(list)

    def _add(a, b, r, d):
        if a and b:
            adj[a].append(Neighbor(name=b, snippet=f"{r}|{d}"))

    for e in edges:
        src = id2name.get(e.get("src") or e.get("source"))
        dst = id2name.get(e.get("dst") or e.get("target"))
        r = e.get("type") or e.get("label") or "related"
        _add(src, dst, r, "out")
        _add(dst, src, r, "in")

    return adj

# ========== 进度条 ==========
def _format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")

def _render_progress(done: int, total: int, start_ts: float, desc="conv") -> None:
    cols = shutil.get_terminal_size((100, 20)).columns
    pct = 0.0 if total == 0 else done / total
    bar_len = max(10, min(40, cols - 40))
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    elapsed = time.time() - start_ts
    rate = elapsed / done if done > 0 else None
    eta = (total - done) * rate if rate else 0.0
    print(f"\r[{desc}] {done:>4d}/{total:<4d} |{bar}| {pct*100:6.2f}% "
          f"Elapsed: {_format_time(elapsed)}  ETA: {_format_time(eta)}", end="", flush=True)
    if done == total:
        print()

# ========== 收集实体 & 邻居融合 ==========
def _iter_toc_nodes(toc: List[Dict[str, Any]], parent_path=""):
    for node in toc:
        title = node.get("title", "")
        node_id = node.get("id", "")
        level = node.get("level", -1)
        path = f"{parent_path} > {title}" if parent_path else title
        yield node, path, node_id, level
        for child in node.get("children", []) or []:
            yield from _iter_toc_nodes([child], path)

def load_entities_from_toc(file_path: Path, explicit_file: Path) -> Dict[str, EntityItem]:
    """从 toc_with_entities_and_relations.json 构建实体，并融合两类邻居"""
    with file_path.open("r", encoding="utf-8") as f:
        toc = json.load(f)

    entities: Dict[str, EntityItem] = {}
    leaf_entities: Dict[str, List[Tuple[str, str]]] = {}

    # 逐小节收集实体、出现位置
    for node, path, node_id, level in _iter_toc_nodes(toc):
        ents = node.get("entities") or []
        if not ents:
            continue
        for e in ents:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            original = (e.get("raw_content") or "").strip()
            alias = e.get("alias") or []
            etype = e.get("type") or ""
            if name not in entities:
                entities[name] = EntityItem(name, alias, etype, original, [], [])
            entities[name].occurrences.append(Occurrence(path, node_id, level, node.get("title", "")))
            leaf_entities.setdefault(node_id, []).append((name, original))

    # 邻居1：同小节共现
    for node_id, name_list in leaf_entities.items():
        for name, _ in name_list:
            neighbor_map = {n.name: n for n in entities[name].neighbors}
            for n_name, n_snip in name_list:
                if n_name == name:
                    continue
                neighbor_map.setdefault(n_name, Neighbor(n_name, (n_snip or "")[:100]))
            entities[name].neighbors = list(neighbor_map.values())

    # 邻居2：显式KG
    graph_adj = load_graph_neighbors(explicit_file)
    for name, item in entities.items():
        explicit_nbs = graph_adj.get(name, [])
        neighbor_map = {n.name: n for n in item.neighbors}
        for nb in explicit_nbs:
            neighbor_map[nb.name] = nb
        # 全局邻居上限
        item.neighbors = list(neighbor_map.values())[:Conv.MAX_NEIGHBORS_GLOBAL]

    return entities

# ========== Prompt ==========
def build_system_prompt() -> str:
    return (
        "你是“MATLAB科学计算课程知识图谱专家”。"
        "请基于邻域关系增强实体描述，保持术语准确，中文输出，200–300字以内。"
    )

def build_user_prompt(entity: EntityItem) -> str:
    neighbor_lines = []
    for nb in entity.neighbors:
        snip = (nb.snippet or "").replace("\n", " ").strip()
        if "|" in snip:
            r, d, *_ = snip.split("|")
            neighbor_lines.append(f"- {nb.name}（关系: {r}, 方向: {d}）")
        elif snip:
            neighbor_lines.append(f"- {nb.name}：{snip}")
        else:
            neighbor_lines.append(f"- {nb.name}")
    occ_str = "; ".join([o.path for o in entity.occurrences[:3]])
    return (
        f"目标实体：{entity.name}\n"
        f"原始描述：{entity.original or '（无描述）'}\n"
        f"邻域实体：\n" + "\n".join(neighbor_lines) + "\n"
        "任务：基于上下文生成增强描述，聚焦其定义、作用及与邻域的关系。\n"
        "请返回JSON：{\"name\": \"实体名\", \"updated_description\": \"增强后的描述\"}"
    )

# ========== 限速 ==========
_rate_lock = threading.Lock()
_last_ts = 0.0
_min_interval = (1.0 / Conv.RATE_LIMIT_QPS) if Conv.RATE_LIMIT_QPS > 0 else 0.0

def _rate_limit_block():
    if _min_interval <= 0:
        return
    global _last_ts
    with _rate_lock:
        now = time.time()
        delta = now - _last_ts
        if delta < _min_interval:
            time.sleep(_min_interval - delta)
        _last_ts = time.time()

# ========== LLM 调用 ==========
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 OpenAI 兼容 /chat/completions；支持无鉴权本地网关"""
    if Conv.DRY_RUN:
        return ""

    if not APIConfig.API_BASE:
        logger.warning("API_BASE 为空，跳过 LLM 调用。")
        return ""

    headers = {"Content-Type": "application/json"}
    if getattr(APIConfig, "API_KEY", None):
        headers["Authorization"] = f"Bearer {APIConfig.API_KEY}"

    payload = {
        "model": APIConfig.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": Conv.TEMPERATURE,
        "max_tokens": Conv.MAX_TOKENS
    }

    last_err: Optional[Exception] = None
    for k in range(Conv.RETRIES):
        try:
            _rate_limit_block()
            if Conv.EXTRA_THROTTLE_SEC > 0:
                time.sleep(Conv.EXTRA_THROTTLE_SEC)

            resp = requests.post(
                f"{APIConfig.API_BASE}{Conv.CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=Conv.API_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            logger.warning(f"LLM 调用失败({k+1}/{Conv.RETRIES})：{e}")
            time.sleep(Conv.RETRY_BACKOFF_BASE ** k)
    logger.error(f"LLM 请求失败：{last_err}")
    return ""

# ========== 解析与清洗 ==========
def safe_json_loads(txt: str) -> Dict[str, Any]:
    """安全解析：优先直解析→平层正则→外层截取"""
    if not txt or not isinstance(txt, str):
        return {}
    txt = txt.strip()
    # 1) 直接解析
    try:
        return json.loads(txt)
    except Exception:
        pass
    # 2) 平层 JSON 对象
    candidates = re.findall(r"\{[^{}]*\}", txt)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except Exception:
            continue
    # 3) 外层截取
    start, end = txt.find("{"), txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        segment = txt[start:end + 1]
        try:
            return json.loads(segment)
        except Exception:
            cleaned = re.sub(r"^[^{]*|[^}]*$", "", segment)
            try:
                return json.loads(cleaned)
            except Exception:
                return {}
    return {}

def _enforce_len_200_300(s: str) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) > 300:
        s = s[:300]
        for c in "。；，）】":
            pos = s.rfind(c)
            if 200 <= pos <= 300:
                return s[:pos+1]
    return s

# ========== 主流程 ==========
def run_conv():
    Conv.ensure_paths()

    toc_path = Conv.FILE_TOC_ENT_REL
    explicit_path = Conv.FILE_EXPLICIT_KG

    if not toc_path.exists():
        raise FileNotFoundError(
            f"未找到 toc_with_entities_and_relations.json：{toc_path}\n"
            f"（可用环境变量 CONV_TOC_ENT_REL 覆盖路径）"
        )
    if not explicit_path.exists():
        logger.warning(f"显式KG未找到：{explicit_path}（可用 CONV_EXPLICIT_KG 覆盖）")

    logger.info("开始运行 Contextual-based Convolution（conv）阶段")
    logger.info(f"输入文件: {toc_path}")
    logger.info(f"显式KG文件: {explicit_path}")

    # 1) 加载实体与邻居
    entities = load_entities_from_toc(toc_path, explicit_path)

    # 2) 生成增强描述
    system_prompt = build_system_prompt()
    results: Dict[str, Any] = {}

    limit = int(os.getenv("CONV_LIMIT", "0")) or None
    items = list(entities.items())[:limit] if limit else list(entities.items())
    total = len(items)
    start_ts = time.time()

    logger.info(f"共加载 {total} 个实体，开始调用 LLM 生成增强描述...")

    for idx, (name, item) in enumerate(items, 1):
        _render_progress(idx - 1, total, start_ts)
        user_prompt = build_user_prompt(item)

        content = call_llm(system_prompt, user_prompt)
        obj = safe_json_loads(content)
        upd = (obj.get("updated_description") or "").strip() or item.original or f"{name}：暂无描述。"
        upd = _enforce_len_200_300(upd)
        item.updated_description = upd

        results[name] = {
            "name": name,
            "alias": item.alias,
            "type": item.type,
            "original": item.original,
            "updated_description": upd,
            "occurrences": [asdict(o) for o in item.occurrences],
            "neighbors": [asdict(nb) for nb in item.neighbors]
        }

        logger.info(f"[{idx}/{total}] {name} 生成成功。")

    # 3) 输出结果（仅一个文件）
    with Conv.FILE_CONV_RESULT.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    duration = round(time.time() - start_ts, 2)
    logger.info(f"Conv 阶段完成，共处理 {len(results)} 个实体。")
    logger.info(f"输出文件: {Conv.FILE_CONV_RESULT}")
    logger.info(f"总耗时: {duration} 秒")
    print(f"\n✅ Conv 阶段完成，日志写入: {LOG_PATH}")

# ========== CLI ==========
if __name__ == "__main__":
    run_conv()
