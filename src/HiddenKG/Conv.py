# -*- coding: utf-8 -*-
"""
HiddenKG 第一步：Contextual-based Convolution（conv）

输出：
  - HiddenKG/output/conv_entities.json

日志：
  - HiddenKG/logs/conv.log  （详细）
  - 终端：仅必要信息 + 进度条

依赖：
  - HiddenKG/config/config.yaml（通过 include_files 引入 config/conv.yaml）
  - 需要 config 内含:
      APIConfig: { API_BASE, API_KEY, MODEL_NAME, TIMEOUT_SECS(可选) }
      ConvConfig: 见 conv.yaml
"""

from __future__ import annotations

import json
import time
import logging
import requests
import re
import os
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from itertools import combinations
import yaml
from tqdm import tqdm
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed  # ← 新增

# ========== 配置加载 ==========
def load_config(config_file: Path) -> dict:
    with config_file.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    if "include_files" in base:
        merged = dict(base)
        for rel in base["include_files"]:
            inc_path = (config_file.parent / rel).resolve()
            with inc_path.open("r", encoding="utf-8") as ff:
                part = yaml.safe_load(ff) or {}
            merged.update(part)
        return merged
    return base

# 当前模块目录：src/HiddenKG
script_dir = Path(__file__).resolve().parent
config_file = script_dir / "config" / "config.yaml"
config = load_config(config_file)

APIConfig = config.get("APIConfig", {})
ConvConfig = config.get("ConvConfig", {})

# 小工具：从 ConvConfig 取值（带默认）
def C(key: str, default=None):
    return ConvConfig.get(key, default)

# 目录
hidden_dir = script_dir                                 # src/HiddenKG
explicit_dir = script_dir.parent / "ExplicitKG"         # src/ExplicitKG
hidden_output_dir = hidden_dir / "output"
hidden_logs_dir = hidden_dir / "logs"
explicit_output_dir = explicit_dir / "output"
hidden_output_dir.mkdir(parents=True, exist_ok=True)
hidden_logs_dir.mkdir(parents=True, exist_ok=True)

# 文件路径
FILE_TOC_ENT_REL = explicit_output_dir / C("TOC_ENT_REL_NAME", "toc_with_entities_and_relations.json")
FILE_CONV_RESULT = hidden_output_dir / C("CONV_RESULT_NAME", "conv_entities.json")
FILE_CONV_PROMPTS = hidden_output_dir / C("CONV_PROMPTS_NAME", "conv_prompts.json")  # 若后续需要可写

# ========== 日志 ==========
LOG_PATH = hidden_logs_dir / "conv.log"
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logger = logging.getLogger("Conv")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
    snippet: str = ""  # 关系描述（同小节摘要或短文本）

@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str
    original: str
    occurrences: List[Occurrence]
    neighbors: List[Neighbor]
    updated_description: str = ""

# ========== TOC 遍历与实体读取 ==========
def _iter_toc_nodes(toc: List[Dict[str, Any]], parent_path=""):
    for node in toc:
        title = node.get("title", "")
        node_id = node.get("id", "")
        level = node.get("level", -1)
        path = f"{parent_path} > {title}" if parent_path else title
        yield node, path, node_id, level
        for child in node.get("children", []) or []:
            yield from _iter_toc_nodes([child], path)

def load_entities_from_toc(file_path: Path) -> Dict[str, EntityItem]:
    """
    从 TOC 载入实体，并根据“同小节共现”统计全局权重，按权重排序写入 neighbors。
    snippet：cooccur|undirected|w=<count>|sec=<node_id>|title=<section_title>|raw=<trimmed_raw>
    """
    with file_path.open("r", encoding="utf-8") as f:
        toc = json.load(f)

    entities: Dict[str, EntityItem] = {}
    leaf_bucket: Dict[str, List[Tuple[str, str, str, str]]] = {}
    max_neighbors = int(C("MAX_NEIGHBORS_GLOBAL", 300))

    # 1) 叶小节收集实体
    for node, path, node_id, level in _iter_toc_nodes(toc):
        ents = node.get("entities") or []
        if not ents:
            continue
        sec_title = node.get("title", "")
        sec_id = node.get("id", "")
        triples: List[Tuple[str, str, str, str]] = []
        for e in ents:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            raw = (e.get("raw_content") or "").strip()
            alias = e.get("alias") or []
            etype = e.get("type") or ""
            if name not in entities:
                entities[name] = EntityItem(name, alias, etype, raw, [], [])
            entities[name].occurrences.append(Occurrence(path, node_id, level, sec_title))
            triples.append((name, raw, sec_title, sec_id))
        if triples:
            leaf_bucket.setdefault(node_id, []).extend(triples)

    # 2) 全局共现统计
    cooccur = defaultdict(int)  # key=(a,b) a<b
    best_example: Dict[Tuple[str, str], Tuple[int, str, str, str]] = {}
    for sec_id, triples in leaf_bucket.items():
        uniq_names = {}
        for n, raw, s_title, s_id in triples:
            uniq_names[n] = (raw, s_title, s_id)
        names = sorted(uniq_names.keys())
        for a, b in combinations(names, 2):
            key = (a, b) if a < b else (b, a)
            cooccur[key] += 1
        for a in names:
            for b in names:
                if a == b:
                    continue
                raw_b, s_title, s_id = uniq_names[b]
                k = (a, b)
                w = cooccur[(a, b) if a < b else (b, a)]
                old = best_example.get(k)
                if (old is None) or (w > old[0]):
                    best_example[k] = (w, s_id, s_title, raw_b)

    # 3) 构建邻居（排序截断）
    def _safe(s: str) -> str:
        return (s or "").replace("|", " ").replace("\n", " ").strip()

    for name, ent in entities.items():
        touched = set()
        for (a, b), _w in cooccur.items():
            if a == name:
                touched.add(b)
            elif b == name:
                touched.add(a)

        tmp_list = []
        for other in touched:
            key = (name, other)
            w = cooccur[(name, other) if name < other else (other, name)]
            w2, sec_id, sec_title, raw_other = best_example.get(key, (w, "", "", ""))
            w = max(w, w2)
            snippet = f"cooccur|undirected|w={w}|sec={_safe(sec_id)}|title={_safe(sec_title)}|raw={_safe(raw_other)[:80]}"
            tmp_list.append((w, other, snippet))

        tmp_list.sort(key=lambda t: (-t[0], t[1]))
        keep = tmp_list[:max_neighbors] if max_neighbors > 0 else tmp_list
        ent.neighbors = [Neighbor(name=o, snippet=snip) for _, o, snip in keep]

    # 4) 对称补齐 + 截断
    for a, ent in entities.items():
        has = {n.name for n in ent.neighbors}
        for b in list(has):
            if a not in {n.name for n in entities[b].neighbors}:
                w, sec_id, sec_title, raw_a = best_example.get((b, a), (1, "", "", ""))
                w = max(w, cooccur[(a, b) if a < b else (b, a)])
                snip = f"cooccur|undirected|w={w}|sec={_safe(sec_id)}|title={_safe(sec_title)}|raw={_safe(raw_a)[:80]}"
                entities[b].neighbors.append(Neighbor(name=a, snippet=snip))

        if max_neighbors > 0 and len(entities[a].neighbors) > max_neighbors:
            def _w(n: Neighbor) -> int:
                m = re.search(r"w=(\d+)", n.snippet)
                return -(int(m.group(1)) if m else 1)
            entities[a].neighbors.sort(key=lambda n: (_w(n), n.name))
            entities[a].neighbors = entities[a].neighbors[:max_neighbors]

    logger.info(f"[TOC] 加载实体 {len(entities)} 个；来自文件：{file_path}")
    return entities

# ========== Prompt（严格使用配置里的模板） ==========
def build_system_prompt() -> str:
    return C("PROMPT_SYSTEM", "").strip()

def _fmt_escape(s: str) -> str:
    # 防止 .format 被值里的花括号吞掉（模板本身你已用 {{ }} 处理过）
    return (s or "").replace("{", "{{").replace("}", "}}")

def build_user_prompt(entity: EntityItem) -> str:
    neighbor_lines = []
    for nb in entity.neighbors:
        snip = (nb.snippet or "").replace("\n", " ").strip()
        neighbor_lines.append(f"- {nb.name}：{snip}" if snip else f"- {nb.name}")
    neighbors_block = "\n".join(neighbor_lines) if neighbor_lines else "（无）"
    occ_str = "; ".join([o.path for o in entity.occurrences[:3]]) or "（无）"

    tpl = C("PROMPT_USER_TEMPLATE", "")
    return tpl.format(
        entity_name=_fmt_escape(entity.name),
        original_desc=_fmt_escape(entity.original or "（无描述）"),
        occurrences=_fmt_escape(occ_str),
        neighbors=_fmt_escape(neighbors_block)
    )

# ========== 限速 ==========
_rate_lock = threading.Lock()
_last_ts = 0.0

def _rate_limit_block():
    qps = float(C("RATE_LIMIT_QPS", 0))
    if qps <= 0:
        return
    min_interval = 1.0 / qps
    global _last_ts
    with _rate_lock:
        now = time.time()
        delta = now - _last_ts
        if delta < min_interval:
            time.sleep(min_interval - delta)
        _last_ts = time.time()

# ========== LLM 调用 ==========
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """调用 OpenAI 兼容 /chat/completions；支持无鉴权本地网关"""
    if bool(C("DRY_RUN", False)):
        return ""

    api_base = APIConfig.get("API_BASE", "")
    if not api_base:
        logger.warning("API_BASE 为空，跳过 LLM 调用。")
        return ""

    headers = {"Content-Type": "application/json"}
    api_key = APIConfig.get("API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": APIConfig.get("MODEL_NAME", ""),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": float(C("TEMPERATURE", 0.2)),
        "max_tokens": int(C("MAX_TOKENS", 1000))
    }

    timeout = int(C("API_TIMEOUT", 120))
    retries = int(C("RETRIES", 3))
    path = C("CHAT_COMPLETIONS_PATH", "/chat/completions")
    extra_sleep = float(C("EXTRA_THROTTLE_SEC", 0.0))
    backoff = float(C("RETRY_BACKOFF_BASE", 1.8))

    last_err: Optional[Exception] = None
    for k in range(retries):
        try:
            _rate_limit_block()
            if extra_sleep > 0:
                time.sleep(extra_sleep)

            resp = requests.post(
                f"{api_base}{path}",
                headers=headers,
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text
        except Exception as e:
            last_err = e
            logger.warning(f"LLM 调用失败({k+1}/{retries})：{e}")
            time.sleep(backoff ** k)
    logger.error(f"LLM 请求失败：{last_err}")
    return ""

# ========== 解析与清洗 ==========
def safe_json_loads(txt: str) -> Dict[str, Any]:
    if not txt or not isinstance(txt, str):
        return {}
    txt = txt.strip()
    try:
        return json.loads(txt)
    except Exception:
        pass
    candidates = re.findall(r"\{[^{}]*\}", txt)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict) and "name" in obj:
                return obj
        except Exception:
            continue
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

# ========== 主流程（并发版） ==========
def run_conv():
    if not FILE_TOC_ENT_REL.exists():
        raise FileNotFoundError(
            f"未找到 toc_with_entities_and_relations.json：{FILE_TOC_ENT_REL}\n"
            f"（文件名在 ConvConfig.TOC_ENT_REL_NAME；路径已自动拼接为 ExplicitKG/output 下）"
        )

    print(f"▶ 开始 Conv：{FILE_TOC_ENT_REL.name}")
    logger.info("开始运行 Contextual-based Convolution（conv）阶段")
    logger.info(f"输入文件: {FILE_TOC_ENT_REL}")

    # 1) 加载实体与邻居（仅同小节共现）
    entities = load_entities_from_toc(FILE_TOC_ENT_REL)

    # 2) 生成增强描述
    system_prompt = build_system_prompt()
    results: Dict[str, Any] = {}

    limit_env = os.getenv("CONV_LIMIT", "0")
    limit = int(limit_env) if limit_env.isdigit() and int(limit_env) > 0 else None
    items = list(entities.items())[:limit] if limit else list(entities.items())
    total = len(items)
    workers = int(C("WORKERS", 6))  # ← 从配置读取并发数
    start_ts = time.time()
    logger.info(f"共加载 {total} 个实体，并发 {workers}，开始调用 LLM 生成增强描述...")

    def _process(name_item: Tuple[str, EntityItem]) -> Tuple[str, Dict[str, Any]]:
        name, item = name_item
        user_prompt = build_user_prompt(item)
        content = call_llm(system_prompt, user_prompt)
        obj = safe_json_loads(content)
        upd = (obj.get("updated_description") or "").strip() or item.original or f"{name}：暂无描述。"
        upd = _enforce_len_200_300(upd)
        item.updated_description = upd
        return name, {
            "name": name,
            "alias": item.alias,
            "type": item.type,
            "original": item.original,
            "updated_description": upd,
            "occurrences": [asdict(o) for o in item.occurrences],
            "neighbors": [asdict(nb) for nb in item.neighbors]
        }

    # 并发执行 + 进度条（按完成更新）
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_process, it) for it in items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="LLM 生成增强描述", ncols=90, file=sys.stdout):
            try:
                name, res = fut.result()
                results[name] = res
                logger.info(f"[conv] {name} 生成成功。")
            except Exception as e:
                logger.warning(f"[conv] 任务失败：{type(e).__name__}: {str(e)[:200]}")

    # 3) 输出结果
    FILE_CONV_RESULT.parent.mkdir(parents=True, exist_ok=True)
    with FILE_CONV_RESULT.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    duration = round(time.time() - start_ts, 2)
    logger.info(f"Conv 阶段完成，共处理 {len(results)} 个实体。")
    logger.info(f"输出文件: {FILE_CONV_RESULT}")
    logger.info(f"总耗时: {duration} 秒")
    print(f"✅ Conv 完成：{len(results)} 个实体，并发 {workers}，耗时 {duration} s  →  {FILE_CONV_RESULT}")
    print(f"📝 详细日志：{LOG_PATH}")

# ========== CLI ==========
if __name__ == "__main__":
    run_conv()
