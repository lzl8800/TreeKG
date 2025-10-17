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
from typing import Dict, List, Tuple, Any, DefaultDict, Optional
from collections import defaultdict
import yaml
from tqdm import tqdm
import sys

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

# 目录
hidden_dir = script_dir                                 # src/HiddenKG
explicit_dir = script_dir.parent / "ExplicitKG"         # src/ExplicitKG
hidden_output_dir = hidden_dir / "output"
hidden_logs_dir = hidden_dir / "logs"
explicit_output_dir = explicit_dir / "output"
hidden_output_dir.mkdir(parents=True, exist_ok=True)
hidden_logs_dir.mkdir(parents=True, exist_ok=True)

# 文件路径
FILE_TOC_ENT_REL = explicit_output_dir / ConvConfig.get("TOC_ENT_REL_NAME", "toc_with_entities_and_relations.json")
FILE_CONV_RESULT = hidden_output_dir / ConvConfig.get("CONV_RESULT_NAME", "conv_entities.json")
FILE_CONV_PROMPTS = hidden_output_dir / ConvConfig.get("CONV_PROMPTS_NAME", "conv_prompts.json")  # 若后续需要可写

# 运行参数
TEMPERATURE = float(ConvConfig.get("TEMPERATURE", 0.2))
MAX_TOKENS = int(ConvConfig.get("MAX_TOKENS", 300))
API_TIMEOUT = int(ConvConfig.get("API_TIMEOUT", 120))
RETRIES = int(ConvConfig.get("RETRIES", 3))
CHAT_COMPLETIONS_PATH = ConvConfig.get("CHAT_COMPLETIONS_PATH", "/chat/completions")
RATE_LIMIT_QPS = float(ConvConfig.get("RATE_LIMIT_QPS", 0))
EXTRA_THROTTLE_SEC = float(ConvConfig.get("EXTRA_THROTTLE_SEC", 0.0))
RETRY_BACKOFF_BASE = float(ConvConfig.get("RETRY_BACKOFF_BASE", 1.8))
MAX_NEIGHBORS_GLOBAL = int(ConvConfig.get("MAX_NEIGHBORS_GLOBAL", 30))
DRY_RUN = bool(ConvConfig.get("DRY_RUN", False))

# ========== 日志 ==========
LOG_PATH = hidden_logs_dir / "conv.log"
# 文件日志：INFO 级别（详细）
file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

# 控制台日志：WARNING（仅必要信息；进度条由 tqdm 负责）
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

logger = logging.getLogger("Conv")
logger.setLevel(logging.INFO)
# 防止重复添加
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
else:
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
    仅从 TOC 读取实体与“同小节共现邻居”。
    """
    with file_path.open("r", encoding="utf-8") as f:
        toc = json.load(f)

    entities: Dict[str, EntityItem] = {}
    leaf_entities: Dict[str, List[Tuple[str, str]]] = {}

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

    # 邻居：同小节共现
    for node_id, name_list in leaf_entities.items():
        for name, _ in name_list:
            neighbor_map = {n.name: n for n in entities[name].neighbors}
            for n_name, n_snip in name_list:
                if n_name == name:
                    continue
                neighbor_map.setdefault(n_name, Neighbor(n_name, (n_snip or "")[:100]))
            # 限额
            entities[name].neighbors = list(neighbor_map.values())[:MAX_NEIGHBORS_GLOBAL]

    logger.info(f"[TOC] 加载实体 {len(entities)} 个；来自文件：{file_path}")
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
        neighbor_lines.append(f"- {nb.name}：{snip}" if snip else f"- {nb.name}")
    occ_str = "; ".join([o.path for o in entity.occurrences[:3]])
    return (
        f"目标实体：{entity.name}\n"
        f"原始描述：{entity.original or '（无描述）'}\n"
        f"出现位置：{occ_str or '（无）'}\n"
        f"邻域实体：\n" + "\n".join(neighbor_lines) + "\n"
        "任务：基于上下文生成增强描述，聚焦其定义、作用及与邻域的关系。\n"
        "请返回JSON：{\"name\": \"实体名\", \"updated_description\": \"增强后的描述\"}"
    )

# ========== 限速 ==========
_rate_lock = threading.Lock()
_last_ts = 0.0
_min_interval = (1.0 / RATE_LIMIT_QPS) if RATE_LIMIT_QPS > 0 else 0.0

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
    if DRY_RUN:
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
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS
    }

    last_err: Optional[Exception] = None
    for k in range(RETRIES):
        try:
            _rate_limit_block()
            if EXTRA_THROTTLE_SEC > 0:
                time.sleep(EXTRA_THROTTLE_SEC)

            resp = requests.post(
                f"{api_base}{CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            logger.warning(f"LLM 调用失败({k+1}/{RETRIES})：{e}")
            time.sleep(RETRY_BACKOFF_BASE ** k)
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

# ========== 主流程 ==========
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

    limit = int(os.getenv("CONV_LIMIT", "0")) or None
    items = list(entities.items())[:limit] if limit else list(entities.items())
    total = len(items)
    start_ts = time.time()
    logger.info(f"共加载 {total} 个实体，开始调用 LLM 生成增强描述...")

    # 进度条（只在终端展示）
    for name, item in tqdm(items, desc="LLM 生成增强描述", ncols=90, file=sys.stdout):
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
        logger.info(f"[conv] {name} 生成成功。")

    # 3) 输出结果
    FILE_CONV_RESULT.parent.mkdir(parents=True, exist_ok=True)
    with FILE_CONV_RESULT.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    duration = round(time.time() - start_ts, 2)
    logger.info(f"Conv 阶段完成，共处理 {len(results)} 个实体。")
    logger.info(f"输出文件: {FILE_CONV_RESULT}")
    logger.info(f"总耗时: {duration} 秒")
    print(f"✅ Conv 完成：{len(results)} 个实体，耗时 {duration} s  →  {FILE_CONV_RESULT}")
    print(f"📝 详细日志：{LOG_PATH}")

# ========== CLI ==========
if __name__ == "__main__":
    run_conv()
