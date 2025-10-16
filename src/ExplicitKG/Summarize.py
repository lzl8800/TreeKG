# -*- coding: utf-8 -*-
"""
Summarize.py
Tree-KG 阶段一·步骤2：自底向上摘要（.docx 源）
- 读取 toc_structure.json（含每个节点的 level、id、title、children）
- 在 .docx 中定位每个节点的段落范围 para_start/para_end（稳健匹配）
- 叶子节点：抽正文 -> LLM 并发摘要
- 非叶节点：并发聚合子摘要 -> 生成上层摘要
- 分层自底向上，同层并发；带重试与指数退避
- 输出 toc_with_summaries.json
"""

import argparse
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from tqdm import tqdm
import openai

# ===== 配置 =====
try:
    from config import APIConfig, SummarizeConfig
except Exception as e:
    raise RuntimeError(f"无法导入配置：{e}")

# ===== 日志 =====
logger = logging.getLogger("Summarize")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ===== OpenAI 兼容接口 =====
openai.api_base = APIConfig.API_BASE
openai.api_key = APIConfig.API_KEY

# ===== 工具函数 =====
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()

def make_heading_string(level: int, node_id: str, title: str) -> str:
    """生成用于 .docx 中检索的标题字符串"""
    title = normalize_text(title)
    if level == 1:
        num = re.sub(r"章$", "", node_id)
        return f"第{num}章 {title}"
    else:
        return f"{node_id} {title}"

def load_docx_paragraphs(docx_path: Path) -> List[str]:
    doc = Document(str(docx_path))
    return [normalize_text(p.text or "") for p in doc.paragraphs]

def find_heading_para(paras: List[str], heading: str) -> Optional[int]:
    """优先包含匹配；退化为编号前缀匹配"""
    h = normalize_text(heading)
    for i, t in enumerate(paras):
        if h and h in t:
            return i
    m = re.match(r"^第(\d+)章", h)
    only = (f"第{m.group(1)}章" if m else (h.split(" ", 1)[0] if " " in h else h))
    for i, t in enumerate(paras):
        if only and only in t:
            return i
    return None

def flatten_nodes(toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def dfs(n: Dict[str, Any]):
        out.append(n)
        for c in n.get("children", []):
            dfs(c)
    for r in toc:
        dfs(r)
    return out

def attach_para_ranges(toc: List[Dict[str, Any]], paras: List[str]) -> None:
    ordered = flatten_nodes(toc)
    idx_map: Dict[int, Optional[int]] = {}
    for node in ordered:
        lvl = node.get("level", 0)
        h = make_heading_string(lvl, node["id"], node["title"])
        idx_map[id(node)] = find_heading_para(paras, h)

    for i, node in enumerate(ordered):
        start = idx_map[id(node)]
        if start is None or start < 0:
            node["para_start"] = None
            node["para_end"] = None
            continue
        end_idx = len(paras) - 1
        cur_level = node.get("level", 0)
        for j in range(i + 1, len(ordered)):
            nxt = ordered[j]
            nxt_start = idx_map[id(nxt)]
            if nxt_start is not None and nxt_start >= 0 and nxt.get("level", 0) <= cur_level:
                end_idx = max(start, nxt_start - 1)
                break
        node["para_start"] = start
        node["para_end"] = end_idx

def extract_para_span(paras: List[str], start: int, end: int) -> str:
    if start is None or end is None or start < 0 or end < 0 or end < start:
        return ""
    body = paras[start + 1 : end + 1]   # 跳过标题行
    text = "\n".join(t for t in body if t)
    return text.strip()

def split_chunks(text: str, max_len: int, overlap: int) -> List[str]:
    t = text.strip()
    if not t:
        return []
    if len(t) <= max_len:
        return [t]
    chunks, i = [], 0
    while i < len(t):
        chunks.append(t[i:i+max_len])
        if i + max_len >= len(t):
            break
        i = i + max_len - overlap
        if i < 0: i = 0
    return chunks

# ===== LLM 请求（重试+退避）=====
def _chat_once(prompt: str) -> str:
    resp = openai.ChatCompletion.create(
        model=APIConfig.MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=SummarizeConfig.TEMPERATURE,
        timeout=SummarizeConfig.REQUEST_TIMEOUT,
    )
    return resp["choices"][0]["message"]["content"].strip()

def chat_with_retry(prompt: str) -> str:
    last_err = None
    for k in range(SummarizeConfig.RETRY_ATTEMPTS):
        try:
            return _chat_once(prompt)
        except Exception as e:
            last_err = e
            backoff = (SummarizeConfig.RETRY_BACKOFF_BASE ** k)
            time.sleep(backoff)
    raise RuntimeError(f"LLM 请求失败（已重试 {SummarizeConfig.RETRY_ATTEMPTS} 次）: {last_err}")

def summarize_leaf_text(raw_text: str) -> str:
    if not raw_text.strip():
        return ""
    parts = split_chunks(raw_text, SummarizeConfig.MAX_CHARS, SummarizeConfig.CHUNK_OVERLAP)
    summaries: List[str] = []
    for ck in parts:
        prompt = SummarizeConfig.LEAF_PROMPT.format(
            target_len=SummarizeConfig.TARGET_SUMMARY_LEN, content=ck
        )
        summaries.append(chat_with_retry(prompt))
    if len(summaries) == 1:
        return summaries[0]
    merged = "\n\n".join(summaries)
    prompt = SummarizeConfig.AGG_PROMPT.format(
        target_len=SummarizeConfig.TARGET_SUMMARY_LEN, content=merged
    )
    return chat_with_retry(prompt)

def aggregate_children(children: List[Dict[str, Any]]) -> str:
    pieces = []
    for ch in children:
        s = (ch.get("summary") or "").strip()
        if s:
            pieces.append(f"[{ch.get('id','')}] {s}")
    if not pieces:
        return ""
    content = "\n\n".join(pieces)
    prompt = SummarizeConfig.AGG_PROMPT.format(
        target_len=SummarizeConfig.TARGET_SUMMARY_LEN, content=content
    )
    return chat_with_retry(prompt)

def is_leaf(node: Dict[str, Any]) -> bool:
    return not node.get("children")

# ===== 分层并发：自底向上，同层并发 =====
def compute_depths(toc: List[Dict[str, Any]]) -> int:
    """为每个节点打上 depth 字段（根为1），返回最大深度"""
    max_depth = 1
    def dfs(n: Dict[str, Any], d: int):
        nonlocal max_depth
        n["_depth"] = d
        max_depth = max(max_depth, d)
        for c in n.get("children", []):
            dfs(c, d + 1)
    for r in toc:
        dfs(r, 1)
    return max_depth

def nodes_at_depth(toc: List[Dict[str, Any]], d: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    def dfs(n: Dict[str, Any]):
        if n.get("_depth") == d:
            out.append(n)
        for c in n.get("children", []):
            dfs(c)
    for r in toc:
        dfs(r)
    return out

def run(docx: Path, toc_json: Path, out_json: Path) -> None:
    if not docx.exists():
        raise FileNotFoundError(f"未找到 .docx：{docx}")
    if not toc_json.exists():
        raise FileNotFoundError(f"未找到 TOC：{toc_json}，请先运行 TextSegmentation.py")

    with toc_json.open("r", encoding=SummarizeConfig.ENCODING) as f:
        toc: List[Dict[str, Any]] = json.load(f)

    paras = load_docx_paragraphs(docx)
    attach_para_ranges(toc, paras)

    # 打深度标签
    max_d = compute_depths(toc)

    # 统计总节点数用于进度条
    total_nodes = len(flatten_nodes(toc))
    pbar = tqdm(total=total_nodes, desc="生成摘要（分层并发）")

    def summarize_node(n: Dict[str, Any]):
        """单节点摘要任务：叶子=正文摘要；非叶=聚合子摘要"""
        if is_leaf(n):
            raw = extract_para_span(paras, n.get("para_start"), n.get("para_end"))
            n["summary"] = summarize_leaf_text(raw)
        else:
            n["summary"] = aggregate_children(n.get("children", []))
        pbar.update(1)

    # 自底向上：从最大深度到 1，每层并发执行
    with ThreadPoolExecutor(max_workers=SummarizeConfig.MAX_WORKERS) as pool:
        for d in range(max_d, 0, -1):
            layer_nodes = nodes_at_depth(toc, d)
            # 提交本层所有节点任务
            futures = [pool.submit(summarize_node, n) for n in layer_nodes]
            # 等待本层全部完成，再进入上一层（保证父节点读取到子摘要）
            for fu in as_completed(futures):
                fu.result()

    pbar.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding=SummarizeConfig.ENCODING) as f:
        json.dump(toc, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ 已写出：{out_json.resolve()}")
    logger.info("每个节点包含：id/title/children/para_start/para_end/summary")

def main():
    ap = argparse.ArgumentParser(description="自底向上并发摘要（.docx 源）")
    ap.add_argument("--docx", type=str, default=str(SummarizeConfig.DOCX_PATH),
                    help="输入 .docx 路径")
    ap.add_argument("--toc", type=str, default=str(SummarizeConfig.TOC_JSON_PATH),
                    help="TOC JSON 路径")
    ap.add_argument("--out", type=str, default=str(SummarizeConfig.OUT_JSON_PATH),
                    help="输出 JSON 路径")
    ap.add_argument("--loglevel", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()
    logger.setLevel(getattr(logging, args.loglevel))
    run(Path(args.docx), Path(args.toc), Path(args.out))

if __name__ == "__main__":
    main()
