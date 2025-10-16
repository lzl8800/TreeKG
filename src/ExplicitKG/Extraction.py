# -*- coding: utf-8 -*-
"""
Extraction.py
Tree-KG 阶段二·步骤1：从章节摘要中提取 Physics 实体与关系
- 读取 toc_with_summaries.json（可在 config/extraction.py 中重命名）
- 并发：对每个小节摘要调用 LLM 提取实体与关系（实体→关系串行，节点之间并行）
- 稳健：重试 + 指数退避 +（可选）QPS 限速 + 进度条
- 输出 toc_with_entities_and_relations.json（可在配置中修改）
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from tqdm import tqdm

# ===== 读取配置 =====
try:
    from config import APIConfig
    from config.extract import ExtractionConfig
except Exception as e:
    raise RuntimeError(f"无法导入配置：{e}")

# ===== OpenAI 兼容接口 =====
openai.api_base = APIConfig.API_BASE
openai.api_key = APIConfig.API_KEY
MODEL_NAME = APIConfig.MODEL_NAME

# ===== 速率限制（可选）=====
_rate_lock = threading.Lock()
_last_ts = 0.0
_min_interval = (1.0 / ExtractionConfig.RATE_LIMIT_QPS) if ExtractionConfig.RATE_LIMIT_QPS > 0 else 0.0

def _rate_limit_block():
    """简单全局限速：保证两次请求间隔 >= 1/QPS"""
    if _min_interval <= 0:
        return
    global _last_ts
    with _rate_lock:
        now = time.time()
        delta = now - _last_ts
        if delta < _min_interval:
            time.sleep(_min_interval - delta)
        _last_ts = time.time()

# ===== 工具函数 =====
def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def _strip_code_fences(s: str) -> str:
    # 兼容模型偶尔输出 ```json ... ```
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def safe_json_loads(content: str) -> Dict:
    try:
        content = _strip_code_fences(content)
        return json.loads(content)
    except Exception:
        return {}

def _chat_once(prompt: str) -> str:
    _rate_limit_block()                                # QPS 控制（可关）
    if ExtractionConfig.EXTRA_THROTTLE_SEC > 0:       # 额外固定节流（可关）
        time.sleep(ExtractionConfig.EXTRA_THROTTLE_SEC)
    resp = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=ExtractionConfig.TEMPERATURE,
        timeout=ExtractionConfig.REQUEST_TIMEOUT,
    )
    return resp["choices"][0]["message"]["content"].strip()

def chat_with_retry(prompt: str) -> str:
    last_err: Optional[Exception] = None
    for k in range(ExtractionConfig.RETRY_ATTEMPTS):
        try:
            return _chat_once(prompt)
        except Exception as e:
            last_err = e
            backoff = (ExtractionConfig.RETRY_BACKOFF_BASE ** k)
            time.sleep(backoff)
    raise RuntimeError(f"LLM 请求失败（已重试 {ExtractionConfig.RETRY_ATTEMPTS} 次）: {last_err}")

def call_openai_json(prompt: str) -> Dict:
    try:
        content = chat_with_retry(prompt)
        return safe_json_loads(content)
    except Exception:
        return {}

# ===== 抽取逻辑 =====
def extract_entities(section_summary: str) -> List[Dict]:
    section_summary = clean_text(section_summary)
    prompt = ExtractionConfig.ENTITY_PROMPT.format(section_summary=section_summary)
    data = call_openai_json(prompt)
    ents = data.get("entities", [])
    # 轻量清洗：去除空名；alias 正规化为空数组
    cleaned = []
    for e in ents if isinstance(ents, list) else []:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        alias = e.get("alias") or []
        if isinstance(alias, str):
            alias = [alias] if alias.strip() else []
        cleaned.append({
            "name": name,
            "alias": [a for a in alias if isinstance(a, str) and a.strip()],
            "type": (e.get("type") or "").strip(),
            "raw_content": (e.get("raw_content") or "").strip(),
        })
    return cleaned

def extract_relations(section_summary: str, entities: List[Dict]) -> List[Dict]:
    section_summary = clean_text(section_summary)
    names = [e.get("name", "").strip() for e in entities if e.get("name")]
    entity_list = ", ".join([n for n in names if n])
    if not entity_list:
        return []
    prompt = ExtractionConfig.RELATION_PROMPT.format(
        section_summary=section_summary,
        entity_list=entity_list
    )
    data = call_openai_json(prompt)
    rels = data.get("relations", [])
    cleaned = []
    for r in rels if isinstance(rels, list) else []:
        src = (r.get("source") or "").strip()
        tgt = (r.get("target") or "").strip()
        if not src or not tgt:
            continue
        cleaned.append({
            "source": src,
            "target": tgt,
            "type": (r.get("type") or "").strip(),
            "description": (r.get("description") or "").strip(),
        })
    return cleaned

def process_one_subsection(summary: str) -> Dict:
    ents = extract_entities(summary)
    if not ents:
        return {"entities": [], "relations": []}
    rels = extract_relations(summary, ents)
    return {"entities": ents, "relations": rels}

def collect_subsections(toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """收集 level=3 小节（section 的 children），并且含有 summary。"""
    subs: List[Dict[str, Any]] = []
    for ch in toc:
        for sec in ch.get("children", []) or []:
            for sub in sec.get("children", []) or []:
                if sub.get("summary"):
                    subs.append(sub)
    return subs

# ===== 主流程（并发）=====
def run(in_json: Path, out_json: Path, max_workers: int):
    if not in_json.exists():
        raise FileNotFoundError(f"未找到输入：{in_json}")

    with in_json.open("r", encoding=ExtractionConfig.ENCODING) as f:
        toc: List[Dict[str, Any]] = json.load(f)

    subsections = collect_subsections(toc)
    total = len(subsections)

    if total == 0:
        # 直接输出空结构或原结构
        with out_json.open("w", encoding=ExtractionConfig.ENCODING) as f:
            json.dump(toc, f, ensure_ascii=False, indent=2)
        print(f"✅ 无需抽取（没有找到含摘要的小节）。已写出：{out_json.resolve()}")
        return

    pbar = tqdm(total=total, desc="🔍 实体与关系提取", ncols=90)

    def worker(subnode: Dict[str, Any]):
        res = process_one_subsection(subnode.get("summary", ""))
        subnode["entities"] = res["entities"]
        subnode["relations"] = res["relations"]
        pbar.update(1)

    # 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, sub) for sub in subsections]
        for fu in as_completed(futures):
            # 触发异常立即抛出，方便定位
            fu.result()

    pbar.close()

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding=ExtractionConfig.ENCODING) as f:
        json.dump(toc, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 实体与关系提取完成：{out_json.resolve()}")

def main():
    ap = argparse.ArgumentParser(description="从章节摘要中并发抽取实体与关系")
    ap.add_argument("--in", dest="in_json", type=str, default=str(ExtractionConfig.IN_JSON_PATH),
                    help="输入 JSON（默认：toc_with_summaries.json）")
    ap.add_argument("--out", dest="out_json", type=str, default=str(ExtractionConfig.OUT_JSON_PATH),
                    help="输出 JSON（默认：toc_with_entities_and_relations.json）")
    ap.add_argument("--workers", type=int, default=ExtractionConfig.MAX_WORKERS,
                    help="并发线程数（默认见配置 EXT_MAX_WORKERS）")
    args = ap.parse_args()

    run(Path(args.in_json), Path(args.out_json), max_workers=args.workers)

if __name__ == "__main__":
    main()
