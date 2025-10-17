# -*- coding: utf-8 -*-
"""
HiddenKG/Dedup/llm.py — YAML配置直读版
- 统一从 HiddenKG/config/config.yaml 读取主配置，并合并 include_files（相对 config.yaml 解析）
- 使用 APIConfig + DedupConfig 的参数（温度/超时/重试/QPS/节流/退避/路径/DRY_RUN）
- Prompt 模板直接来自 DedupConfig: LLM_SYSTEM_PROMPT / LLM_USER_PROMPT
"""

from __future__ import annotations

import json
import re
import time
import threading
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import requests
import yaml

# =========================
# 配置加载
# =========================
def _load_config() -> dict:
    """
    从 HiddenKG/config/config.yaml 读取主配置，并合并 include_files。
    include_files 中的每个条目都按“相对 config.yaml 所在目录”解析。
    """
    hiddenkg_dir = Path(__file__).resolve().parents[1]          # .../src/HiddenKG
    cfg_path = hiddenkg_dir / "config" / "config.yaml"          # .../src/HiddenKG/config/config.yaml
    if not cfg_path.exists():
        raise FileNotFoundError(f"未找到配置文件：{cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}

    merged = dict(base)
    incs = base.get("include_files", []) or []
    for rel in incs:
        inc_path = (cfg_path.parent / rel).resolve()
        if not inc_path.exists():
            raise FileNotFoundError(f"未找到 include 文件：{inc_path}")
        with inc_path.open("r", encoding="utf-8") as ff:
            sub = yaml.safe_load(ff) or {}
        merged.update(sub)

    return merged

_CFG = _load_config()
_API: Dict[str, Any] = _CFG["APIConfig"]
_DCFG: Dict[str, Any] = _CFG["DedupConfig"]

# Prompt 模板（直接来自 YAML；缺失时给出极简兜底）
_SYS_TMPL = (_DCFG.get("LLM_SYSTEM_PROMPT") or "").strip() or (
    "你是“MATLAB科学计算课程知识图谱专家”，擅长实体消歧。"
    "仅基于实体的名称、角色、描述判断是否为同一概念，输出严格 JSON，不要多余文本。"
)
_USR_TMPL = (_DCFG.get("LLM_USER_PROMPT") or "").strip() or (
    "### 实体1\n"
    "- 名称：{name_a}\n"
    "- 角色：{role_a}\n"
    "- 描述：{desc_a}\n\n"
    "### 实体2\n"
    "- 名称：{name_b}\n"
    "- 角色：{role_b}\n"
    "- 描述：{desc_b}\n\n"
    "### 任务\n"
    "判断两个实体是否为同一概念，输出 JSON（严格按键名）：\n"
    "{schema}\n"
)

# =========================
# 限速（QPS）
# =========================
_rate_lock = threading.Lock()
_last_ts = 0.0
_qps = float(_DCFG.get("RATE_LIMIT_QPS", 0) or 0)
_MIN_INTERVAL = (1.0 / _qps) if _qps > 0 else 0.0

def _rate_limit_block():
    """QPS 限速：保证两次请求间隔 >= 1/QPS"""
    if _MIN_INTERVAL <= 0:
        return
    global _last_ts
    with _rate_lock:
        now = time.time()
        delta = now - _last_ts
        if delta < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - delta)
        _last_ts = time.time()

# 复用会话减少握手
_SESSION = requests.Session()

# =========================
# 工具
# =========================
def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "…")

def _text(ent) -> str:
    return (ent.updated_description or ent.original or "").strip()

# =========================
# LLM 调用
# =========================
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    OpenAI 兼容 /chat/completions
    - 使用 DedupConfig 的温度、超时、重试、节流、退避、路径、DRY_RUN
    - 使用 APIConfig 的 API_BASE / API_KEY / MODEL_NAME
    """
    if bool(_DCFG.get("DRY_RUN", False)):
        return '{"is_same": false, "reason": "dry_run_mode"}'

    api_base = (_API.get("API_BASE") or "").strip()
    if not api_base:
        print("[WARNING] API_BASE 为空，跳过 LLM 调用")
        return '{"is_same": false, "reason": "api_base_empty"}'

    headers = {"Content-Type": "application/json"}
    api_key = (_API.get("API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": _API.get("MODEL_NAME", ""),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(_DCFG.get("TEMPERATURE", 0.0)),
        "max_tokens": int(_DCFG.get("MAX_TOKENS", 128)),
    }

    retries = int(_DCFG.get("RETRIES", 3))
    timeout = int(_DCFG.get("API_TIMEOUT", 120))
    throttle = float(_DCFG.get("EXTRA_THROTTLE_SEC", 0.0))
    backoff_base = float(_DCFG.get("RETRY_BACKOFF_BASE", 1.8))
    path = _DCFG.get("CHAT_COMPLETIONS_PATH", "/chat/completions")

    last_err: Optional[Exception] = None
    for k in range(retries):
        try:
            _rate_limit_block()
            if throttle > 0:
                time.sleep(throttle)

            resp = _SESSION.post(
                f"{api_base}{path}",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            print(f"[WARNING] LLM 调用失败（{k+1}/{retries}）：{str(e)[:120]}")
            time.sleep(backoff_base ** k)

    print(f"[ERROR] LLM 调用彻底失败：{str(last_err)[:120]}")
    return '{"is_same": false, "reason": "llm_call_failed"}'

# =========================
# 解析
# =========================
def safe_parse_llm_result(content: str) -> Tuple[bool, str]:
    """
    安全解析：直解析 → 平层对象 → 外层截取
    """
    if not content or not isinstance(content, str):
        return False, "empty_llm_response"

    try:
        obj = json.loads(content.strip())
        return bool(obj.get("is_same", False)), (obj.get("reason") or str(obj))[:200]
    except Exception:
        pass

    flats = re.findall(r"\{[^{}]*\}", content)
    for c in flats:
        try:
            obj = json.loads(c)
            return bool(obj.get("is_same", False)), (obj.get("reason", "parsed_from_flat_json"))[:200]
        except Exception:
            continue

    a, b = content.find("{"), content.rfind("}")
    if a != -1 and b != -1 and b > a:
        try:
            obj = json.loads(content[a:b + 1])
            return bool(obj.get("is_same", False)), (obj.get("reason", "parsed_from_segment"))[:200]
        except Exception:
            pass

    return False, f"parse_failed: {content[:100]}"

# =========================
# 对外：是否同一实体
# =========================
def llm_is_same(ent_a, ent_b, truncate_desc: bool, desc_maxlen: int) -> Tuple[bool, str, bool]:
    """
    返回：(is_same, reason, used_fallback)
    - 提示词模板直接使用 YAML 中的 LLM_SYSTEM_PROMPT / LLM_USER_PROMPT
    - 占位符：name_a/role_a/desc_a/name_b/role_b/desc_b/schema
    """
    # 描述处理
    desc_a = _text(ent_a)
    desc_b = _text(ent_b)
    if truncate_desc:
        desc_a = _shorten(desc_a, int(_DCFG.get("DESC_MAXLEN", desc_maxlen)))
        desc_b = _shorten(desc_b, int(_DCFG.get("DESC_MAXLEN", desc_maxlen)))

    schema = '{"is_same": true/false, "reason": "简要说明判断依据（100字内）"}'

    # 渲染用户提示词
    try:
        user_prompt = _USR_TMPL.format(
            name_a=ent_a.name,
            role_a=(ent_a.role or "无"),
            desc_a=(desc_a or "无"),
            name_b=ent_b.name,
            role_b=(ent_b.role or "无"),
            desc_b=(desc_b or "无"),
            schema=schema,
        )
    except KeyError:
        # 若模板占位符写错，给出兜底模板以不中断流程
        user_prompt = (
            "### 实体1\n"
            f"- 名称：{ent_a.name}\n"
            f"- 角色：{ent_a.role or '无'}\n"
            f"- 描述：{desc_a or '无'}\n\n"
            "### 实体2\n"
            f"- 名称：{ent_b.name}\n"
            f"- 角色：{ent_b.role or '无'}\n"
            f"- 描述：{desc_b or '无'}\n\n"
            "### 任务\n"
            f"判断两个实体是否为同一概念，输出 JSON（严格按键名）：\n{schema}\n"
        )

    system_prompt = _SYS_TMPL

    content = call_llm(system_prompt, user_prompt)
    is_same, reason = safe_parse_llm_result(content)
    used_fallback = reason in {"empty_llm_response", "api_base_empty", "llm_call_failed"} or reason.startswith("parse_failed")
    return is_same, reason[:200], used_fallback

# =========================
# 兜底工具（供 Dedup 使用）
# =========================
def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _fallback_is_same(a, b, last_err: str) -> Tuple[bool, str]:
    alias_a = {_normalize_name(x) for x in ([a.name] + a.alias)}
    alias_b = {_normalize_name(x) for x in ([b.name] + b.alias)}
    if alias_a & alias_b:
        return True, "fallback: alias/name_intersection"
    return False, f"fallback: no_intersection (err={last_err[:50]})"
