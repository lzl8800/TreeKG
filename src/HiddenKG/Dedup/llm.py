import json
import re
import time
import requests
import threading
from typing import Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# 从配置导入（对齐 conv，使用 Conv 配置的超时/重试等参数）
from HiddenKG.config import APIConfig, Conv

# ========== 限速配置（直接复用 conv 的限速逻辑） ==========
_rate_lock = threading.Lock()
_last_ts = 0.0
# 从 Conv 配置读取 QPS 限制，避免硬编码
_min_interval = (1.0 / Conv.RATE_LIMIT_QPS) if Conv.RATE_LIMIT_QPS > 0 else 0.0


def _rate_limit_block():
    """复用 conv 的 QPS 限速逻辑，避免接口过载"""
    if _min_interval <= 0:
        return
    global _last_ts
    with _rate_lock:
        now = time.time()
        delta = now - _last_ts
        if delta < _min_interval:
            time.sleep(_min_interval - delta)
        _last_ts = time.time()


# ========== 基础工具函数（对齐 conv 的格式处理） ==========
def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "…")


def _text(ent) -> str:
    return (ent.updated_description or ent.original or "").strip()


# ========== LLM 调用核心函数（完全对齐 conv 逻辑） ==========
def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    对齐 conv 的 LLM 调用逻辑：
    1. 支持无 API_KEY（本地网关无鉴权）
    2. 用 Conv 配置的超时/重试/节流参数
    3. 正确拼接 API 路径（API_BASE + CHAT_COMPLETIONS_PATH）
    4. 指数退避重试 + 固定节流
    """
    # Dry Run 模式（同 conv，用于测试）
    if Conv.DRY_RUN:
        return '{"is_same": false, "reason": "dry_run_mode"}'

    # 校验 API 基础路径（同 conv，优先检查 API_BASE）
    if not APIConfig.API_BASE:
        print(f"[WARNING] API_BASE 为空，跳过 LLM 调用")
        return '{"is_same": false, "reason": "api_base_empty"}'

    # 构建请求头（支持无 API_KEY，同 conv）
    headers = {"Content-Type": "application/json"}
    if getattr(APIConfig, "API_KEY", None) and APIConfig.API_KEY.strip():
        headers["Authorization"] = f"Bearer {APIConfig.API_KEY.strip()}"

    # 构建请求体（对齐 conv 的参数风格，用 Conv 配置的温度）
    payload = {
        "model": APIConfig.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": Conv.TEMPERATURE,  # 复用 conv 的温度配置，避免不一致
        "max_tokens": 128  # 实体去重判定无需长输出，保持精简
    }

    # 重试逻辑（同 conv 的指数退避）
    last_err: Optional[Exception] = None
    for retry_idx in range(Conv.RETRIES):
        try:
            # 1. 限速（同 conv 的 QPS 控制）
            _rate_limit_block()
            # 2. 固定节流（同 conv，避免突发请求）
            if Conv.EXTRA_THROTTLE_SEC > 0:
                time.sleep(Conv.EXTRA_THROTTLE_SEC)

            # 3. 发送请求（对齐 conv 的路径拼接方式）
            resp = requests.post(
                url=f"{APIConfig.API_BASE}{Conv.CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=Conv.API_TIMEOUT  # 复用 conv 的超时配置
            )
            resp.raise_for_status()  # 触发 HTTP 错误（如 401/404/500）

            # 4. 提取结果（同 conv 的响应解析逻辑）
            return resp.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            last_err = e
            # 日志风格对齐 conv，包含重试次数
            print(f"[WARNING] LLM 调用失败（{retry_idx + 1}/{Conv.RETRIES}）：{str(e)[:100]}")
            # 指数退避等待（同 conv 的重试策略）
            time.sleep(Conv.RETRY_BACKOFF_BASE ** retry_idx)

    # 所有重试失败
    print(f"[ERROR] LLM 调用彻底失败：{str(last_err)[:100]}")
    return '{"is_same": false, "reason": "llm_call_failed"}'


# ========== 结果解析（对齐 conv 的安全解析逻辑） ==========
def safe_parse_llm_result(content: str) -> Tuple[bool, str]:
    """复用 conv 的安全 JSON 解析逻辑，避免格式错误导致崩溃"""
    if not content or not isinstance(content, str):
        return False, "empty_llm_response"

    # 1. 直接解析（优先尝试）
    try:
        obj = json.loads(content.strip())
        is_same = obj.get("is_same", False)
        reason = obj.get("reason", str(obj))[:200]  # 限制 reason 长度
        return bool(is_same), reason
    except Exception:
        pass

    # 2. 提取平层 JSON 对象（应对 LLM 输出多余文本）
    candidates = re.findall(r"\{[^{}]*\}", content)
    for cand in candidates:
        try:
            obj = json.loads(cand)
            is_same = obj.get("is_same", False)
            reason = obj.get("reason", "parsed_from_flat_json")[:200]
            return bool(is_same), reason
        except Exception:
            continue

    # 3. 外层截取（最后尝试，应对格式混乱）
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            obj = json.loads(content[start_idx:end_idx + 1])
            is_same = obj.get("is_same", False)
            reason = obj.get("reason", "parsed_from_segment")[:200]
            return bool(is_same), reason
        except Exception:
            pass

    # 所有解析方式失败
    return False, f"parse_failed: {content[:100]}"


# ========== 对外接口（实体去重判定） ==========
def llm_is_same(ent_a, ent_b, truncate_desc: bool, desc_maxlen: int) -> Tuple[bool, str, bool]:
    """
    对齐 conv 逻辑后的实体去重判定：
    返回：(是否同一实体, 原因, 是否使用兜底)
    """
    # 构建 Prompt（参考 conv 的结构化风格，清晰简洁）
    system_prompt = (
        "你是“MATLAB科学计算课程知识图谱专家”，擅长实体消歧。"
        "仅基于实体的名称、角色、描述判断是否为同一概念，输出严格 JSON 格式，不要多余文本。"
    )

    # 处理描述截断（同原逻辑，但格式对齐 conv）
    desc_a = _text(ent_a)
    desc_b = _text(ent_b)
    if truncate_desc:
        desc_a = _shorten(desc_a, desc_maxlen)
        desc_b = _shorten(desc_b, desc_maxlen)

    # 结构化 User Prompt（对齐 conv 的清晰格式，便于 LLM 理解）
    user_prompt = (
        f"### 实体1\n"
        f"- 名称：{ent_a.name}\n"
        f"- 角色：{ent_a.role or '无'}\n"
        f"- 描述：{desc_a or '无'}\n\n"
        f"### 实体2\n"
        f"- 名称：{ent_b.name}\n"
        f"- 角色：{ent_b.role or '无'}\n"
        f"- 描述：{desc_b or '无'}\n\n"
        f"### 任务\n"
        f"判断两个实体是否为同一概念，输出 JSON：\n"
        f'{{"is_same": true/false, "reason": "简要说明判断依据（100字内）"}}'
    )

    # 调用 LLM（核心逻辑对齐 conv）
    llm_content = call_llm(system_prompt, user_prompt)
    # 解析结果（安全解析，对齐 conv）
    is_same, reason = safe_parse_llm_result(llm_content)
    # 兜底标记（仅当解析失败或 LLM 明确返回失败时为 True）
    used_fallback = reason in ["empty_llm_response", "api_base_empty", "llm_call_failed", "parse_failed"]

    return is_same, reason[:200], used_fallback


# ========== 兜底逻辑（保留原逻辑，补充日志） ==========
def _fallback_is_same(a, b, last_err: str) -> Tuple[bool, str]:
    """超保守兜底：仅当 name/alias 交集时合并，同原逻辑"""
    alias_a = {_normalize_name(x) for x in ([a.name] + a.alias)}
    alias_b = {_normalize_name(x) for x in ([b.name] + b.alias)}
    if alias_a & alias_b:
        reason = "fallback: alias/name_intersection"
        print(f"[FALLBACK] 触发别名兜底合并：{a.name} <-> {b.name}")
        return True, reason
    reason = f"fallback: no_intersection (err={last_err[:50]})"
    return False, reason


def _normalize_name(s: str) -> str:
    """同原逻辑，标准化名称用于兜底判断"""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s