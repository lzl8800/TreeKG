# 继续使用之前的模块
from __future__ import annotations

import json
import re
import time
import logging
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

# 统一导入配置命名空间（不使用别名）
from HiddenKG.config import APIConfig
from HiddenKG.config.aggr import AggrConfig as Aggr

# ========== 日志 ==========
LOG_PATH = Aggr.LOG_DIR / "aggr.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("Aggr")

# 复用长连接，减少握手与队头阻塞
SESSION = requests.Session()

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
    snippet: str = ""  # 例："entity_related|undirected" 或 "has_subordinate|out"

@dataclass
class EntityItem:
    name: str
    alias: List[str]
    type: str
    original: str
    occurrences: List[Occurrence]
    neighbors: List[Neighbor]
    role: str = ""  # "core" / "non-core"
    updated_description: str = ""  # 预留

# ======== 提示词（第一行只输出标签） ========
def build_system_prompt() -> str:
    return (
        "你是知识图谱构建专家。任务：判断实体是否为核心（core）或非核心（non-core）。\n"
        "规则：\n"
        "- core 实体通常是该领域的核心概念或重要术语（如主要学科概念、关键技术）。\n"
        "- non-core 实体通常是附属、具体或次要术语（如具体应用、辅助工具）。\n"
        "请严格按以下格式作答：\n"
        "第一行：只输出 core 或 non-core（不加标点、不加解释）\n"
        "第二行起：若需要再给出简要说明。\n"
        "若不确定，倾向输出 non-core。"
    )

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

def build_user_prompt_for_role(entity: EntityItem, section_summary: str) -> str:
    neighbors = entity.neighbors[:6]
    occs = entity.occurrences[:2]
    desc = _truncate(entity.original, 300)
    lines = []
    for nb in neighbors:
        sn = _truncate((nb.snippet or "").replace("\n", " "), 80)
        if "|" in sn:
            r, d, *_ = sn.split("|")
            lines.append(f"- {nb.name}（关系:{r.strip()}, 方向:{d.strip()}）")
        else:
            lines.append(f"- {nb.name}：{sn}")

    occ_str = "; ".join(o.path for o in occs) or "(无)"
    neigh_str = "\n".join(lines) if lines else "(无)"

    return (
        "第一行只输出最终标签（core 或 non-core）。\n\n"
        f"目标实体：{entity.name}\n"
        f"实体类型：{entity.type}\n"
        f"实体描述：{desc}\n"
        f"同现目录（最多2条）：{occ_str}\n"
        "邻域实体（最多6个）：\n" + neigh_str + "\n"
        f"章节摘要（可选）：{_truncate(section_summary, 200)}\n"
    )

# ======== LLM 调用与解析 ========
_RE_CORE = r"(?:\bcore\b|核心)(?!\s*(?:词|概念|内容))"
_RE_NONCORE = r"(?:\bnon[-\s]?core\b|非核心)"
ROLE_PATTERN = re.compile(rf"{_RE_NONCORE}|{_RE_CORE}", flags=re.I)

def _parse_role(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", "")
    # 1) 先看第一行（严格只取首个非空行）
    for ln in t.splitlines():
        ln = ln.strip().strip(" :：，。;；\"'`*").lower()
        if not ln:
            continue
        if ln in {"core", "核心"}:
            return "core"
        if ln in {"non-core", "noncore", "non core", "非核心"}:
            return "non-core"
        if ln.startswith(("non-core", "non core", "非核心")):
            return "non-core"
        if ln.startswith(("core", "核心")):
            return "core"
        break
    # 2) 回退：全文正则（non-core 优先）
    matches = list(ROLE_PATTERN.finditer(t))
    if not matches:
        return ""
    last = matches[-1].group(0).lower().strip()
    return "non-core" if ("non" in last or "非核" in last) else "core"

def call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, bool, str, int, bool]:
    if Aggr.DRY_RUN:
        return "", False, "dry_run_enabled", 0, True
    if not APIConfig.API_BASE:
        logger.warning("API_BASE 为空，跳过 LLM 调用。")
        return "", False, "api_base_empty", 0, False

    headers = {"Content-Type": "application/json"}
    if getattr(APIConfig, "API_KEY", None):
        headers["Authorization"] = f"Bearer {APIConfig.API_KEY}"

    payload = {
        "model": APIConfig.MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": Aggr.TEMPERATURE,
        "max_tokens": Aggr.MAX_TOKENS,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    last_err = ""
    for attempt in range(1, Aggr.RETRIES + 1):
        try:
            resp = SESSION.post(
                f"{APIConfig.API_BASE}{Aggr.CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=Aggr.API_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content:
                return content, True, "", attempt, False
            last_err = "empty_content"
        except Exception as e:
            last_err = str(e)
            logger.warning(f"LLM 调用失败(第 {attempt}/{Aggr.RETRIES} 次)：{last_err}")
            time.sleep(2 ** attempt)
    return "", False, last_err or "unknown_error", Aggr.RETRIES, False

def _fallback_role(entity: EntityItem) -> Tuple[str, str]:
    score = len(entity.occurrences) + 0.5 * len(entity.neighbors)
    role = "core" if score >= 4 else "non-core"
    return role, f"fallback_by_heuristic: score={score:.1f} threshold=4 -> {role}"

def judge_role(entity: EntityItem, section_summary: str) -> Tuple[str, dict]:
    sys_p = build_system_prompt()
    usr_p = build_user_prompt_for_role(entity, section_summary)
    content, ok, err, attempts, dry = call_llm(sys_p, usr_p)
    parsed = _parse_role(content)
    used_fallback = False
    note = ""
    if (not ok) or (parsed not in ("core", "non-core")):
        used_fallback = True
        parsed, note = _fallback_role(entity)

    diag = {
        "ok": ok,
        "attempts": attempts,
        "used_fallback": used_fallback,
        "note": note,
        "raw": content,
        "dry_run": dry,
    }
    return parsed, diag

# ======== 结构调整（横改纵 + 树约束） ========
def _is_vertical(snippet: str) -> bool:
    head = (snippet or "").split("|", 1)[0].strip().lower()
    return head in {"has_subordinate", "has_parent", "has_entity", "has_subsection"}

def _is_horizontal(snippet: str) -> bool:
    return not _is_vertical(snippet)

def _mark_subordinate() -> str:
    return "has_subordinate|out"   # core -> non-core

def _mark_parent() -> str:
    return "has_parent|in"         # non-core -> core（回指）

def _build_noncore_parent_candidates(entities: Dict[str, EntityItem]) -> Dict[str, List[str]]:
    cand: Dict[str, List[str]] = {}
    for core_name, core_ent in entities.items():
        if core_ent.role != "core":
            continue
        for nb in core_ent.neighbors:
            child = entities.get(nb.name)
            if not child:
                continue
            if child.role == "non-core" and _is_horizontal(nb.snippet):
                cand.setdefault(child.name, []).append(core_name)
    return cand

def _choose_single_parent(candidates: Dict[str, List[str]]) -> Dict[str, str]:
    chosen: Dict[str, str] = {}
    for child, parents in candidates.items():
        if parents:
            chosen[child] = parents[0]
    return chosen

def _apply_edge_conversion(entities: Dict[str, EntityItem], chosen: Dict[str, str]) -> Tuple[int, int]:
    added = 0
    removed = 0
    for child_name, parent_name in chosen.items():
        core_ent = entities.get(parent_name)
        child_ent = entities.get(child_name)
        if not core_ent or not child_ent:
            continue

        # 删除 core 与 child 的横向边
        old = len(core_ent.neighbors)
        core_ent.neighbors = [
            Neighbor(n.name, n.snippet)
            for n in core_ent.neighbors
            if not (n.name == child_name and _is_horizontal(n.snippet))
        ]
        removed += (old - len(core_ent.neighbors))

        old = len(child_ent.neighbors)
        child_ent.neighbors = [
            Neighbor(n.name, n.snippet)
            for n in child_ent.neighbors
            if not (n.name == parent_name and _is_horizontal(n.snippet))
        ]
        removed += (old - len(child_ent.neighbors))

        # 添加纵边 core -> child
        if not any(n.name == child_name and n.snippet.split("|", 1)[0] == "has_subordinate"
                   for n in core_ent.neighbors):
            core_ent.neighbors.append(Neighbor(name=child_name, snippet=_mark_subordinate()))
            added += 1

        # 子节点回指
        if not any(n.name == parent_name and n.snippet.split("|", 1)[0] == "has_parent"
                   for n in child_ent.neighbors):
            child_ent.neighbors.append(Neighbor(name=parent_name, snippet=_mark_parent()))
            added += 1

    # 保证每个 non-core 只有一个 has_parent（树约束）
    for ent in entities.values():
        if ent.role != "non-core":
            continue
        parents = [n for n in ent.neighbors if n.snippet.split("|", 1)[0] == "has_parent"]
        if len(parents) <= 1:
            continue
        keep = chosen.get(ent.name)
        ent.neighbors = [
            n for n in ent.neighbors
            if n.snippet.split("|", 1)[0] != "has_parent" or n.name == keep
        ]

    return added, removed

# ======== 主流程 ========
def run_aggr():
    Aggr.ensure_paths()

    # 读取实体文件（上游 Conv 输出）
    conv_path = Path(Aggr.FILE_CONV_RESULT)
    if not conv_path.exists():
        raise FileNotFoundError(
            f"未找到 conv 实体文件：{conv_path}\n"
            f"可通过环境变量 AGGR_CONV_INPUT 覆盖该路径。"
        )

    with conv_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # 转对象
    entities: Dict[str, EntityItem] = {}
    for name, data in raw.items():
        entities[name] = EntityItem(
            name=name,
            alias=data.get("alias", []),
            type=data.get("type", ""),
            original=data.get("original", ""),
            updated_description=data.get("updated_description", ""),
            occurrences=[Occurrence(**o) for o in data.get("occurrences", [])],
            neighbors=[Neighbor(**n) for n in data.get("neighbors", [])],
        )

    section_summary = "章节内容摘要"  # 可替换/扩展

    # 阶段1：角色判定
    logger.info("===== [Stage 1] 角色判定（第一行标签；短输出；失败启发式兜底）=====")
    for name, ent in tqdm(entities.items(), desc="Assigning roles", ncols=100):
        role, diag = judge_role(ent, section_summary)
        ent.role = role
        logger.info(
            f"[aggr][role] entity={name} -> role={role} | "
            f"llm_ok={diag['ok']}, attempts={diag['attempts']}, used_fallback={diag['used_fallback']}"
        )

    # 阶段2：结构调整（横改纵 + 树约束）
    logger.info("===== [Stage 2] 结构调整（横改纵 + 树约束）=====")
    candidates = _build_noncore_parent_candidates(entities)
    chosen = _choose_single_parent(candidates)
    added, removed = _apply_edge_conversion(entities, chosen)
    logger.info(f"[aggr][adjust] 候选父子对总数={sum(len(v) for v in candidates.values())}，选择父:子={len(chosen)} 对")
    logger.info(f"[aggr][adjust] 新增纵边(含回指)={added}，删除横边={removed}")

    # 写出结果
    out = {}
    for name, ent in tqdm(entities.items(), desc="Writing results", ncols=100):
        out[name] = {
            "name": ent.name,
            "alias": ent.alias,
            "type": ent.type,
            "original": ent.original,
            "updated_description": ent.updated_description,
            "role": ent.role,
            "occurrences": [asdict(o) for o in ent.occurrences],
            "neighbors": [asdict(n) for n in ent.neighbors],
        }

    out_path = Path(Aggr.FILE_AGGR_RESULT)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"[aggr] 完成，共处理 {len(out)} 个实体。输出文件：{out_path}")
    print(f"\n✅ Aggr 阶段完成，日志写入: {LOG_PATH}")

# ======== CLI ========
if __name__ == "__main__":
    run_aggr()
