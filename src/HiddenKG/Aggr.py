from __future__ import annotations
import json
import time
import logging
import requests
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
from tqdm import tqdm
import re
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========================= 配置加载（YAML）
def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# HiddenKG 目录
SCRIPT_DIR = Path(__file__).resolve().parent
CFG_DIR = SCRIPT_DIR / "config"

# 主配置：HiddenKG/config/config.yaml
MAIN_CFG_PATH = CFG_DIR / "config.yaml"
config = load_yaml(MAIN_CFG_PATH)

# 递归加载 include_files（相对 HiddenKG/config/ 拼接；绝对路径原样用）
includes = config.get("include_files", []) or []
for inc in includes:
    inc_path = Path(inc)
    if not inc_path.is_absolute():
        inc_path = (CFG_DIR / inc).resolve()
    if inc_path.exists():
        cfg_add = load_yaml(inc_path)
        if cfg_add:
            config.update(cfg_add)

# 取出 API 与 Aggr 配置（字典）
APIConfig = config.get("APIConfig", {})
AggrCfg   = config.get("AggrConfig", {})
# 关键修复①：PROMPTS 同时兼容顶层和 AggrConfig 下两种放法
PROMPTS   = (config.get("PROMPTS") or AggrCfg.get("PROMPTS") or {})

# ========================= 路径拼接
OUTPUT_DIR = SCRIPT_DIR / "output"
LOGS_DIR   = SCRIPT_DIR / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 名字到实际路径
CONV_IN_PATH  = OUTPUT_DIR / AggrCfg.get("CONV_IN_NAME", "conv_entities.json")
AGGR_OUT_PATH = OUTPUT_DIR / AggrCfg.get("OUT_NAME", "aggr_entities.json")
LOG_PATH      = LOGS_DIR   / AggrCfg.get("LOG_NAME", "aggr.log")

# ========================= 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger("Aggr")

# ========================= 运行参数
TEMPERATURE = float(AggrCfg.get("TEMPERATURE", 0.0))
MAX_TOKENS  = int(AggrCfg.get("MAX_TOKENS", 1000))
API_TIMEOUT = int(AggrCfg.get("API_TIMEOUT", 120))
RETRIES     = int(AggrCfg.get("RETRIES", 3))
CHAT_COMPLETIONS_PATH = AggrCfg.get("CHAT_COMPLETIONS_PATH", "/chat/completions")
DRY_RUN     = bool(int(AggrCfg.get("DRY_RUN", 0)))

LIMIT = int(AggrCfg.get("LIMIT", 0))
TREE_ENFORCE = bool(int(AggrCfg.get("TREE_ENFORCE", 1)))
PROGRESS_NCOLS = int(AggrCfg.get("PROGRESS_NCOLS", 100))
WORKERS = int(AggrCfg.get("WORKERS", os.cpu_count() or 6))
ENCODING = AggrCfg.get("ENCODING", "utf-8")

API_BASE = APIConfig.get("API_BASE", "")
API_KEY  = APIConfig.get("API_KEY", "")
MODEL    = APIConfig.get("MODEL_NAME", "")

# ========================= 数据结构
@dataclass
class Occurrence:
    path: str
    node_id: str
    level: int
    title: str

@dataclass
class Neighbor:
    name: str
    snippet: str = ""  # 如 "cooccur|undirected|w=..." 或 "rel|directed|type=part-of|src=core|dst=non-core"
    type: str = ""     # 标准化关系名：cooccur / prerequisite / part-of / applies-to / example-of / synonym / contrasts-with / related

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

# ========================= 工具：模板/解析/HTTP
def _prompts_section() -> dict:
    return PROMPTS or {}

def build_system_prompt_role() -> str:
    return (_prompts_section().get("system") or "").strip()

def build_system_prompt_relation() -> str:
    return (_prompts_section().get("relation_system") or "").strip()

def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return (s[:n] + "…") if len(s) > n else s

def _nb_list(ent: EntityItem, k=5) -> str:
    xs = []
    for nb in ent.neighbors[:k]:
        xs.append(f"- {nb.name}｜{_truncate(nb.snippet, 60)}")
    return "\n".join(xs) if xs else "(无)"

def build_user_prompt_for_role(entity: EntityItem, section_summary: str) -> str:
    neighbors = entity.neighbors[:6]
    occs = entity.occurrences[:2]
    desc = _truncate(entity.original, 300)
    lines = []
    for nb in neighbors:
        sn = _truncate((nb.snippet or "").replace("\n", " "), 80)
        head = sn.split("|", 1)[0] if "|" in sn else "neighbor"
        lines.append(f"- {nb.name}（{head}）")
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

# —— 关系标签集合 —— #
REL_TYPES = {"prerequisite","part-of","applies-to","example-of","synonym","contrasts-with","related"}
REL_PATTERN = re.compile(r"\b(prerequisite|part-of|applies-to|example-of|synonym|contrasts-with|related)\b", re.I)
REL_LABEL_CHOICES = "prerequisite|part-of|applies-to|example-of|synonym|contrasts-with|related"

# 关键修复②：关系模板安全渲染，避免 KeyError
class _SafeDict(dict):
    def __missing__(self, key):
        # 未提供的占位符保持原样，避免 KeyError
        return "{" + key + "}"

def build_user_prompt_for_relation(core: EntityItem, child: EntityItem) -> str:
    tpl = (_prompts_section().get("relation_user_template") or "").strip()
    if not tpl:
        return (
            "只输出一行：在 {prerequisite|part-of|applies-to|example-of|synonym|contrasts-with|related} 中选择一个最贴切的关系类型。\n\n"
            f"【Core】{core.name}（type={core.type}）\n描述：{_truncate(core.original, 200)}\n\n"
            f"【Non-Core】{child.name}（type={child.type}）\n描述：{_truncate(child.original, 200)}\n"
        )

    # 兼容老模板：若发现单花括号的 7 类标签，把它替换成受控占位符
    legacy = "{prerequisite|part-of|applies-to|example-of|synonym|contrasts-with|related}"
    if legacy in tpl:
        tpl = tpl.replace(legacy, "{label_choices}")

    fields = _SafeDict({
        "core_name": core.name,
        "core_type": core.type,
        "core_desc": _truncate(core.original, 260),
        "core_neighbors": _nb_list(core, 5),
        "child_name": child.name,
        "child_type": child.type,
        "child_desc": _truncate(child.original, 260),
        "child_neighbors": _nb_list(child, 5),
        "label_choices": REL_LABEL_CHOICES,  # 给模板使用
    })

    # 只格式化一次，且缺失字段不报错
    return tpl.format_map(fields)

def call_llm(system_prompt: str, user_prompt: str) -> Tuple[str, bool, str, int, bool]:
    """OpenAI 兼容 /chat/completions；线程安全：每次请求独立调用 requests.post"""
    if DRY_RUN:
        return "", False, "dry_run_enabled", 0, True
    if not API_BASE:
        logger.warning("API_BASE 为空，跳过 LLM 调用。")
        return "", False, "api_base_empty", 0, False

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    last_err = ""
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(
                f"{API_BASE}{CHAT_COMPLETIONS_PATH}",
                headers=headers,
                json=payload,
                timeout=API_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if content:
                return content, True, "", attempt, False
            last_err = "empty_content"
        except Exception as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
    return "", False, last_err or "unknown_error", RETRIES, False

# ========================= 角色判定（并发）
_RE_CORE = r"(?:\bcore\b|核心)(?!\s*(?:词|概念|内容))"
_RE_NONCORE = r"(?:\bnon[-\s]?core\b|非核心)"
ROLE_PATTERN = re.compile(rf"{_RE_NONCORE}|{_RE_CORE}", flags=re.I)

def _parse_role(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\r", "")
    # 只看第一行
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
    # 回退：全文正则
    matches = list(ROLE_PATTERN.finditer(t))
    if not matches:
        return ""
    last = matches[-1].group(0).lower().strip()
    return "non-core" if ("non" in last or "非核" in last) else "core"

def judge_role(entity: EntityItem, section_summary: str) -> Tuple[str, dict]:
    sys_p = build_system_prompt_role()
    usr_p = build_user_prompt_for_role(entity, section_summary)
    content, ok, err, attempts, dry = call_llm(sys_p, usr_p)
    parsed = _parse_role(content)
    used_fallback = False
    note = ""
    if (not ok) or (parsed not in ("core", "non-core")):
        used_fallback = True
        weight = float(AggrCfg.get("FALLBACK", {}).get("neighbor_weight", 0.5))
        threshold = float(AggrCfg.get("FALLBACK", {}).get("threshold", 4))
        score = len(entity.occurrences) + weight * len(entity.neighbors)
        parsed = "core" if score >= threshold else "non-core"
        note = f"fallback_by_heuristic: score={score:.1f} threshold={threshold} -> {parsed}"
    diag = {"ok": ok, "attempts": attempts, "used_fallback": used_fallback, "note": note, "raw": content, "dry_run": dry}
    return parsed, diag

# ========================= 父选择 + 兜底
def _is_horizontal(snippet: str) -> bool:
    head = (snippet or "").split("|", 1)[0].strip().lower()
    vertical_prefixes = [s.lower() for s in (AggrCfg.get("VERTICAL_REL_TYPES") or [])]
    return head not in vertical_prefixes and head != "rel"  # cooccur 等视为横向

def _cooccur_weight(s: str) -> int:
    m = re.search(r"\bw=(\d+)\b", s or "")
    try:
        return int(m.group(1)) if m else 0
    except Exception:
        return 0

def _build_noncore_parent_candidates(entities: Dict[str, EntityItem]) -> Dict[str, List[str]]:
    """从 core 的横向邻居里挑 non-core 作为子候选"""
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

def _fallback_pick_core_for_noncore(nc: EntityItem, entities: Dict[str, EntityItem]) -> Optional[str]:
    """兜底：优先选择与 non-core 共现权重最高的 core；再同 section 交集；再任取 core。"""
    best_core, best_w = None, -1
    for nb in nc.neighbors:
        ce = entities.get(nb.name)
        if not ce or ce.role != "core":
            continue
        w = _cooccur_weight(nb.snippet)
        if w > best_w:
            best_w, best_core = w, ce.name
    if best_core:
        return best_core

    nc_secs = {o.node_id for o in nc.occurrences}
    if nc_secs:
        score: Dict[str, int] = {}
        for cname, cent in entities.items():
            if cent.role != "core":
                continue
            c_secs = {o.node_id for o in cent.occurrences}
            inter = len(nc_secs & c_secs)
            if inter > 0:
                score[cname] = inter
        if score:
            return max(score.items(), key=lambda kv: kv[1])[0]

    for cname, cent in entities.items():
        if cent.role == "core":
            return cname
    return None

# ========================= 关系类型判定（并发）
def _parse_relation_type(text: str) -> Optional[str]:
    if not text:
        return None
    first = text.splitlines()[0].strip().lower()
    first = re.sub(r"[\"'`*：:，,。;\\s]+$", "", first)
    if first in REL_TYPES:
        return first
    m = REL_PATTERN.search(text)
    if m:
        cand = m.group(1).lower()
        if cand in REL_TYPES:
            return cand
    return None

def classify_relation_type(core_ent: EntityItem, child_ent: EntityItem) -> str:
    if DRY_RUN:
        return "related"
    sys_p = build_system_prompt_relation()
    usr_p = build_user_prompt_for_relation(core_ent, child_ent)
    content, ok, err, attempts, dry = call_llm(sys_p, usr_p)
    rel = _parse_relation_type(content or "")
    if rel in REL_TYPES:
        # 小偏置：如果 core 是方法/函数/算法而 child 不是，优先 applies-to
        if rel == "related" and any(k in core_ent.type.lower() for k in ["method","function","algorithm"]) and \
           not any(k in child_ent.type.lower() for k in ["method","function","algorithm"]):
            return "applies-to"
        return rel
    # 兜底
    if core_ent.name == child_ent.name or (set(core_ent.alias) & set(child_ent.alias)):
        return "synonym"
    if any(k in core_ent.type.lower() for k in ["method","function","algorithm"]) and \
       not any(k in child_ent.type.lower() for k in ["method","function","algorithm"]):
        return "applies-to"
    return "related"

def _mk_rel_snippet(rel_type: str, src_role="core", dst_role="non-core") -> str:
    return f"rel|directed|type={rel_type}|src={src_role}|dst={dst_role}"

# ========================= 类型补齐（写文件前）
def _infer_type_from_snippet(sn: str) -> str:
    if not sn:
        return ""
    head = sn.split("|", 1)[0].lower()
    if head == "rel":
        m = re.search(r"type=([a-z\\-]+)", sn, flags=re.I)
        return (m.group(1).lower() if m else "related")
    return head  # e.g. "cooccur", "has_subordinate" 等

# ========================= 主流程
def run_aggr():
    # 路径检查
    if not CONV_IN_PATH.exists():
        raise FileNotFoundError(
            f"未找到 conv 实体文件：{CONV_IN_PATH}\n"
            f"请确认 HiddenKG/output/ 目录或在 aggr.yaml 中设置 CONV_IN_NAME。"
        )

    with CONV_IN_PATH.open("r", encoding=ENCODING) as f:
        raw = json.load(f)

    # 转对象 + LIMIT
    items = list(raw.items())
    if LIMIT and LIMIT > 0:
        items = items[:LIMIT]

    entities: Dict[str, EntityItem] = {}
    for name, data in items:
        entities[name] = EntityItem(
            name=name,
            alias=data.get("alias", []),
            type=data.get("type", ""),
            original=data.get("original", ""),
            updated_description=data.get("updated_description", ""),
            occurrences=[Occurrence(**o) for o in data.get("occurrences", [])],
            neighbors=[Neighbor(**n) for n in data.get("neighbors", [])],
        )

    section_summary = "章节内容摘要"  # 如需，可从文件读取

    # 阶段1：角色判定（并发）
    logger.info(f"===== [Stage 1] 角色判定（并发={WORKERS}）=====")
    def _role_job(ent: EntityItem) -> Tuple[str, str, dict]:
        role, diag = judge_role(ent, section_summary)
        return ent.name, role, diag

    futures = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for ent in entities.values():
            futures.append(ex.submit(_role_job, ent))
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=PROGRESS_NCOLS, desc="Assign roles"):
            try:
                name, role, diag = fut.result()
                entities[name].role = role
                logger.info(f"[role] {name} -> {role} | ok={diag['ok']} attempts={diag['attempts']} fallback={diag['used_fallback']}")
            except Exception as e:
                logger.warning(f"[role] 任务失败：{e}")

    # 阶段2：父选择（保证每个 non-core 至少有一个 core；TREE_ENFORCE 时单父）
    logger.info("===== [Stage 2] 选择父 core 并保证覆盖所有 non-core =====")
    candidates = _build_noncore_parent_candidates(entities)  # child -> [core...]
    chosen: Dict[str, str] = {}
    for child, parents in candidates.items():
        if parents:
            chosen[child] = parents[0]

    for name, ent in entities.items():
        if ent.role != "non-core":
            continue
        if name not in chosen:
            alt = _fallback_pick_core_for_noncore(ent, entities)
            if alt:
                chosen[name] = alt

    if TREE_ENFORCE:
        # 当前实现就是单父，无需额外处理
        pass

    total_noncore = sum(1 for e in entities.values() if e.role == "non-core")
    logger.info(f"non-core 总数={total_noncore}，已分配父={len(chosen)}")

    # 阶段3：LLM 并发判定关系类型 + 写边（仅 core->non-core）
    logger.info(f"===== [Stage 3] 关系类型判定（并发={WORKERS}）并写边 =====")
    def _rel_job(child_name: str, parent_name: str) -> Tuple[str, str, str]:
        core_ent  = entities[parent_name]
        child_ent = entities[child_name]
        rel_type = classify_relation_type(core_ent, child_ent)
        return child_name, parent_name, rel_type

    rel_futs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for child_name, parent_name in chosen.items():
            rel_futs.append(ex.submit(_rel_job, child_name, parent_name))
        for fut in tqdm(as_completed(rel_futs), total=len(rel_futs), ncols=PROGRESS_NCOLS, desc="Classify relations"):
            try:
                child_name, parent_name, rel_type = fut.result()
                core_ent  = entities[parent_name]
                child_ent = entities[child_name]

                # 移除两侧横向边
                old_core = len(core_ent.neighbors)
                core_ent.neighbors = [
                    n for n in core_ent.neighbors
                    if not (n.name == child_name and _is_horizontal(n.snippet))
                ]
                core_removed = old_core - len(core_ent.neighbors)

                old_child = len(child_ent.neighbors)
                child_ent.neighbors = [
                    n for n in child_ent.neighbors
                    if not (n.name == parent_name and _is_horizontal(n.snippet))
                ]
                child_removed = old_child - len(child_ent.neighbors)

                # 添加 core -> non-core（不写回指），同时写 type
                sn = _mk_rel_snippet(rel_type, "core", "non-core")
                if not any(n.name == child_name and n.snippet == sn for n in core_ent.neighbors):
                    core_ent.neighbors.append(Neighbor(name=child_name, snippet=sn, type=rel_type))

                total_removed = max(core_removed, 0) + max(child_removed, 0)
                logger.info(f"[edge] {parent_name} -> {child_name} ({rel_type}) | removed_horizontal={total_removed}")
            except Exception as e:
                logger.warning(f"[rel] 任务失败：{e}")

    # —— 写出前统一补齐 neighbors[*].type —— #
    for ent in entities.values():
        for nb in ent.neighbors:
            if not getattr(nb, "type", ""):
                nb.type = _infer_type_from_snippet(nb.snippet)

    # 写出结果
    out = {}
    for name, ent in tqdm(entities.items(), desc="Write results", ncols=PROGRESS_NCOLS):
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

    AGGR_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with AGGR_OUT_PATH.open("w", encoding=ENCODING) as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    logger.info(f"完成，共处理 {len(out)} 个实体。输出文件：{AGGR_OUT_PATH}")
    print(f"\n✅ Aggr 阶段完成（并发 {WORKERS}），日志：{LOG_PATH}")

# ========================= CLI
if __name__ == "__main__":
    run_aggr()
