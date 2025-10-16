# -*- coding: utf-8 -*-
"""
TextSegmentation.py  (严格版·增强·配置外置)
- 章/节/小节/知识点：仅段首匹配；样式优先，正则回退（可配置）
- 标题清洗与中文数字解析；兼容全/半角冒号、奇异空格、尾部页码点线
- 告警（层级跳跃/编号不一致）可打印+落盘
- 默认从 config/ 读取通用路径与本流程专用配置
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.text.paragraph import Paragraph

# —— 配置 —— #
try:
    from config import OUTPUT_DIR, TextSegConfig
except Exception as e:
    raise RuntimeError(f"无法导入配置：{e}")

# —— 日志 —— #
logger = logging.getLogger("TextSegmentation")
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# =========================
# 正则 & 工具
# =========================

# 标题必须含中文或字母（避免纯标点/页码行）
HAS_LETTER_RE = re.compile(r"[\u4e00-\u9fa5A-Za-z]")

# 从配置编译正则
CHAPTER_RE     = re.compile(TextSegConfig.CHAPTER_PATTERN)
SECTION_RE     = re.compile(TextSegConfig.SECTION_PATTERN)
SUBSECTION_RE  = re.compile(TextSegConfig.SUBSECTION_PATTERN)
POINT_RE       = re.compile(TextSegConfig.POINT_PATTERN)

CN_NUM_MAP = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "百": 100
}

def cn2int(s: str) -> int:
    s = (s or "").strip()
    if not s:
        return 1
    if s.isdigit():
        return int(s)
    total, tmp, seen = 0, 0, False
    for ch in s:
        if ch == "百":
            total = (total if total else 1) * 100
            tmp = 0; seen = True
        elif ch == "十":
            total += (tmp if tmp else 1) * 10
            tmp = 0; seen = True
        else:
            v = CN_NUM_MAP.get(ch)
            if v is None:
                try:
                    return int(s)
                except Exception:
                    return 1
            tmp = v; seen = True
    if seen:
        total += tmp
        return total or 1
    try:
        return int(s)
    except Exception:
        return 1


def _normalize_spaces(text: str) -> str:
    if not text:
        return ""
    if TextSegConfig.NORMALIZE_SPACES:
        for ch in TextSegConfig.SPACE_SUBSTITUTIONS:
            text = text.replace(ch, " ")
        text = re.sub(r"\s{2,}", " ", text)
    return text


def clean_title(t: str) -> str:
    """去尾部点线+页码、两端空白/冒号；压缩奇异空格。"""
    if not t:
        return ""
    t = _normalize_spaces(t)
    if TextSegConfig.REMOVE_TRAILING_PAGE_NO:
        t = re.sub(r"[\.·・—\-＿\s　]*\d+\s*$", "", t)
    if TextSegConfig.STRIP_COLONS:
        t = t.strip(" ：:　\t")
    return t.strip()


def pick_level_by_style(par: Paragraph) -> Optional[int]:
    """从段落样式/outlineLvl/编号层级推断层级（1..4）。"""
    # 1) 样式名
    try:
        name = (par.style.name or "").strip().lower()
        lvl = TextSegConfig.HEADING_MAP.get(name)
        if lvl:
            return lvl
    except Exception:
        pass

    # 2) outline level
    try:
        el = par._p.xpath("./w:pPr/w:outlineLvl")
        if el:
            lvl = int(el[0].val)
            if 0 <= lvl <= 3:
                return lvl + 1
    except Exception:
        pass

    # 3) 编号层级（ilvl）
    try:
        ilvl = par._p.xpath("./w:pPr/w:numPr/w:ilvl")
        if ilvl:
            lvl = int(ilvl[0].val)
            if 0 <= lvl <= 3:
                return lvl + 1
    except Exception:
        pass

    return None


# =========================
# 解析主逻辑
# =========================

def parse_docx(docx_path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    doc = Document(str(docx_path))
    toc: List[Dict[str, Any]] = []

    cur_ch = None   # level=1
    cur_sec = None  # level=2
    cur_sub = None  # level=3

    warnings: List[str] = []

    for para in doc.paragraphs:
        raw = (para.text or "").strip()
        if not raw:
            continue

        text_norm = _normalize_spaces(raw).strip()
        matched = False

        # —— 样式优先 —— #
        lvl = pick_level_by_style(para) if TextSegConfig.USE_STYLE_FIRST else None

        if lvl == 1 and TextSegConfig.MAX_LEVEL >= 1:
            m = CHAPTER_RE.match(text_norm)
            if m:
                num_raw, title = m.group(1), clean_title(m.group(2))
                if HAS_LETTER_RE.search(title):
                    node = {"level": 1, "id": f"{cn2int(num_raw)}章", "title": title, "children": []}
                    toc.append(node)
                    cur_ch, cur_sec, cur_sub = node, None, None
                    matched = True

        elif lvl == 2 and cur_ch and TextSegConfig.MAX_LEVEL >= 2:
            m = SECTION_RE.match(text_norm)
            if m:
                full, ch, sec, title = m.group(1), m.group(2), m.group(3), clean_title(m.group(4))
                if HAS_LETTER_RE.search(title):
                    exp_ch = str(int(cur_ch["id"][:-1]))
                    if ch == exp_ch:
                        node = {"level": 2, "id": full, "title": title, "children": []}
                        cur_ch["children"].append(node)
                        cur_sec, cur_sub = node, None
                        matched = True
                    else:
                        warnings.append(f"[节章号不一致] 期望{exp_ch}.x，实际{full} —— 段落：{text_norm}")

        elif lvl == 3 and cur_sec and TextSegConfig.MAX_LEVEL >= 3:
            m = SUBSECTION_RE.match(text_norm)
            if m:
                full, title = m.group(1), clean_title(m.group(5))
                if HAS_LETTER_RE.search(title) and full.startswith(cur_sec["id"] + "."):
                    node = {"level": 3, "id": full, "title": title, "children": []}
                    cur_sec["children"].append(node)
                    cur_sub = node
                    matched = True
                else:
                    warnings.append(f"[小节前缀不匹配] 期望前缀{cur_sec['id']}., 实际{full}")

        elif lvl == 4 and cur_sub and TextSegConfig.MAX_LEVEL >= 4:
            m = POINT_RE.match(text_norm)
            if m:
                full, title = m.group(1), clean_title(m.group(6))
                if HAS_LETTER_RE.search(title) and full.startswith(cur_sub["id"] + "."):
                    node = {"level": 4, "id": full, "title": title}
                    cur_sub["children"].append(node)
                    matched = True
                else:
                    warnings.append(f"[知识点前缀不匹配] 期望前缀{cur_sub['id']}., 实际{full}")

        if matched:
            continue

        # —— 正则回退 —— #
        if not TextSegConfig.ENABLE_REGEX_FALLBACK:
            continue

        # 章
        m = CHAPTER_RE.match(text_norm)
        if m and TextSegConfig.MAX_LEVEL >= 1:
            num_raw, title = m.group(1), clean_title(m.group(2))
            if HAS_LETTER_RE.search(title):
                node = {"level": 1, "id": f"{cn2int(num_raw)}章", "title": title, "children": []}
                toc.append(node)
                cur_ch, cur_sec, cur_sub = node, None, None
            continue

        # 节
        m = SECTION_RE.match(text_norm)
        if m and cur_ch and TextSegConfig.MAX_LEVEL >= 2:
            full, ch, sec, title = m.group(1), m.group(2), m.group(3), clean_title(m.group(4))
            if HAS_LETTER_RE.search(title):
                exp_ch = str(int(cur_ch["id"][:-1]))
                if ch == exp_ch:
                    node = {"level": 2, "id": full, "title": title, "children": []}
                    cur_ch["children"].append(node)
                    cur_sec, cur_sub = node, None
                else:
                    warnings.append(f"[节章号不一致] 期望{exp_ch}.x，实际{full}")
            continue

        # 小节
        m = SUBSECTION_RE.match(text_norm)
        if m and cur_sec and TextSegConfig.MAX_LEVEL >= 3:
            full, title = m.group(1), clean_title(m.group(5))
            if HAS_LETTER_RE.search(title) and full.startswith(cur_sec["id"] + "."):
                node = {"level": 3, "id": full, "title": title, "children": []}
                cur_sec["children"].append(node)
                cur_sub = node
            else:
                warnings.append(f"[小节前缀不匹配] 期望前缀{cur_sec['id']}., 实际{full}")
            continue

        # 知识点
        m = POINT_RE.match(text_norm)
        if m and cur_sub and TextSegConfig.MAX_LEVEL >= 4:
            full, title = m.group(1), clean_title(m.group(6))
            if HAS_LETTER_RE.search(title) and full.startswith(cur_sub["id"] + "."):
                node = {"level": 4, "id": full, "title": title}
                cur_sub["children"].append(node)
            else:
                warnings.append(f"[知识点前缀不匹配] 期望前缀{cur_sub['id']}., 实际{full}")
            continue

    return toc, warnings


# =========================
# I/O & CLI
# =========================

def save_toc(toc: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding=TextSegConfig.ENCODING) as f:
        json.dump(toc, f, ensure_ascii=False, indent=2)

def save_warnings(warnings: List[str], warn_path: Path) -> None:
    if not TextSegConfig.SAVE_WARNINGS_FILE or not warnings:
        return
    warn_path.parent.mkdir(parents=True, exist_ok=True)
    with warn_path.open("w", encoding=TextSegConfig.ENCODING) as f:
        for w in warnings:
            f.write(w + "\n")

def main():
    parser = argparse.ArgumentParser(description="解析 Word 目录结构（章-节-小节-知识点）")
    parser.add_argument("--docx", type=str, default=str(TextSegConfig.DOCX_PATH),
                        help="输入 Word 文件路径（默认 config/text.py 的 TextSegConfig.DOCX_PATH）")
    parser.add_argument("--out", type=str, default=str(TextSegConfig.TOC_PATH),
                        help="输出 JSON 路径（默认 config/text.py 的 TextSegConfig.TOC_PATH）")
    parser.add_argument("--warn", type=str, default=str(TextSegConfig.WARN_PATH),
                        help="告警日志路径（默认 config/text.py 的 TextSegConfig.WARN_PATH）")
    parser.add_argument("--loglevel", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.loglevel))

    in_path = Path(args.docx)
    if not in_path.exists():
        raise FileNotFoundError(f"未找到输入文件：{in_path}")

    out_path = Path(args.out)
    warn_path = Path(args.warn)

    logger.info(f"开始解析：{in_path}")
    toc, warnings = parse_docx(in_path)
    save_toc(toc, out_path)
    save_warnings(warnings, warn_path)

    logger.info(f"✅ 完成文本分割：{out_path.resolve()}")
    if toc:
        logger.info(f"首章：{toc[0].get('id')}  {toc[0].get('title')}")
    if warnings:
        n = len(warnings)
        logger.warning(f"共有 {n} 条格式告警（非致命）")
        for w in warnings[:TextSegConfig.WARNINGS_PRINT_TOP]:
            logger.warning("  - " + w)
        if n > TextSegConfig.WARNINGS_PRINT_TOP:
            logger.warning("  ...（其余已省略；完整内容见文件：%s）", warn_path.name)

if __name__ == "__main__":
    main()
