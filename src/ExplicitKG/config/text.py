# -*- coding: utf-8 -*-
"""
config/text.py
仅供 TextSegmentation 使用的配置项：输入/输出、正则、样式映射、清洗与行为开关等。
"""
from pathlib import Path
from .config import OUTPUT_DIR

class TextSegConfig:
    # —— I/O（默认值，可被 CLI 覆盖）——
    DOCX_FILENAME: str = "MATLAB完全自学一本通.docx"
    DOCX_PATH: Path = OUTPUT_DIR / DOCX_FILENAME

    TOC_FILENAME: str = "toc_structure.json"
    TOC_PATH: Path = OUTPUT_DIR / TOC_FILENAME

    WARN_FILENAME: str = "toc_warnings.log"
    WARN_PATH: Path = OUTPUT_DIR / WARN_FILENAME

    ENCODING: str = "utf-8"

    # —— 行为策略 ——
    USE_STYLE_FIRST: bool = True             # 样式优先
    ENABLE_REGEX_FALLBACK: bool = True       # 样式失效时回退正则
    MAX_LEVEL: int = 4                       # 最大层级（1=章,2=节,3=小节,4=知识点）
    WARNINGS_PRINT_TOP: int = 10             # 控制台最多打印的告警条数
    SAVE_WARNINGS_FILE: bool = True          # 是否写入告警文件

    # —— 标题清洗 ——
    NORMALIZE_SPACES: bool = True
    REMOVE_TRAILING_PAGE_NO: bool = True
    STRIP_COLONS: bool = True
    SPACE_SUBSTITUTIONS = ["\u00A0", "\u2003", "\u3000"]  # 不同空格类型

    # —— 段落样式映射（大小写不敏感）——
    HEADING_MAP = {
        "heading 1": 1, "heading1": 1, "标题 1": 1, "标题1": 1,
        "heading 2": 2, "heading2": 2, "标题 2": 2, "标题2": 2,
        "heading 3": 3, "heading3": 3, "标题 3": 3, "标题3": 3,
        "heading 4": 4, "heading4": 4, "标题 4": 4, "标题4": 4,
    }

    # —— 正则（仅段首匹配）——
    CHAPTER_PATTERN     = r"^\s*第\s*([一二三四五六七八九十百零〇0-9]+)\s*章[：:\s　]+(.+?)\s*$"
    SECTION_PATTERN     = r"^\s*((\d{1,2})\.(\d{1,2}))\s*[：:\s　]+(.+?)\s*$"                        # 1.1
    SUBSECTION_PATTERN  = r"^\s*((\d{1,2})\.(\d{1,2})\.(\d{1,2}))\s*[：:\s　]+(.+?)\s*$"             # 1.1.1
    POINT_PATTERN       = r"^\s*((\d{1,2})\.(\d{1,2})\.(\d{1,2})\.(\d{1,2}))\s*[：:\s　]+(.+?)\s*$"   # 1.1.1.1
