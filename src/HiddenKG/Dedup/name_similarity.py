import re
from typing import List  # 新增导入

def _is_cjk(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s))

# 修改返回值注解：list[str] -> List[str]
def _name_tokens_token(s: str) -> List[str]:
    s = _normalize_name(s)
    return [t for t in re.split(r"[\s·\-\(\)（），,、：:]+", s) if t]

# 修改返回值注解：list[str] -> List[str]
def _name_tokens_char(s: str) -> List[str]:
    s = _normalize_name(s)
    # 仅保留字母数字与汉字
    s = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", s)
    return list(s)

# 修改返回值注解：list[str] -> List[str]
def _name_tokens_bigram(s: str) -> List[str]:
    toks = _name_tokens_char(s)
    if len(toks) < 2:
        return toks
    return [toks[i] + toks[i + 1] for i in range(len(toks) - 1)]

def _name_jaccard_mode(a: str, b: str, mode: str = "auto") -> float:
    """
    名称相似度 Jaccard：
      mode: token | char | bigram | auto
      auto: 若任一侧包含CJK汉字，采用 bigram；否则 token。
    """
    if mode == "auto":
        mode = "bigram" if (_is_cjk(a) or _is_cjk(b)) else "token"
    if mode == "token":
        A, B = set(_name_tokens_token(a)), set(_name_tokens_token(b))
    elif mode == "char":
        A, B = set(_name_tokens_char(a)), set(_name_tokens_char(b))
    elif mode == "bigram":
        A, B = set(_name_tokens_bigram(a)), set(_name_tokens_bigram(b))
    else:
        A, B = set(_name_tokens_token(a)), set(_name_tokens_token(b))
    if not A and not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

def _normalize_name(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s