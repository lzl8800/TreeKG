# -*- coding: utf-8 -*-
"""
Embedding.py
使用抽象模型接口生成实体嵌入（支持批处理、AMP、L2归一化、PKL存储）
"""

import os
import json
import pickle
import argparse
import logging
from typing import Dict, List
import numpy as np
from pathlib import Path
import yaml
from log_utils import setup_stage_logger


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "output" / "02_hidden_kg"
logger = setup_stage_logger("embedding", OUTPUT_DIR, console_level=logging.INFO)

# ====== 模型入口（保持原有） ======
try:
    from HiddenKG.model.main import get_encoder
except Exception:
    from model.main import get_encoder  # 独立运行时兜底

# ====== 配置加载 ======
def _safe_load_yaml(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def load_config(config_path: Path) -> dict:
    """
    读取 HiddenKG/config/config.yaml，并按 include_files 依次合并（相对 HiddenKG/config/）
    """
    base = _safe_load_yaml(config_path)
    merged = dict(base)

    includes = (base.get("include_files") or []) if isinstance(base, dict) else []
    for inc in includes:
        inc_path = (config_path.parent / inc).resolve()
        if not inc_path.exists():
            # 兼容写成 "config/xxx.yaml" 的老习惯
            alt = (config_path.parent / "config" / inc).resolve()
            if alt.exists():
                inc_path = alt
        if not inc_path.exists():
            raise FileNotFoundError(f"include 文件未找到：{inc} -> {inc_path}")
        part = _safe_load_yaml(inc_path)
        merged.update(part or {})
    return merged

# ====== 文本处理 ======
def load_entities(file_path: str) -> Dict[str, dict]:
    logger.info("Loading entities for embedding: %s", Path(file_path).resolve())
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded entities count=%s", len(data))
    print(f"📘 已加载实体 {len(data)} 个：{os.path.basename(file_path)}")
    return data

def entity_text(e: dict) -> str:
    return (e.get("updated_description") or e.get("original") or "").strip() or "[EMPTY]"

# ====== 生成嵌入 ======
def generate_embeddings(encoder, entities: Dict[str, dict]) -> Dict[str, np.ndarray]:
    items = sorted(entities.items(), key=lambda kv: kv[0])
    names: List[str] = [k for k, _ in items]
    texts: List[str] = [entity_text(v) for _, v in items]

    logger.info("Generating embeddings. entities=%s", len(items))
    mat = encoder.encode_texts(texts)
    name_to_vec = {n: mat[i] for i, n in enumerate(names)}

    print(f"✅ 已生成嵌入向量：{mat.shape[0]} × {mat.shape[1]}")
    logger.info("Generated embeddings shape=%s x %s", mat.shape[0], mat.shape[1])
    return name_to_vec

def save_embeddings_pkl(name_to_vec: Dict[str, np.ndarray], out_file: str):
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(name_to_vec, f)
    logger.info("Saved embeddings: %s, count=%s", Path(out_file).resolve(), len(name_to_vec))
    print(f"💾 已保存嵌入文件：{out_file}")

# ====== 主入口 ======
def main():
    # 基准目录：HiddenKG/
    script_dir = SCRIPT_DIR
    cfg_root = script_dir / "config"
    cfg_path = cfg_root / "config.yaml"

    config = load_config(cfg_path)
    emb = config.get("EmbConfig", {})  # 来自 emb.yaml

    # —— 自动拼接路径 —— #
    # 输入/输出都在统一输出根目录的隐式层目录下
    output_dir = OUTPUT_DIR
    input_path  = output_dir / emb.get("INPUT_NAME", "conv_entities.json")
    output_path = output_dir / emb.get("OUTPUT_NAME", "node_embeddings.pkl")

    # BERT 目录在 HiddenKG/model/<BERT_DIR_NAME>
    bert_dir = script_dir / "model" / emb.get("BERT_DIR_NAME", "bert-base-chinese")

    # 运行参数
    batch_size = int(emb.get("BATCH_SIZE", 16))
    max_length = int(emb.get("MAX_LENGTH", 512))
    use_amp    = bool(emb.get("USE_AMP", True))
    device     = emb.get("DEVICE", "cuda")

    # CLI 覆盖（可选）
    parser = argparse.ArgumentParser(description="生成实体嵌入向量")
    parser.add_argument("--infile", type=str, default=str(input_path))
    parser.add_argument("--outfile", type=str, default=str(output_path))
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--max_length", type=int, default=max_length)
    parser.add_argument("--no_amp", action="store_true", help="禁用 AMP（半精度）")
    args = parser.parse_args()

    logger.info(
        "Embedding started. infile=%s, outfile=%s, model=%s, device=%s, batch_size=%s, max_length=%s, amp=%s",
        Path(args.infile).resolve(),
        Path(args.outfile).resolve(),
        bert_dir,
        device,
        args.batch_size,
        args.max_length,
        (not args.no_amp) and use_amp,
    )
    try:
        encoder = get_encoder(
            model_name=str(bert_dir),
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_amp=(not args.no_amp) and use_amp,
        )
        entities = load_entities(args.infile)
        name_to_vec = generate_embeddings(encoder, entities)
        save_embeddings_pkl(name_to_vec, args.outfile)
        logger.info("Embedding finished. log=%s", logger.log_path)
    except Exception:
        logger.exception("Embedding failed.")
        raise

if __name__ == "__main__":
    main()
