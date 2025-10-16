# -*- coding: utf-8 -*-
"""
Embedding.py
使用抽象模型接口生成实体嵌入（支持批处理、AMP、L2归一化、PKL存储）
"""

import os
import json
import pickle
import argparse
from typing import Dict, List
import numpy as np

# ===== 导入配置与模型入口 =====
try:
    from HiddenKG.config import Emb as EmbConfig
    from HiddenKG.model.main import get_encoder
except Exception:
    EmbConfig = None  # type: ignore
    from model.main import get_encoder  # 独立运行时路径兜底


# ================ 实体读取与文本提取 ================
def load_entities(file_path: str) -> Dict[str, dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📘 已加载实体 {len(data)} 个：{os.path.basename(file_path)}")
    return data


def entity_text(e: dict) -> str:
    return (e.get("updated_description") or e.get("original") or "").strip() or "[EMPTY]"


# ================ 生成嵌入 ================
def generate_embeddings(encoder, entities: Dict[str, dict]) -> Dict[str, np.ndarray]:
    items = sorted(entities.items(), key=lambda kv: kv[0])
    names: List[str] = [k for k, _ in items]
    texts: List[str] = [entity_text(v) for _, v in items]

    mat = encoder.encode_texts(texts)
    name_to_vec = {n: mat[i] for i, n in enumerate(names)}

    print(f"✅ 已生成嵌入向量：{mat.shape[0]} × {mat.shape[1]}")
    return name_to_vec


# ================ 保存 ================
def save_embeddings_pkl(name_to_vec: Dict[str, np.ndarray], out_file: str):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(name_to_vec, f)
    print(f"💾 已保存嵌入文件：{out_file}")


# ================ 主入口 ================
def main():
    # 从配置加载默认参数
    if EmbConfig:
        EmbConfig.ensure_paths()
        infile = os.fspath(EmbConfig.INPUT_FILE)
        outfile = os.fspath(EmbConfig.OUTPUT_FILE)
        batch_size = EmbConfig.BATCH_SIZE
        max_length = EmbConfig.MAX_LENGTH
        use_amp = EmbConfig.USE_AMP
        model_dir = os.fspath(EmbConfig.BERT_BASE_DIR)
        device = EmbConfig.DEVICE
    else:
        infile = os.getenv("EMB_INPUT", "HiddenKG/output/conv_entities.json")
        outfile = os.getenv("EMB_OUTPUT", "HiddenKG/output/node_embeddings.pkl")
        batch_size = int(os.getenv("EMB_BATCH_SIZE", "16"))
        max_length = int(os.getenv("EMB_MAX_LENGTH", "512"))
        use_amp = bool(int(os.getenv("EMB_USE_AMP", "1")))
        model_dir = os.getenv("EMB_BERT_DIR", "bert-base-chinese")
        device = os.getenv("EMB_DEVICE", "cuda")

    parser = argparse.ArgumentParser(description="生成实体嵌入向量")
    parser.add_argument("--infile", type=str, default=infile)
    parser.add_argument("--outfile", type=str, default=outfile)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--max_length", type=int, default=max_length)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    # 初始化模型
    encoder = get_encoder(
        model_name=model_dir,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_amp=not args.no_amp and use_amp,
    )

    # 生成并保存
    entities = load_entities(args.infile)
    name_to_vec = generate_embeddings(encoder, entities)
    save_embeddings_pkl(name_to_vec, args.outfile)


if __name__ == "__main__":
    main()
