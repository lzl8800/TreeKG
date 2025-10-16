# -*- coding: utf-8 -*-
"""
HiddenKG/model/main.py
统一模型入口，方便未来拓展多编码器（BERT, RoBERTa, SimCSE, LLaMA等）
"""

from .bert import BertEncoder

def get_encoder(model_name: str = "bert-base-chinese", **kwargs):
    """
    获取指定名称的文本编码器。
    支持：bert-base-chinese（默认）
    """
    model_name = model_name.lower()
    if "bert" in model_name:
        return BertEncoder(model_dir=model_name, **kwargs)
    else:
        raise ValueError(f"暂不支持的模型名称：{model_name}")
