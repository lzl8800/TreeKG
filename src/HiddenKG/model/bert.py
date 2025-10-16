# -*- coding: utf-8 -*-
"""
HiddenKG/model/bert_model.py
BERT 编码器模块：
- 加载 tokenizer + model
- 封装 encode_texts()：批处理文本 -> 句向量
- 支持 AMP / GPU / L2 归一化
"""

import os
import numpy as np
import torch
from torch import Tensor
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

class BertEncoder:
    def __init__(
        self,
        model_dir: str = "bert-base-chinese",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        max_length: int = 512,
        batch_size: int = 16,
    ):
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.use_amp = use_amp
        self.max_length = max_length
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        print(f"✅ BERT 模型加载完成：{model_dir} ({self.device})")

    @staticmethod
    def _masked_mean_pool(last_hidden: Tensor, attn_mask: Tensor) -> Tensor:
        mask = attn_mask.unsqueeze(-1)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        if x.ndim == 1:
            denom = np.linalg.norm(x) + eps
            return (x / denom).astype(np.float32)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + eps
        return (x / denom).astype(np.float32)

    @torch.inference_mode()
    def encode_texts(self, texts):
        """
        输入：List[str]
        输出：np.ndarray [N, hidden_dim]
        """
        all_vecs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="BERT编码", ncols=80):
            batch = texts[i:i + self.batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            try:
                if self.use_amp and self.device.type == "cuda":
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = self.model(**enc)
                else:
                    out = self.model(**enc)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and self.device.type == "cuda" and self.batch_size > 1:
                    torch.cuda.empty_cache()
                    self.batch_size = max(1, self.batch_size // 2)
                    print(f"⚠️ OOM 回退：batch_size -> {self.batch_size}")
                    return self.encode_texts(texts)
                else:
                    raise

            hidden = out.last_hidden_state
            pooled = self._masked_mean_pool(hidden, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.detach().cpu().float().numpy())

        mat = np.vstack(all_vecs).astype(np.float32)
        return self._l2_normalize(mat)
