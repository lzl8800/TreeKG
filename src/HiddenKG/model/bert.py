# -*- coding: utf-8 -*-
"""
HiddenKG/model/bert.py
BERT 编码器模块：
- 加载 tokenizer + model
- 封装 encode_texts()：批处理文本 -> 句向量
- 支持 AMP（仅 CUDA 时启用）/ GPU / L2 归一化
- OOM 自动降批回退
"""

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

        # —— 设备兜底逻辑 —— #
        req_device = (device or "").lower().strip()
        if req_device == "auto":
            req_device = "cuda" if torch.cuda.is_available() else "cpu"

        if req_device.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️ 检测到 PyTorch 未启用 CUDA，设备已从 'cuda' 回退为 'cpu'")
            req_device = "cpu"

        self.device = torch.device(req_device)
        # 仅在 CUDA 可用时启用 AMP
        self.use_amp = bool(use_amp and self.device.type == "cuda")

        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        # 加载权重（可用本地目录或 huggingface 名称）
        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertModel.from_pretrained(model_dir)
        self.model.to(self.device).eval()

        print(f"✅ BERT 模型加载完成：{model_dir} ({self.device})  AMP={self.use_amp}")

    @staticmethod
    def _masked_mean_pool(last_hidden: Tensor, attn_mask: Tensor) -> Tensor:
        """按注意力掩码对 token 向量做平均池化"""
        mask = attn_mask.unsqueeze(-1)  # [B, L, 1]
        summed = (last_hidden * mask).sum(dim=1)         # [B, H]
        counts = mask.sum(dim=1).clamp(min=1e-9)         # [B, 1]
        return summed / counts

    @staticmethod
    def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """L2 归一化（数值安全）"""
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
        - 自动处理分批、截断、AMP（仅 CUDA）
        - OOM 自动降批
        """
        texts = list(texts or [])
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_vecs = []
        i = 0
        bs = max(1, int(self.batch_size))

        pbar = tqdm(total=len(texts), desc="BERT编码", ncols=80)
        while i < len(texts):
            batch = texts[i:i + bs]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}

            try:
                if self.use_amp:
                    # 仅 CUDA 可用 autocast
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        out = self.model(**enc)
                else:
                    out = self.model(**enc)
            except RuntimeError as e:
                # 显存不足→清缓存并降批重试；其他错误抛出
                if "out of memory" in str(e).lower() and self.device.type == "cuda" and bs > 1:
                    torch.cuda.empty_cache()
                    bs = max(1, bs // 2)
                    print(f"⚠️ 检测到 OOM，批大小回退为 {bs}，重试该批。")
                    continue
                else:
                    raise

            hidden = out.last_hidden_state            # [B, L, H]
            pooled = self._masked_mean_pool(hidden, enc["attention_mask"])  # [B, H]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)      # 先做一次 L2
            all_vecs.append(pooled.detach().cpu().float().numpy())

            i += len(batch)
            pbar.update(len(batch))

        pbar.close()

        mat = np.vstack(all_vecs).astype(np.float32)  # [N, H]
        return self._l2_normalize(mat)                # 再做一次整体 L2，保证稳健性
