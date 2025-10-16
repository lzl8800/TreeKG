import numpy as np
from typing import Tuple  # 新增导入

# 优先使用 FAISS；不可用则退化到 sklearn
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors  # type: ignore

def knn_topk(embs: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:  # 修改返回值注解
    """
    返回 (dist, idx) 形状为 (N, topk)。使用 L2 距离。
    """
    n, dim = embs.shape
    if HAS_FAISS:
        index = faiss.IndexFlatL2(dim)
        index.add(embs)
        dists, idxs = index.search(embs, topk)
    else:
        nbrs = NearestNeighbors(n_neighbors=topk, algorithm="auto", metric="euclidean")
        nbrs.fit(embs)
        dists, idxs = nbrs.kneighbors(embs)
    return dists, idxs