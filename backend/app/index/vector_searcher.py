"""NumPy flat cosine 向量检索。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from backend.app.index.index_builder import l2_normalize_matrix
from backend.app.index.index_store import IndexBundle, load_index_bundle


@dataclass(frozen=True)
class VectorSearchResult:
    """向量召回结果，不限制 cosine 分数范围。"""

    product_id: str
    score: float
    rank: int
    embedding_index: int


class VectorSearcher:
    """对 image_index 执行 flat cosine top-k 检索。"""

    def __init__(self, bundle: IndexBundle) -> None:
        self.image_index = bundle.image_index
        self.meta = bundle.meta
        self.product_ids = list(bundle.meta.get("product_ids", []))
        self.product_id_to_index = dict(bundle.meta.get("product_id_to_index", {}))
        self.embedding_dim = int(bundle.meta.get("embedding_dim", 0))

    @classmethod
    def from_index_dir(cls, index_dir: Path | None = None) -> "VectorSearcher":
        """从 index 目录加载 searcher。"""
        return cls(load_index_bundle(index_dir))

    def search(
        self,
        query_vector: Sequence[float] | np.ndarray,
        top_k: int = 20,
        exclude_product_id: str | None = None,
        exclude_embedding_index: int | None = None,
    ) -> list[VectorSearchResult]:
        """对 query_vector 做 cosine top-k，支持排除自身。"""
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0。")

        query = _prepare_query_vector(query_vector, self.embedding_dim)
        query = l2_normalize_matrix(query.reshape(1, -1))[0]
        excluded_indices = self._resolve_excluded_indices(
            exclude_product_id=exclude_product_id,
            exclude_embedding_index=exclude_embedding_index,
        )

        scores = self.image_index @ query
        candidate_indices = [
            index for index in range(len(self.product_ids)) if index not in excluded_indices
        ]
        if not candidate_indices:
            return []

        candidate_scores = scores[candidate_indices]
        order = np.argsort(-candidate_scores, kind="stable")
        selected = order[: min(top_k, len(order))]

        results: list[VectorSearchResult] = []
        for rank, order_index in enumerate(selected, start=1):
            embedding_index = int(candidate_indices[int(order_index)])
            results.append(
                VectorSearchResult(
                    product_id=self.product_ids[embedding_index],
                    score=float(scores[embedding_index]),
                    rank=rank,
                    embedding_index=embedding_index,
                )
            )
        return results

    def _resolve_excluded_indices(
        self,
        exclude_product_id: str | None,
        exclude_embedding_index: int | None,
    ) -> set[int]:
        """把 product_id 和 embedding_index 排除项统一成 index 集合。"""
        excluded_indices: set[int] = set()
        if exclude_product_id is not None:
            if exclude_product_id not in self.product_id_to_index:
                raise ValueError(f"exclude_product_id 不存在于 index: {exclude_product_id}")
            excluded_indices.add(int(self.product_id_to_index[exclude_product_id]))
        if exclude_embedding_index is not None:
            if exclude_embedding_index < 0 or exclude_embedding_index >= len(self.product_ids):
                raise ValueError("exclude_embedding_index 超出 index 范围。")
            excluded_indices.add(exclude_embedding_index)
        return excluded_indices


def search_vectors(
    query_vector: Sequence[float] | np.ndarray,
    index_dir: Path | None = None,
    top_k: int = 20,
    exclude_product_id: str | None = None,
    exclude_embedding_index: int | None = None,
) -> list[VectorSearchResult]:
    """便捷函数：加载 index 并执行一次检索。"""
    return VectorSearcher.from_index_dir(index_dir).search(
        query_vector=query_vector,
        top_k=top_k,
        exclude_product_id=exclude_product_id,
        exclude_embedding_index=exclude_embedding_index,
    )


def search_vectors_placeholder() -> list[VectorSearchResult]:
    """兼容旧调用：真实检索请使用 VectorSearcher。"""
    return []


def _prepare_query_vector(query_vector: Sequence[float] | np.ndarray, embedding_dim: int) -> np.ndarray:
    """把 query vector 规整成一维 float32，并校验维度。"""
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim == 2 and query.shape[0] == 1:
        query = query[0]
    if query.ndim != 1:
        raise ValueError("query_vector 必须是一维向量，或 shape 为 (1, embedding_dim)。")
    if query.shape[0] != embedding_dim:
        raise ValueError("query_vector 维度与 index embedding_dim 不一致。")
    return query.astype(np.float32, copy=False)
