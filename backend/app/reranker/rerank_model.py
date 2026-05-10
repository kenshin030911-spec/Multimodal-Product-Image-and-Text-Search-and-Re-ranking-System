"""规则重排模型。"""

from __future__ import annotations

from typing import Sequence

from backend.app.reranker.feature_builder import RerankFeature
from backend.app.schemas.rerank import RerankFeature as SchemaRerankFeature
from backend.app.schemas.rerank import RerankScore


RECALL_WEIGHT = 1.00
METADATA_MATCH_WEIGHT = 0.20
FRESHNESS_WEIGHT = 0.05


class RuleBasedReranker:
    """轻量、可解释的规则 reranker baseline。"""

    def score_feature(self, feature: RerankFeature) -> float:
        """按固定权重计算单条候选的重排分。"""
        return (
            RECALL_WEIGHT * feature.recall_score
            + METADATA_MATCH_WEIGHT * feature.metadata_match_score
            + FRESHNESS_WEIGHT * feature.freshness_score
        )

    def score(self, features: Sequence[RerankFeature]) -> dict[str, float]:
        """批量计算重排分，分数不裁剪到 [0, 1]。"""
        return {feature.product_id: self.score_feature(feature) for feature in features}


class PlaceholderReranker:
    """占位 reranker，保留给旧调用和旧测试。"""

    def score(self, features: list[SchemaRerankFeature]) -> list[RerankScore]:
        """旧占位模型不做真实打分，固定返回空结果。"""
        _ = features
        return []
