"""评估相关数据结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalQuery(BaseModel):
    """弱监督或少量人工评估查询样本。"""

    query_id: str
    query_type: Literal["text"] = "text"
    query: str | None = Field(default=None, description="兼容旧字段的文本查询")
    query_text: str | None = Field(default=None, description="文本查询")
    expected_article_type: str | None = None
    expected_base_colour: str | None = None
    expected_gender: str | None = None
    expected_sub_category: str | None = None
    positive_product_ids: list[str] = Field(default_factory=list)
    source_product_id: str | None = None
    label_source: Literal["manual", "weak_metadata"] = "manual"
    query_template_name: str | None = None
    query_template_fields: list[str] = Field(default_factory=list)
    query_generation_mode: str | None = None

    @model_validator(mode="after")
    def fill_query_aliases(self) -> "EvalQuery":
        """允许 eval_queries.jsonl 使用 query 或 query_text。"""
        query_text = self.query_text or self.query
        if query_text is None or not query_text.strip():
            raise ValueError("query_text 不能为空。")
        self.query_text = query_text.strip()
        self.query = query_text.strip()
        if not self.positive_product_ids and self.label_source == "manual":
            self.label_source = "weak_metadata"
        return self


class EvaluationMetrics(BaseModel):
    """重排前后对比指标；保留旧字段并允许不同 K 的扩展字段。"""

    model_config = ConfigDict(extra="allow")

    precision_at_10_before_rerank: float = 0.0
    precision_at_10_after_rerank: float = 0.0
    recall_at_10_before_rerank: float = 0.0
    recall_at_10_after_rerank: float = 0.0
    hit_at_10_before_rerank: float = 0.0
    hit_at_10_after_rerank: float = 0.0
    mrr_before_rerank: float = 0.0
    mrr_after_rerank: float = 0.0
    ndcg_at_10_before_rerank: float = 0.0
    ndcg_at_10_after_rerank: float = 0.0


class EvaluationSummaryResponse(BaseModel):
    """评估摘要接口响应。"""

    metrics: EvaluationMetrics = Field(default_factory=EvaluationMetrics)
    prepared: bool = False
    placeholder: bool = True
    generated_at: str | None = None
    query_count: int = 0
    metric_k: int | None = None
    candidate_k: int | None = None
    max_queries: int | None = None
    eval_source: str | None = None
    label_policy: str | None = None
    query_generation_mode: str | None = None
    query_templates: str | None = None
    queries_per_product: int | None = None
    max_query_variants: int | None = None
    query_template_names: list[str] = Field(default_factory=list)
    query_generation_note: str | None = None
    include_trained_reranker: bool = False
    include_pairwise_reranker: bool = False
    vector_recall: dict[str, float] = Field(default_factory=dict)
    rule_rerank: dict[str, float] = Field(default_factory=dict)
    binary_trained_rerank: dict[str, float] = Field(default_factory=dict)
    trained_rerank: dict[str, float] = Field(default_factory=dict)
    pairwise_rerank: dict[str, float] = Field(default_factory=dict)
    delta_rule_vs_vector: dict[str, float] = Field(default_factory=dict)
    delta_binary_trained_vs_vector: dict[str, float] = Field(default_factory=dict)
    delta_binary_trained_vs_rule: dict[str, float] = Field(default_factory=dict)
    delta_trained_vs_vector: dict[str, float] = Field(default_factory=dict)
    delta_trained_vs_rule: dict[str, float] = Field(default_factory=dict)
    delta_pairwise_vs_vector: dict[str, float] = Field(default_factory=dict)
    delta_pairwise_vs_rule: dict[str, float] = Field(default_factory=dict)
    delta_pairwise_vs_binary_trained: dict[str, float] = Field(default_factory=dict)
    before_rerank: dict[str, float] = Field(default_factory=dict)
    after_rerank: dict[str, float] = Field(default_factory=dict)
    delta: dict[str, float] = Field(default_factory=dict)
    summary_path: str | None = None
    details_path: str | None = None
    text_report_path: str | None = None
    message: str = "第一阶段占位评估结果，尚未运行真实评估。"
