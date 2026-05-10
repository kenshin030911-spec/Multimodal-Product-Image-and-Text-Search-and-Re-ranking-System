"""reranker 训练相关模块。"""

__all__ = ["build_reranker_dataset"]


def __getattr__(name: str):
    """Lazily expose training helpers without creating import cycles."""
    if name == "build_reranker_dataset":
        from backend.app.training.sample_builder import build_reranker_dataset

        return build_reranker_dataset
    raise AttributeError(name)
