"""模型登记占位模块。"""


def get_active_model_info() -> dict[str, object]:
    """第一阶段没有可用 reranker 模型。"""
    return {"model_loaded": False, "placeholder": True}
