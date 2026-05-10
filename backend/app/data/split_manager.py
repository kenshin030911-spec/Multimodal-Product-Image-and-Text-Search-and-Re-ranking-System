"""训练、验证、测试集划分占位模块。"""


def get_dataset_splits() -> dict[str, list[str]]:
    """第一阶段不读取真实数据，返回空划分。"""
    return {"train": [], "val": [], "test": []}
