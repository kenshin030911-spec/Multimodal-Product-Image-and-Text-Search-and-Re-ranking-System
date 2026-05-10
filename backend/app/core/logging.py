"""日志配置。"""

import logging


def configure_logging() -> None:
    """设置基础日志格式，第一阶段不引入复杂日志框架。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
