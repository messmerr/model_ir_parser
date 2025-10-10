"""模板注册与获取"""

from typing import List

from model_ir_parser.ir import Cell
from ..task_mapping import ParallelTemplate
from .default import DefaultTemplate
from .attention import MQATemplate0, MHATemplate0


# Cell 名称到模板列表的映射
TEMPLATE_REGISTRY = {
    "MHA": [MHATemplate0],
    "MQA": [MQATemplate0],
    "SwiGLU": [DefaultTemplate],
    "GLU": [DefaultTemplate],
    "MLP": [DefaultTemplate],
}


def get_templates(cell: Cell) -> List[ParallelTemplate]:
    """获取 Cell 对应的所有分配模板

    Args:
        cell: Cell 实例

    Returns:
        模板列表，如果未注册则返回 [DefaultTemplate]
    """
    cell_name = cell.get_name()
    return TEMPLATE_REGISTRY.get(cell_name, [DefaultTemplate])
