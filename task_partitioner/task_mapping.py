"""任务并行模板基类"""

from abc import ABC, abstractmethod
from typing import Optional

from model_ir_parser.ir import Cell
from .schedule import TaskMapping


class ParallelTemplate(ABC):
    """任务并行模板抽象基类

    负责将一个 Cell 内的所有 Task 分配到多个设备上
    """

    @staticmethod
    @abstractmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        """将 Cell 的任务映射到设备

        Args:
            cell: 要分配的 Cell
            num_devices: 可用设备数量

        Returns:
            TaskMapping 如果分配成功，否则返回 None
            (例如当 task 数量少于设备数时返回 None)
        """
        raise NotImplementedError
