"""Task Partitioner - 任务分区模块

硬件无关的任务划分枚举引擎，基于 Transformer IR 生成所有可行的并行执行方案。
"""

from .engine import PartitionEngine
from .schedule import (
    TaskMapping,
    CellSchedule,
    StageSchedule,
    ParallelSchedule
)
from .utils import CommType, CollectiveComm

__all__ = [
    "PartitionEngine",
    "TaskMapping",
    "CellSchedule",
    "StageSchedule",
    "ParallelSchedule",
    "CommType",
    "CollectiveComm",
]

__version__ = "0.1.0"
