"""调度方案数据结构定义"""

from dataclasses import dataclass
from typing import Dict, List

from model_ir_parser.ir import Block, Cell, Task
from .utils import CollectiveComm


@dataclass
class TaskMapping:
    """Level 4: Task 到设备的映射

    Attributes:
        tasks_per_device: 每个设备上的任务列表
                         格式: List[Dict[task_type -> List[Task]]]
        collective_comm: 该映射对应的集合通信操作
    """
    tasks_per_device: List[Dict[str, List[Task]]]
    collective_comm: CollectiveComm

    def get_num_devices(self) -> int:
        """获取设备数量"""
        return len(self.tasks_per_device)

    def __repr__(self) -> str:
        if not self.tasks_per_device:
            return "TaskMapping(empty)"

        # 统计第一个设备上的任务类型和数量
        first_device_tasks = self.tasks_per_device[0]
        task_counts = {k: len(v) for k, v in first_device_tasks.items()}

        return f"TaskMapping(tasks_per_device={task_counts})x{self.get_num_devices()}"


@dataclass
class CellSchedule:
    """Level 3: Cell 调度方案

    Attributes:
        cell: Cell 实例
        num_replicas: Cell 的副本数（数据并行）
        task_mapping: Task 到设备的映射
    """
    cell: Cell
    num_replicas: int
    task_mapping: TaskMapping

    def get_num_devices(self) -> int:
        """获取该 Cell 占用的总设备数"""
        return self.num_replicas * self.task_mapping.get_num_devices()

    def __repr__(self) -> str:
        return (
            f"CellSchedule({self.cell.get_name()}, "
            f"{self.num_replicas} replicas, "
            f"{self.task_mapping})"
        )


@dataclass
class StageSchedule:
    """Level 2: Stage 调度方案

    Attributes:
        block: Block 实例（包含多个 Cell）
        num_blocks: 该 stage 包含的 block 数量
        cell_schedules: 每个 Cell 的调度方案
        reshard_comms: Cell 间的数据重分布通信
                      格式: List[List[CollectiveComm]]
                      reshard_comms[i] 表示从 cell[i] 到 cell[i+1] 的通信
    """
    block: Block
    num_blocks: int
    cell_schedules: List[CellSchedule]
    reshard_comms: List[List[CollectiveComm]]

    def __post_init__(self):
        """验证数据一致性"""
        num_cells = len(self.block.cells)
        if len(self.cell_schedules) != num_cells:
            raise ValueError(
                f"cell_schedules length {len(self.cell_schedules)} "
                f"does not match block cells {num_cells}"
            )
        if len(self.reshard_comms) != num_cells:
            raise ValueError(
                f"reshard_comms length {len(self.reshard_comms)} "
                f"does not match block cells {num_cells}"
            )

    def __repr__(self) -> str:
        msg = f"StageSchedule({self.num_blocks} blocks)\n"
        for i, cs in enumerate(self.cell_schedules):
            msg += f"  Cell[{i}]: {cs}\n"
            if self.reshard_comms[i]:
                for comm in self.reshard_comms[i]:
                    msg += f"  Reshard: {comm}\n"
        return msg


@dataclass
class ParallelSchedule:
    """Level 1: 完整的并行执行方案

    Attributes:
        num_model_replicas: 模型副本数（数据并行）
        num_stages: Pipeline 阶段数
        stage_schedule: Stage 内部的调度方案
    """
    num_model_replicas: int
    num_stages: int
    stage_schedule: StageSchedule

    def get_total_devices(self) -> int:
        """获取该方案占用的总设备数"""
        stage_devices = self.stage_schedule.cell_schedules[0].get_num_devices()
        return self.num_model_replicas * self.num_stages * stage_devices

    def __repr__(self) -> str:
        return (
            f"ParallelSchedule(\n"
            f"  num_model_replicas={self.num_model_replicas},\n"
            f"  num_stages={self.num_stages},\n"
            f"  stage_schedule={self.stage_schedule}"
            f")"
        )
