"""默认任务分配模板"""

from typing import Dict, List, Optional

from model_ir_parser.ir import Cell, Task
from ..task_mapping import ParallelTemplate
from ..schedule import TaskMapping
from ..utils import CollectiveComm, CommType


class DefaultTemplate(ParallelTemplate):
    """默认模板：将所有 Task 均匀分配到设备

    参考: apex_plus/parallel/templates/default.py
    """

    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        """均匀分配 Cell 的所有 Task 到设备

        Args:
            cell: 要分配的 Cell
            num_devices: 设备数量

        Returns:
            TaskMapping 或 None（如果任务数不足）
        """
        # 按类型分组所有 task
        tasks_per_type: Dict[str, List[Task]] = {}
        for task in cell.get_tasks():
            task_type = task.get_type()
            if task_type not in tasks_per_type:
                tasks_per_type[task_type] = []
            tasks_per_type[task_type].append(task)

        # 初始化每个设备的任务字典
        tasks_per_device: List[Dict[str, List[Task]]] = []
        for _ in range(num_devices):
            tasks_per_device.append({})

        # 对每种类型的 task，均匀分配到设备
        for task_type, tasks in tasks_per_type.items():
            num_tasks = len(tasks)

            # 计算每个设备应分配的任务数
            num_tasks_per_device = [num_tasks // num_devices] * num_devices
            for i in range(num_tasks % num_devices):
                num_tasks_per_device[i] += 1

            # 分配任务
            start = 0
            for i in range(num_devices):
                end = start + num_tasks_per_device[i]
                tasks_per_device[i][task_type] = tasks[start:end]
                start = end

        # 检查最后一个设备是否为空（任务数不足的情况）
        if all(len(task_dict) == 0 or all(len(v) == 0 for v in task_dict.values())
               for task_dict in tasks_per_device[-1:]):
            return None

        # 创建 TaskMapping，使用 AllReduce 作为默认通信方式
        task_mapping = TaskMapping(
            tasks_per_device,
            CollectiveComm(CommType.AllReduce, num_devices)
        )

        return task_mapping
