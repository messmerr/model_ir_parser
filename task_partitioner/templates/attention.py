"""注意力层任务分配模板"""

from typing import Dict, List, Optional

from model_ir_parser.ir import Cell, Task, MQA, MHA
from ..task_mapping import ParallelTemplate
from ..schedule import TaskMapping
from ..utils import CollectiveComm, CommType


class MQATemplate0(ParallelTemplate):
    """MQA 优化模板：针对 GQA（Grouped Query Attention）优化

    对于九歌模型：40 query heads, 4 KV heads (每个 KV 头对应 10 个 query 头)
    确保同一 KV 组的 query heads 分配到相邻设备

    参考: apex_plus/parallel/templates/attention.py
    """

    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        """按 KV 头分组分配 query heads

        Args:
            cell: MQA Cell
            num_devices: 设备数量

        Returns:
            TaskMapping 或 None
        """
        if not isinstance(cell, MQA):
            return None

        if cell.num_query_heads < num_devices:
            # Query heads 不足以分配到所有设备
            return None

        tasks_per_device: List[Dict[str, List[Task]]] = [{}] * num_devices

        if cell.num_kv_heads < num_devices:
            # 场景 1: KV 头少于设备数，需要在设备间复制 KV 头
            # 每个 KV 头分配若干设备
            num_devices_per_kv = [num_devices // cell.num_kv_heads] * cell.num_kv_heads
            for i in range(num_devices % cell.num_kv_heads):
                num_devices_per_kv[i] += 1

            tasks_per_device = []
            for _ in range(num_devices):
                tasks_per_device.append({})

            device_idx = 0
            for kv_id in range(cell.num_kv_heads):
                n_dev = num_devices_per_kv[kv_id]

                # 计算每个设备分配的 query heads 数量
                query_per_dev = [cell.num_query_per_kv // n_dev] * n_dev
                for j in range(cell.num_query_per_kv % n_dev):
                    query_per_dev[j] += 1

                # 分配该 KV 组的 query heads
                start = kv_id * cell.num_query_per_kv
                for j in range(n_dev):
                    end = start + query_per_dev[j]
                    tasks_per_device[device_idx].setdefault("MQAHead", []).extend(
                        cell.heads[start:end]
                    )
                    device_idx += 1
                    start = end
        else:
            # 场景 2: KV 头 >= 设备数，均匀分配 KV 头到设备
            kv_per_device = [cell.num_kv_heads // num_devices] * num_devices
            for i in range(cell.num_kv_heads % num_devices):
                kv_per_device[i] += 1

            tasks_per_device = []
            for _ in range(num_devices):
                tasks_per_device.append({})

            start = 0
            for i in range(num_devices):
                end = start + kv_per_device[i]
                # 每个 KV 头对应 num_query_per_kv 个 query heads
                query_start = start * cell.num_query_per_kv
                query_end = end * cell.num_query_per_kv
                tasks_per_device[i].setdefault("MQAHead", []).extend(
                    cell.heads[query_start:query_end]
                )
                start = end

        task_mapping = TaskMapping(
            tasks_per_device,
            CollectiveComm(CommType.AllReduce, num_devices)
        )

        return task_mapping


class MHATemplate0(ParallelTemplate):
    """MHA 默认模板：均匀分配 attention heads

    对于标准 MHA，直接使用均匀分配即可
    """

    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        """均匀分配 attention heads

        Args:
            cell: MHA Cell
            num_devices: 设备数量

        Returns:
            TaskMapping 或 None
        """
        if not isinstance(cell, MHA):
            return None

        if cell.num_heads < num_devices:
            return None

        # 计算每个设备的 heads 数量
        heads_per_device = [cell.num_heads // num_devices] * num_devices
        for i in range(cell.num_heads % num_devices):
            heads_per_device[i] += 1

        tasks_per_device = []
        start = 0
        for i in range(num_devices):
            end = start + heads_per_device[i]
            tasks_per_device.append({
                "AttentionHead": cell.heads[start:end]
            })
            start = end

        task_mapping = TaskMapping(
            tasks_per_device,
            CollectiveComm(CommType.AllReduce, num_devices)
        )

        return task_mapping
