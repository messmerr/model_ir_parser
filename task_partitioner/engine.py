"""任务分区搜索引擎

参考: apex_plus/search/engine.py
"""

import math
import json
from typing import List, Optional
from pathlib import Path

from model_ir_parser.ir import Transformer, Block
from .schedule import ParallelSchedule, StageSchedule, CellSchedule
from .reshard import is_reshardable, get_reshard_comm
from .templates import get_templates


def _get_divisors(n: int) -> List[int]:
    """获取 n 的所有因子

    Args:
        n: 正整数

    Returns:
        因子列表（从小到大）
    """
    return [i for i in range(1, n + 1) if n % i == 0]


class PartitionEngine:
    """任务分区引擎

    基于 Transformer IR，枚举所有可行的并行执行方案
    """

    def __init__(self, ir: Transformer):
        """初始化分区引擎

        Args:
            ir: Transformer IR
        """
        self.ir = ir

    def generate_schedules(
        self,
        num_blocks: int,
        block: Block,
        num_devices: int
    ) -> List[ParallelSchedule]:
        """生成所有可行的并行调度方案

        实现四层枚举：
        1. Model-level data parallelism (模型副本)
        2. Pipeline parallelism (流水线阶段)
        3. Cell-level data parallelism (Cell 副本)
        4. Task parallelism (任务分配)

        Args:
            num_blocks: Block 总数
            block: Block 实例
            num_devices: 设备总数

        Returns:
            所有可行的 ParallelSchedule 列表
        """
        parallel_schedules: List[ParallelSchedule] = []

        # Level 1: 模型副本并行
        for num_replicas in _get_divisors(num_devices):
            num_replica_devices = num_devices // num_replicas

            # Level 2: 流水线并行
            for num_stages in _get_divisors(num_replica_devices):
                # 检查 blocks 能否均匀分配到 stages
                if num_blocks % num_stages != 0:
                    continue

                num_stage_devices = num_replica_devices // num_stages
                num_blocks_per_stage = num_blocks // num_stages

                # Level 3: Cell 副本并行
                # 为每个 cell 生成所有可能的 CellSchedule
                schedules_per_cell: List[List[CellSchedule]] = []

                for cell in block.cells:
                    cell_schedules = []

                    for num_cell_replicas in _get_divisors(num_stage_devices):
                        num_cell_replica_devices = num_stage_devices // num_cell_replicas

                        # Level 4: Task 并行
                        for template in get_templates(cell):
                            task_mapping = template.map_tasks(
                                cell, num_cell_replica_devices
                            )

                            if task_mapping is None:
                                # 任务数不足，无法分配
                                continue

                            cell_schedule = CellSchedule(
                                cell,
                                num_cell_replicas,
                                task_mapping
                            )
                            cell_schedules.append(cell_schedule)

                            # 优化：如果只有一个设备，其他模板都会产生相同结果
                            if num_cell_replica_devices == 1:
                                break

                    schedules_per_cell.append(cell_schedules)

                # 生成所有 Cell 组合
                # 使用笛卡尔积枚举
                num_schedules = 1
                for schedules in schedules_per_cell:
                    num_schedules *= len(schedules)

                for i in range(num_schedules):
                    cell_schedules: List[CellSchedule] = []
                    idx = i

                    for schedules in schedules_per_cell:
                        cell_schedules.append(schedules[idx % len(schedules)])
                        idx //= len(schedules)

                    # 验证约束
                    if not self._validate_cell_combination(cell_schedules):
                        continue

                    # 计算 reshard 通信
                    num_cells = len(block.cells)
                    reshard_comms = []
                    valid = True

                    for j in range(num_cells):
                        c1 = cell_schedules[j]
                        c2 = cell_schedules[(j + 1) % num_cells]

                        if not is_reshardable(c1, c2):
                            valid = False
                            break

                        comm = get_reshard_comm(c1, c2)
                        reshard_comms.append(comm)

                    if not valid:
                        continue

                    # 创建 StageSchedule
                    stage_schedule = StageSchedule(
                        block,
                        num_blocks_per_stage,
                        cell_schedules,
                        reshard_comms
                    )

                    # 创建 ParallelSchedule
                    parallel_schedule = ParallelSchedule(
                        num_replicas,
                        num_stages,
                        stage_schedule
                    )

                    parallel_schedules.append(parallel_schedule)

        return parallel_schedules

    def _validate_cell_combination(
        self,
        cell_schedules: List[CellSchedule]
    ) -> bool:
        """验证 Cell 组合的约束

        Args:
            cell_schedules: CellSchedule 列表

        Returns:
            True 如果满足所有约束
        """
        if len(cell_schedules) <= 1:
            return True

        # 约束 1: Cell 副本数必须互质
        num_cell_replicas = [cs.num_replicas for cs in cell_schedules]
        if len(num_cell_replicas) > 1:
            gcd_value = math.gcd(*num_cell_replicas)
            if gcd_value != 1:
                return False

        # 约束 2: 所有 Attention Cell 的副本数必须相同
        attn_cell_replicas = [
            cs.num_replicas for cs in cell_schedules
            if cs.cell.is_attn()
        ]
        if len(set(attn_cell_replicas)) > 1:
            return False

        return True

    def enumerate_all_plans(self, num_devices: int) -> List[ParallelSchedule]:
        """枚举 Decoder 的所有并行方案

        Args:
            num_devices: 设备总数

        Returns:
            所有可行的 ParallelSchedule 列表
        """
        if self.ir.num_decoder_blocks == 0:
            raise ValueError("No decoder blocks in the model")

        return self.generate_schedules(
            self.ir.num_decoder_blocks,
            self.ir.decoder_block,
            num_devices
        )

    def visualize_plan(self, plan: ParallelSchedule) -> str:
        """生成方案的可读文本描述

        Args:
            plan: ParallelSchedule

        Returns:
            可读的文本描述
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"Model Replicas: {plan.num_model_replicas}")
        lines.append(f"Pipeline Stages: {plan.num_stages}")
        lines.append(f"Blocks per Stage: {plan.stage_schedule.num_blocks}")
        lines.append(f"Total Devices: {plan.get_total_devices()}")
        lines.append("")

        for i, cs in enumerate(plan.stage_schedule.cell_schedules):
            lines.append(f"Cell {i}: {cs.cell.get_name()}")
            lines.append(f"  Replicas: {cs.num_replicas}")
            lines.append(f"  Devices per Replica: {cs.task_mapping.get_num_devices()}")
            lines.append(f"  Comm: {cs.task_mapping.collective_comm.comm_type.name}")

            # 显示每个设备的任务分配
            for j, device_tasks in enumerate(cs.task_mapping.tasks_per_device):
                task_counts = {k: len(v) for k, v in device_tasks.items()}
                lines.append(f"    Device {j}: {task_counts}")

            # 显示 reshard 通信
            reshard = plan.stage_schedule.reshard_comms[i]
            if reshard:
                for comm in reshard:
                    lines.append(f"  Reshard: {comm}")

            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def export_json(
        self,
        plans: List[ParallelSchedule],
        output_path: str
    ) -> None:
        """导出方案到 JSON 文件

        Args:
            plans: ParallelSchedule 列表
            output_path: 输出文件路径
        """
        data = []

        for idx, plan in enumerate(plans):
            plan_dict = {
                "schedule_id": idx,
                "num_model_replicas": plan.num_model_replicas,
                "num_stages": plan.num_stages,
                "num_blocks_per_stage": plan.stage_schedule.num_blocks,
                "total_devices": plan.get_total_devices(),
                "cells": []
            }

            for cell_schedule in plan.stage_schedule.cell_schedules:
                cell_dict = {
                    "name": cell_schedule.cell.get_name(),
                    "num_replicas": cell_schedule.num_replicas,
                    "num_devices_per_replica": cell_schedule.task_mapping.get_num_devices(),
                    "comm_type": cell_schedule.task_mapping.collective_comm.comm_type.name,
                    "tasks_per_device": []
                }

                for device_tasks in cell_schedule.task_mapping.tasks_per_device:
                    device_dict = {}
                    for task_type, tasks in device_tasks.items():
                        device_dict[task_type] = len(tasks)
                    cell_dict["tasks_per_device"].append(device_dict)

                plan_dict["cells"].append(cell_dict)

            data.append(plan_dict)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
