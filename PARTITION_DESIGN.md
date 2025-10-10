# Task Partitioner 模块设计文档

## 1. 模块概述

### 1.1 功能定位
基于已解析的 Transformer IR，实现硬件无关的任务划分枚举引擎。参照 APEX 的四层并行化策略，生成所有可行的并行执行方案候选集。

### 1.2 输入输出
- **输入**: Transformer IR + 设备数量
- **输出**: List[ParallelSchedule] - 所有可行的并行划分方案

### 1.3 设计原则
- 硬件无关：不依赖真实集群对象，仅使用抽象设备数
- 纯拓扑划分：只输出方案结构，不做性能仿真
- APEX 兼容：数据结构与 APEX 保持一致

## 2. 核心算法

### 2.1 四层枚举机制

参考 `apex_plus/search/engine.py:41-177`

```python
def generate_schedules(num_blocks, block, num_devices):
    parallel_schedules = []

    # Level 1: 模型副本并行
    for num_replicas in divisors(num_devices):
        num_replica_devices = num_devices // num_replicas

        # Level 2: 流水线并行
        for num_stages in divisors(num_replica_devices):
            if num_blocks % num_stages != 0:
                continue

            num_stage_devices = num_replica_devices // num_stages
            num_blocks_per_stage = num_blocks // num_stages

            # Level 3: Cell 副本并行
            schedules_per_cell = []
            for cell in block.cells:
                cell_schedules = []
                for num_cell_replicas in divisors(num_stage_devices):
                    num_cell_devices = num_stage_devices // num_cell_replicas

                    # Level 4: Task 并行
                    for template in get_templates(cell):
                        task_mapping = template.map_tasks(cell, num_cell_devices)
                        if task_mapping is None:
                            continue

                        cell_schedule = CellSchedule(
                            cell, num_cell_replicas, task_mapping
                        )
                        cell_schedules.append(cell_schedule)

                schedules_per_cell.append(cell_schedules)

            # 生成所有 Cell 组合
            for combination in cartesian_product(schedules_per_cell):
                # 验证约束
                if not validate_combination(combination):
                    continue

                # 计算 reshard 通信
                reshard_comms = compute_reshard(combination)
                if reshard_comms is None:
                    continue

                stage_schedule = StageSchedule(
                    block, num_blocks_per_stage, combination, reshard_comms
                )

                parallel_schedule = ParallelSchedule(
                    num_replicas, num_stages, stage_schedule
                )
                parallel_schedules.append(parallel_schedule)

    return parallel_schedules
```

### 2.2 约束验证规则

**Level 3 约束**:
1. Cell 副本数互质: `gcd(*num_cell_replicas) == 1`
2. 所有 Attention Cell 副本数必须相同
3. 相邻 Cell 之间可进行 reshard

**Level 4 约束**:
1. Task 数量 >= 设备数（否则返回 None）
2. 设备负载尽量均衡

## 3. 数据结构设计

### 3.1 核心类定义

```python
# task_partitioner/schedule.py

@dataclass
class TaskMapping:
    """Level 4: Task 到设备的映射"""
    tasks_per_device: List[Dict[str, List[Task]]]
    collective_comm: CollectiveComm

    def get_num_devices(self) -> int:
        return len(self.tasks_per_device)


@dataclass
class CellSchedule:
    """Level 3: Cell 调度方案"""
    cell: Cell
    num_replicas: int
    task_mapping: TaskMapping

    def get_num_devices(self) -> int:
        return self.num_replicas * self.task_mapping.get_num_devices()


@dataclass
class StageSchedule:
    """Level 2: Stage 调度方案"""
    block: Block
    num_blocks: int
    cell_schedules: List[CellSchedule]
    reshard_comms: List[List[CollectiveComm]]


@dataclass
class ParallelSchedule:
    """Level 1: 完整并行方案"""
    num_model_replicas: int
    num_stages: int
    stage_schedule: StageSchedule

    def get_total_devices(self) -> int:
        stage_devices = self.stage_schedule.cell_schedules[0].get_num_devices()
        return self.num_model_replicas * self.num_stages * stage_devices
```

### 3.2 集合通信类型

```python
# task_partitioner/utils/comm.py

class CommType(Enum):
    AllReduce = auto()
    AllGather = auto()
    AllToAll = auto()
    ReduceScatter = auto()


@dataclass
class CollectiveComm:
    comm_type: CommType
    num_devices: int
    size_factor: float = 1.0
```

## 4. Template 设计

### 4.1 DefaultTemplate - 均匀分配

参考 `apex_plus/parallel/templates/default.py`

```python
class DefaultTemplate:
    @staticmethod
    def map_tasks(cell: Cell, num_devices: int) -> Optional[TaskMapping]:
        tasks_per_type = {}
        for task in cell.get_tasks():
            task_type = task.get_type()
            tasks_per_type.setdefault(task_type, []).append(task)

        tasks_per_device = [{}] * num_devices

        for task_type, tasks in tasks_per_type.items():
            num_tasks = len(tasks)
            tasks_per_device_count = [num_tasks // num_devices] * num_devices
            for i in range(num_tasks % num_devices):
                tasks_per_device_count[i] += 1

            start = 0
            for i in range(num_devices):
                end = start + tasks_per_device_count[i]
                tasks_per_device[i][task_type] = tasks[start:end]
                start = end

        if all(len(x) == 0 for x in tasks_per_device[-1].values()):
            return None

        return TaskMapping(
            tasks_per_device,
            CollectiveComm(CommType.AllReduce, num_devices)
        )
```

### 4.2 MQATemplate0 - GQA 优化分配

参考 `apex_plus/parallel/templates/attention.py`

针对九歌模型: 40 query heads, 4 KV heads (每个 KV 头对应 10 个 query 头)

```python
class MQATemplate0:
    @staticmethod
    def map_tasks(cell: MQA, num_devices: int) -> Optional[TaskMapping]:
        if cell.num_query_heads < num_devices:
            return None

        if cell.num_kv_heads < num_devices:
            # KV 头在设备间复制
            num_devices_per_kv = [num_devices // cell.num_kv_heads] * cell.num_kv_heads
            for i in range(num_devices % cell.num_kv_heads):
                num_devices_per_kv[i] += 1

            tasks_per_device = [{}] * num_devices
            device_idx = 0

            for kv_id in range(cell.num_kv_heads):
                n_dev = num_devices_per_kv[kv_id]
                query_per_dev = [cell.num_query_per_kv // n_dev] * n_dev
                for j in range(cell.num_query_per_kv % n_dev):
                    query_per_dev[j] += 1

                start = kv_id * cell.num_query_per_kv
                for j in range(n_dev):
                    end = start + query_per_dev[j]
                    tasks_per_device[device_idx]["MQAHead"] = cell.heads[start:end]
                    device_idx += 1
                    start = end
        else:
            # KV 头均匀分配
            kv_per_device = [cell.num_kv_heads // num_devices] * num_devices
            for i in range(cell.num_kv_heads % num_devices):
                kv_per_device[i] += 1

            tasks_per_device = [{}] * num_devices
            start = 0
            for i in range(num_devices):
                end = start + kv_per_device[i]
                query_start = start * cell.num_query_per_kv
                query_end = end * cell.num_query_per_kv
                tasks_per_device[i]["MQAHead"] = cell.heads[query_start:query_end]
                start = end

        return TaskMapping(
            tasks_per_device,
            CollectiveComm(CommType.AllReduce, num_devices)
        )
```

### 4.3 Template 注册

```python
# task_partitioner/templates/__init__.py

TEMPLATE_REGISTRY = {
    "MHA": [DefaultTemplate],
    "MQA": [MQATemplate0],
    "SwiGLU": [DefaultTemplate],
    "GLU": [DefaultTemplate],
    "MLP": [DefaultTemplate],
}

def get_templates(cell: Cell) -> List[ParallelTemplate]:
    cell_name = cell.get_name()
    return TEMPLATE_REGISTRY.get(cell_name, [DefaultTemplate])
```

## 5. Reshard 逻辑

### 5.1 可 Reshard 判断

```python
# task_partitioner/reshard.py

def is_reshardable(c1: CellSchedule, c2: CellSchedule) -> bool:
    """判断从 c1 到 c2 是否可以进行数据重分布"""
    # 检查设备数是否相同
    if c1.get_num_devices() != c2.get_num_devices():
        return False

    # 检查副本数是否兼容
    if c1.num_replicas != c2.num_replicas:
        # 需要额外的通信操作
        if c1.num_replicas % c2.num_replicas != 0 and \
           c2.num_replicas % c1.num_replicas != 0:
            return False

    return True


def get_reshard_comm(c1: CellSchedule, c2: CellSchedule) -> List[CollectiveComm]:
    """计算 reshard 所需的集合通信操作"""
    comms = []

    if c1.num_replicas == c2.num_replicas:
        # 副本数相同，无需额外通信
        return comms

    # 简化实现：假设都需要 AllGather 或 ReduceScatter
    if c1.num_replicas < c2.num_replicas:
        # 从少副本到多副本：AllGather
        factor = c2.num_replicas // c1.num_replicas
        comms.append(CollectiveComm(CommType.AllGather, factor))
    else:
        # 从多副本到少副本：ReduceScatter
        factor = c1.num_replicas // c2.num_replicas
        comms.append(CollectiveComm(CommType.ReduceScatter, factor))

    return comms
```

## 6. 模块结构

```
Task_Decoupling/
├── task_partitioner/
│   ├── __init__.py
│   │   └── export: PartitionEngine
│   │
│   ├── engine.py
│   │   └── class PartitionEngine:
│   │       ├── __init__(ir: Transformer)
│   │       ├── generate_schedules(num_blocks, block, num_devices)
│   │       └── enumerate_all_plans(num_devices)
│   │
│   ├── schedule.py
│   │   ├── class TaskMapping
│   │   ├── class CellSchedule
│   │   ├── class StageSchedule
│   │   └── class ParallelSchedule
│   │
│   ├── task_mapping.py
│   │   └── class ParallelTemplate(ABC)
│   │
│   ├── reshard.py
│   │   ├── is_reshardable(c1, c2)
│   │   └── get_reshard_comm(c1, c2)
│   │
│   ├── templates/
│   │   ├── __init__.py
│   │   │   ├── TEMPLATE_REGISTRY
│   │   │   └── get_templates(cell)
│   │   │
│   │   ├── default.py
│   │   │   └── class DefaultTemplate
│   │   │
│   │   ├── attention.py
│   │   │   └── class MQATemplate0
│   │   │
│   │   └── ffn.py
│   │       └── class SwiGLUTemplate (可选)
│   │
│   └── utils/
│       ├── __init__.py
│       └── comm.py
│           ├── enum CommType
│           └── class CollectiveComm
│
├── examples/
│   └── partition_jiuge.py
│       └── 示例：枚举九歌模型在不同设备数下的方案
│
└── tests/
    └── test_partitioner.py
        └── 单元测试
```

## 7. 使用示例

```python
# examples/partition_jiuge.py

from model_ir_parser import JiugeParser
from task_partitioner import PartitionEngine

# 解析九歌配置
parser = JiugeParser()
ir = parser.parse_file("config.json")

# 创建分区引擎
engine = PartitionEngine(ir)

# 枚举 8 设备的所有方案
plans = engine.enumerate_all_plans(num_devices=8)

print(f"生成了 {len(plans)} 个候选方案")

# 打印第一个方案
plan = plans[0]
print(f"模型副本数: {plan.num_model_replicas}")
print(f"Pipeline 阶段数: {plan.num_stages}")
print(f"每阶段 Block 数: {plan.stage_schedule.num_blocks}")

for i, cell_schedule in enumerate(plan.stage_schedule.cell_schedules):
    cell = cell_schedule.cell
    print(f"Cell {i}: {cell.get_name()}")
    print(f"  副本数: {cell_schedule.num_replicas}")
    print(f"  设备数: {cell_schedule.task_mapping.get_num_devices()}")
    print(f"  通信类型: {cell_schedule.task_mapping.collective_comm.comm_type}")
```

## 8. 输出格式

### 8.1 JSON 导出

```python
# task_partitioner/engine.py

class PartitionEngine:
    def export_json(self, plans: List[ParallelSchedule], output_path: str):
        """导出方案到 JSON 文件"""
        data = []
        for idx, plan in enumerate(plans):
            plan_dict = {
                "schedule_id": idx,
                "num_model_replicas": plan.num_model_replicas,
                "num_stages": plan.num_stages,
                "num_blocks_per_stage": plan.stage_schedule.num_blocks,
                "cells": []
            }

            for cell_schedule in plan.stage_schedule.cell_schedules:
                cell_dict = {
                    "name": cell_schedule.cell.get_name(),
                    "num_replicas": cell_schedule.num_replicas,
                    "num_devices": cell_schedule.task_mapping.get_num_devices(),
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

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
```

### 8.2 可读文本输出

```python
def visualize_plan(self, plan: ParallelSchedule) -> str:
    """生成方案的可读文本描述"""
    lines = []
    lines.append(f"Model Replicas: {plan.num_model_replicas}")
    lines.append(f"Pipeline Stages: {plan.num_stages}")
    lines.append(f"Blocks per Stage: {plan.stage_schedule.num_blocks}")
    lines.append("")

    for i, cs in enumerate(plan.stage_schedule.cell_schedules):
        lines.append(f"Cell {i}: {cs.cell.get_name()}")
        lines.append(f"  Replicas: {cs.num_replicas}")
        lines.append(f"  Devices: {cs.task_mapping.get_num_devices()}")
        lines.append(f"  Comm: {cs.task_mapping.collective_comm.comm_type.name}")

        for j, device_tasks in enumerate(cs.task_mapping.tasks_per_device):
            task_counts = {k: len(v) for k, v in device_tasks.items()}
            lines.append(f"    Device {j}: {task_counts}")

        reshard = plan.stage_schedule.reshard_comms[i]
        if reshard:
            for comm in reshard:
                lines.append(f"  Reshard: {comm.comm_type.name}({comm.num_devices} devices)")
        lines.append("")

    return "\n".join(lines)
```

## 9. 实现步骤

### Phase 1: 基础框架 (1天)
1. 创建目录结构
2. 实现 `utils/comm.py`: CommType, CollectiveComm
3. 实现 `schedule.py`: 所有数据类
4. 实现 `task_mapping.py`: ParallelTemplate 基类

### Phase 2: Template 实现 (1-2天)
1. 实现 `templates/default.py`: DefaultTemplate
2. 实现 `templates/attention.py`: MQATemplate0
3. 实现 `templates/__init__.py`: 注册机制
4. 单元测试：验证 task 分配正确性

### Phase 3: Reshard 逻辑 (0.5天)
1. 实现 `reshard.py`: is_reshardable, get_reshard_comm
2. 单元测试：验证 reshard 判断

### Phase 4: 核心引擎 (2-3天)
1. 实现 `engine.py`: PartitionEngine 类
2. 实现四层嵌套循环枚举
3. 实现约束验证:
   - Cell 副本数互质检查
   - Attention Cell 一致性检查
   - Reshard 可行性检查
4. 添加日志输出

### Phase 5: 输出与示例 (1天)
1. 实现 JSON 导出
2. 实现文本可视化
3. 编写 `examples/partition_jiuge.py`
4. 验证输出正确性

### Phase 6: 测试与优化 (1天)
1. 编写完整单元测试
2. 性能测试（枚举速度）
3. 与 APEX 对比验证
4. 文档完善

## 10. 测试计划

### 10.1 单元测试

```python
# tests/test_partitioner.py

def test_default_template():
    """测试默认模板的 task 分配"""
    # 创建一个简单的 MHA cell
    cell = MHA(num_heads=8, hidden_size=512, head_size=64)

    # 测试 4 设备分配
    template = DefaultTemplate()
    mapping = template.map_tasks(cell, num_devices=4)

    assert mapping is not None
    assert mapping.get_num_devices() == 4
    # 每设备应有 2 个 heads
    for device_tasks in mapping.tasks_per_device:
        assert len(device_tasks["AttentionHead"]) == 2

def test_mqa_template():
    """测试 MQA 模板的 GQA 分配"""
    # 九歌模型配置
    cell = MQA(num_query_heads=40, num_kv_heads=4, head_size=64, hidden_size=2560)

    # 测试 8 设备分配
    template = MQATemplate0()
    mapping = template.map_tasks(cell, num_devices=8)

    assert mapping is not None
    assert mapping.get_num_devices() == 8
    # 每设备应有 5 个 query heads
    for device_tasks in mapping.tasks_per_device:
        assert len(device_tasks["MQAHead"]) == 5

def test_reshard():
    """测试 reshard 判断"""
    cell1 = MHA(num_heads=8, hidden_size=512, head_size=64)
    cell2 = SwiGLU(hidden_size=512, intermediate_size=2048)

    template = DefaultTemplate()
    mapping1 = template.map_tasks(cell1, num_devices=4)
    mapping2 = template.map_tasks(cell2, num_devices=4)

    cs1 = CellSchedule(cell1, num_replicas=1, task_mapping=mapping1)
    cs2 = CellSchedule(cell2, num_replicas=1, task_mapping=mapping2)

    assert is_reshardable(cs1, cs2) == True

def test_partition_engine():
    """测试完整的分区引擎"""
    # 创建一个小模型
    block = Block([
        MHA(num_heads=4, hidden_size=256, head_size=64),
        MLP(hidden_size=256, intermediate_size=1024)
    ])
    ir = Transformer.from_blocks(
        vocab_size=1000,
        hidden_size=256,
        num_encoder_blocks=0,
        encoder_block=None,
        num_decoder_blocks=4,
        decoder_block=block
    )

    engine = PartitionEngine(ir)
    plans = engine.enumerate_all_plans(num_devices=2)

    assert len(plans) > 0
    # 验证所有方案的设备总数
    for plan in plans:
        assert plan.get_total_devices() == 2
```

### 10.2 集成测试

```python
def test_jiuge_full():
    """测试九歌模型完整流程"""
    parser = JiugeParser()
    ir = parser.parse_file("config.json")

    engine = PartitionEngine(ir)

    # 测试不同设备数
    for num_devices in [1, 2, 4, 8]:
        plans = engine.enumerate_all_plans(num_devices)
        assert len(plans) > 0

        # 验证每个方案
        for plan in plans:
            assert plan.get_total_devices() == num_devices
            assert plan.num_model_replicas * plan.num_stages <= num_devices
```

## 11. 预期结果

### 11.1 九歌模型 (62层, 8设备)

预期生成方案数量范围: 10-50 个

典型方案示例:
- 方案 1: 1副本 × 1阶段 × (MQA:8设备 + SwiGLU:8设备)
- 方案 2: 1副本 × 2阶段 × (MQA:4设备 + SwiGLU:4设备)
- 方案 3: 2副本 × 1阶段 × (MQA:4设备 + SwiGLU:4设备)
- 方案 4: 1副本 × 1阶段 × (MQA:1副本×8设备 + SwiGLU:1副本×8设备)

### 11.2 性能要求

- 8设备62层: 枚举时间 < 5秒
- 16设备62层: 枚举时间 < 30秒
- 内存占用 < 1GB

## 12. 后续扩展

1. 支持 Encoder-Decoder 模型 (如 T5, Whisper)
2. 支持 MoE 模型 (添加 MoETemplate)
3. 添加约束配置接口 (允许用户禁用某些策略)
4. 性能优化: 剪枝、缓存、并行枚举
5. 可视化工具: 生成设备-任务映射图
