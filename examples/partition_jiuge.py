#!/usr/bin/env python3
"""九歌模型任务分区示例

演示如何使用 task_partitioner 模块枚举九歌模型的并行执行方案
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_ir_parser import JiugeParser
from task_partitioner import PartitionEngine


def main():
    """主函数"""
    # 配置文件路径
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        print(f"错误: 配置文件不存在 {config_path}")
        return 1

    print(f"解析九歌模型配置: {config_path}")
    print("-" * 60)

    # 解析 IR
    parser = JiugeParser()
    ir = parser.parse_file(str(config_path))

    print("模型信息:")
    print(f"  模型类型: {ir.metadata.model_type}")
    print(f"  隐藏层维度: {ir.hidden_size}")
    print(f"  层数: {ir.num_decoder_blocks}")
    print(f"  词表大小: {ir.embedding.vocab_size}")
    print()

    # 显示 Block 结构
    print("Decoder Block 结构:")
    for i, cell in enumerate(ir.decoder_block.cells):
        print(f"  Cell {i}: {cell.get_name()}")
        if hasattr(cell, 'num_query_heads'):
            print(f"    Query Heads: {cell.num_query_heads}, KV Heads: {cell.num_kv_heads}")
        elif hasattr(cell, 'intermediate_size'):
            print(f"    Intermediate Size: {cell.intermediate_size}")
    print()

    # 创建分区引擎
    engine = PartitionEngine(ir)

    # 测试不同的设备数
    device_counts = [2, 4, 8]

    for num_devices in device_counts:
        print("=" * 60)
        print(f"枚举 {num_devices} 设备的并行方案")
        print("=" * 60)

        # 枚举所有方案
        plans = engine.enumerate_all_plans(num_devices)

        print(f"生成了 {len(plans)} 个候选方案\n")

        # 显示前 3 个方案的详细信息
        for i, plan in enumerate(plans[:3]):
            print(f"--- 方案 {i} ---")
            print(engine.visualize_plan(plan))

        # 如果方案太多，只显示统计信息
        if len(plans) > 3:
            print(f"... (还有 {len(plans) - 3} 个方案)\n")

        # 导出到 JSON
        output_path = Path(__file__).parent / f"plans_{num_devices}devices.json"
        engine.export_json(plans, str(output_path))
        print(f"所有方案已导出到: {output_path}\n")

    print("=" * 60)
    print("分区枚举完成！")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
