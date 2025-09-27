#!/usr/bin/env python3
"""九歌模型解析示例"""

import sys
import os
from pathlib import Path

# 将父目录添加到路径，以便导入model_ir_parser
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_ir_parser import JiugeParser


def main():
    """主函数"""
    # 配置文件路径（使用项目中的config.json）
    config_path = Path(__file__).parent.parent / "config.json"

    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        print("Please ensure config.json is in the model_ir_parser directory")
        return 1

    print(f"Parsing config file: {config_path}")
    print("-" * 60)

    # 创建解析器
    parser = JiugeParser()

    try:
        # 解析配置文件
        ir = parser.parse_file(str(config_path))

        # 打印基本信息
        print("Successfully parsed model configuration!")
        print()
        print("=== Model Structure ===")
        print(f"Model Type: {ir.metadata.model_type if ir.metadata else 'Unknown'}")
        print(f"Hidden Size: {ir.hidden_size}")
        print(f"Vocab Size: {ir.embedding.vocab_size}")
        print(f"Number of Decoder Blocks: {ir.num_decoder_blocks}")
        print(f"Number of Encoder Blocks: {ir.num_encoder_blocks}")
        print()

        # 打印Block结构
        print("=== Decoder Block Structure ===")
        if ir.decoder_block:
            for i, cell in enumerate(ir.decoder_block.cells):
                print(f"Cell {i}: {cell.get_name()}")

                if hasattr(cell, 'num_query_heads'):  # MQA
                    print(f"  - Query Heads: {cell.num_query_heads}")
                    print(f"  - KV Heads: {cell.num_kv_heads}")
                    print(f"  - Head Size: {cell.head_size}")
                    print(f"  - GQA Compression Ratio: {cell.num_query_heads}:{cell.num_kv_heads}")
                elif hasattr(cell, 'num_heads'):  # MHA
                    print(f"  - Number of Heads: {cell.num_heads}")
                    print(f"  - Head Size: {cell.head_size}")
                elif hasattr(cell, 'intermediate_size'):  # FFN
                    print(f"  - Hidden Size: {cell.hidden_size}")
                    print(f"  - Intermediate Size: {cell.intermediate_size}")
        print()

        # 打印元信息
        if ir.metadata:
            print("=== Model Metadata ===")
            print(f"Max Position Embeddings: {ir.metadata.max_position}")
            print(f"RoPE Theta: {ir.metadata.rope_theta}")
            print(f"Data Type: {ir.metadata.dtype}")
            print(f"RMS Norm Epsilon: {ir.metadata.rms_norm_eps}")

            # 九歌特有参数
            if ir.metadata.scale_emb != 1.0:
                print(f"Scale Embedding: {ir.metadata.scale_emb}")
            if ir.metadata.scale_depth != 1.0:
                print(f"Scale Depth: {ir.metadata.scale_depth}")
            if ir.metadata.dim_model_base != 256:
                print(f"Dim Model Base: {ir.metadata.dim_model_base}")
            print()

        # 参数统计
        print("=== Parameter Statistics ===")
        param_count = parser.count_parameters(ir)
        print(f"Total Parameters: {param_count:,}")
        print(f"Parameter Size (FP16): {param_count * 2 / (1024**3):.2f} GB")
        print(f"Parameter Size (BF16): {param_count * 2 / (1024**3):.2f} GB")
        print(f"Parameter Size (FP32): {param_count * 4 / (1024**3):.2f} GB")
        print()

        # 验证IR结构
        print("=== Validation ===")
        if ir.validate():
            print("✓ IR structure is valid")
        else:
            print("✗ IR structure validation failed")

        # 导出JSON（可选）
        output_path = Path(__file__).parent / "output_ir.json"
        parser.export_json(ir, str(output_path))
        print(f"\nIR exported to: {output_path}")

        # 打印完整摘要
        print("\n" + "=" * 60)
        print("FULL MODEL SUMMARY")
        print("=" * 60)
        print(parser.get_summary(ir))

    except Exception as e:
        print(f"Error parsing config: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())