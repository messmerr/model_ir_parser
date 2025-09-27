"""九歌模型配置解析器"""

import json
import math
from typing import Dict, Any, Optional
from pathlib import Path

from .ir import (
    Transformer,
    Block,
    MHA,
    MQA,
    MLP,
    GLU,
    SwiGLU,
    Embedding,
    Sampler,
    ModelMetadata
)
from .utils import get_dtype


class JiugeParser:
    """九歌模型配置解析器"""

    def parse_file(self, config_path: str) -> Transformer:
        """从配置文件解析

        Args:
            config_path: 配置文件路径

        Returns:
            Transformer: 解析后的IR结构
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return self.parse(config)

    def parse(self, config: Dict[str, Any]) -> Transformer:
        """解析配置生成IR

        Args:
            config: 模型配置字典

        Returns:
            Transformer: 解析后的IR结构

        Raises:
            ValueError: 如果不是支持的模型类型
        """
        # 验证是否为九歌模型
        model_type = config.get("model_type", "")
        if model_type not in ["fm9g", "fm9g7b", "minicpm"]:
            # 也支持通过architectures字段识别
            architectures = config.get("architectures", [])
            if not any("FM9G" in arch for arch in architectures):
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"This parser only supports FM9G/Jiuge models."
                )

        # 提取元信息
        metadata = self.extract_metadata(config)

        # 构建注意力Cell（支持GQA）
        attention = self._build_attention(config)

        # 构建FFN Cell
        ffn = self._build_ffn(config)

        # 组装Block
        block = Block([attention, ffn])

        # 创建Transformer
        vocab_size = config["vocab_size"]
        hidden_size = config["hidden_size"]
        num_decoder_blocks = config["num_hidden_layers"]

        return Transformer.from_blocks(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_encoder_blocks=0,  # 九歌模型没有编码器
            encoder_block=None,
            num_decoder_blocks=num_decoder_blocks,
            decoder_block=block,
            metadata=metadata
        )

    def _build_attention(self, config: Dict[str, Any]) -> MQA:
        """构建注意力层

        Args:
            config: 模型配置

        Returns:
            Cell: MHA或MQA实例
        """
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        hidden_size = config["hidden_size"]
        head_size = hidden_size // num_heads

        if num_kv_heads < num_heads:
            # 使用MQA（九歌模型的GQA实际上是MQA的一种）
            return MQA(
                num_query_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                hidden_size=hidden_size
            )
        else:
            # 使用标准MHA
            return MHA(
                num_heads=num_heads,
                hidden_size=hidden_size,
                head_size=head_size
            )

    def _build_ffn(self, config: Dict[str, Any]) -> SwiGLU:
        """构建前馈网络层

        Args:
            config: 模型配置

        Returns:
            Cell: FFN Cell实例
        """
        activation = config.get("hidden_act", "silu")
        hidden_size = config["hidden_size"]
        intermediate_size = config["intermediate_size"]

        if activation in ["silu", "swiglu"]:
            return SwiGLU(hidden_size, intermediate_size)
        elif activation in ["gelu", "gelu_new"]:
            return GLU(hidden_size, intermediate_size)
        else:
            return MLP(hidden_size, intermediate_size)

    def extract_metadata(self, config: Dict[str, Any]) -> ModelMetadata:
        """提取模型元信息

        Args:
            config: 模型配置

        Returns:
            ModelMetadata: 模型元信息
        """
        # 计算九歌特有的scale参数
        scale_emb = config.get("scale_emb", 1.0)
        scale_depth = config.get("scale_depth", 1.0)
        dim_model_base = config.get("dim_model_base", 256)

        # 如果配置中没有明确的scale参数，根据InfiniLM的逻辑计算
        if config.get("model_type") in ["fm9g", "minicpm"] and scale_emb != 1.0:
            # 计算scale_output和scale_o/scale_down
            hidden_size = config["hidden_size"]
            num_hidden_layers = config["num_hidden_layers"]
            scale_output = hidden_size // dim_model_base
            scale_o = scale_depth / math.sqrt(num_hidden_layers)
            scale_down = scale_depth / math.sqrt(num_hidden_layers)
        else:
            scale_output = 1.0
            scale_o = 1.0
            scale_down = 1.0

        return ModelMetadata(
            model_type=config.get("model_type", "fm9g"),
            vocab_size=config.get("vocab_size"),
            max_position=config.get("max_position_embeddings", 2048),
            rope_theta=config.get("rope_theta", 100000.0),
            rope_scaling=config.get("rope_scaling"),
            dtype=config.get("torch_dtype", "float16"),
            # 九歌特有参数
            scale_emb=scale_emb,
            scale_depth=scale_depth,
            dim_model_base=dim_model_base,
            # 其他参数
            rms_norm_eps=config.get("rms_norm_eps", 1e-5),
            end_token=config.get("eos_token_id", 2) if isinstance(config.get("eos_token_id"), int)
                     else config.get("eos_token_id", [2])[0] if isinstance(config.get("eos_token_id"), list)
                     else 2
        )

    def count_parameters(self, ir: Transformer) -> int:
        """计算模型参数数量

        Args:
            ir: Transformer IR

        Returns:
            int: 参数数量
        """
        total = 0

        # Embedding参数
        total += ir.embedding.vocab_size * ir.embedding.hidden_size

        # Decoder层参数
        if ir.decoder_block:
            for cell in ir.decoder_block.cells:
                if isinstance(cell, (MHA, MQA)):
                    # 注意力层参数：Q, K, V, O投影
                    if isinstance(cell, MHA):
                        # 标准MHA: 4个投影矩阵
                        total += 4 * cell.hidden_size * cell.hidden_size
                    else:  # MQA
                        # MQA: Q投影 + KV投影（压缩） + O投影
                        q_params = cell.num_query_heads * cell.head_size * cell.hidden_size
                        kv_params = 2 * cell.num_kv_heads * cell.head_size * cell.hidden_size
                        o_params = cell.hidden_size * cell.hidden_size
                        total += q_params + kv_params + o_params

                elif isinstance(cell, (MLP, GLU, SwiGLU)):
                    # FFN参数
                    if isinstance(cell, SwiGLU):
                        # SwiGLU有gate和up两个投影，加上down投影
                        total += 3 * cell.hidden_size * cell.intermediate_size
                    else:
                        # 标准MLP：up和down投影
                        total += 2 * cell.hidden_size * cell.intermediate_size

            # 乘以层数
            total *= ir.num_decoder_blocks

        # LayerNorm参数（每层2个：attention前和FFN前）
        total += 2 * ir.num_decoder_blocks * ir.hidden_size

        # 输出层参数
        total += ir.sampler.vocab_size * ir.sampler.hidden_size

        # 最终的LayerNorm
        total += ir.hidden_size

        return total

    def export_json(self, ir: Transformer, output_path: str) -> None:
        """将IR导出为JSON文件

        Args:
            ir: Transformer IR
            output_path: 输出文件路径
        """
        output = {
            "hidden_size": ir.hidden_size,
            "num_encoder_blocks": ir.num_encoder_blocks,
            "num_decoder_blocks": ir.num_decoder_blocks,
            "embedding": {
                "vocab_size": ir.embedding.vocab_size,
                "hidden_size": ir.embedding.hidden_size
            },
            "decoder_block": {
                "cells": []
            },
            "sampler": {
                "vocab_size": ir.sampler.vocab_size,
                "hidden_size": ir.sampler.hidden_size
            }
        }

        # 添加decoder block的cells信息
        if ir.decoder_block:
            for cell in ir.decoder_block.cells:
                cell_info = {
                    "type": cell.get_name()
                }
                if isinstance(cell, MHA):
                    cell_info.update({
                        "num_heads": cell.num_heads,
                        "hidden_size": cell.hidden_size,
                        "head_size": cell.head_size
                    })
                elif isinstance(cell, MQA):
                    cell_info.update({
                        "num_query_heads": cell.num_query_heads,
                        "num_kv_heads": cell.num_kv_heads,
                        "hidden_size": cell.hidden_size,
                        "head_size": cell.head_size
                    })
                elif isinstance(cell, (MLP, GLU, SwiGLU)):
                    cell_info.update({
                        "hidden_size": cell.hidden_size,
                        "intermediate_size": cell.intermediate_size
                    })
                output["decoder_block"]["cells"].append(cell_info)

        # 添加metadata
        if ir.metadata:
            output["metadata"] = ir.metadata.to_dict()

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    def get_summary(self, ir: Transformer) -> str:
        """获取模型架构摘要

        Args:
            ir: Transformer IR

        Returns:
            str: 模型摘要
        """
        summary = ir.get_summary()

        # 添加参数统计
        param_count = self.count_parameters(ir)
        summary += f"\n\nTotal Parameters: {param_count:,}"
        summary += f"\nParameter Size (FP16): {param_count * 2 / (1024**3):.2f} GB"

        return summary