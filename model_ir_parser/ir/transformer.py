"""Transformer顶层结构定义"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from .block import Block
from .cell import Embedding, Sampler


@dataclass
class ModelMetadata:
    """模型元信息"""
    model_type: str
    vocab_size: int
    max_position: int
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict] = None
    dtype: str = "float16"

    # 九歌特有参数
    scale_emb: float = 1.0
    scale_depth: float = 1.0
    dim_model_base: int = 256

    # 其他可选参数
    rms_norm_eps: float = 1e-5
    end_token: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_type": self.model_type,
            "vocab_size": self.vocab_size,
            "max_position": self.max_position,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "dtype": self.dtype,
            "scale_emb": self.scale_emb,
            "scale_depth": self.scale_depth,
            "dim_model_base": self.dim_model_base,
            "rms_norm_eps": self.rms_norm_eps,
            "end_token": self.end_token
        }


@dataclass
class Transformer:
    """Transformer模型的顶层IR表示（与APEX保持一致）"""
    embedding: Embedding
    num_encoder_blocks: int
    encoder_block: Optional[Block]  # 注意：是单个Block，不是List
    num_decoder_blocks: int
    decoder_block: Block            # 注意：是单个Block，不是List
    sampler: Sampler
    hidden_size: int
    metadata: Optional[ModelMetadata] = None

    def __post_init__(self):
        """初始化后验证"""
        if self.num_encoder_blocks < 0:
            raise ValueError(
                f"num_encoder_blocks must be non-negative, got {self.num_encoder_blocks}"
            )
        elif self.num_encoder_blocks == 0:
            if self.encoder_block is not None:
                raise ValueError(
                    f"encoder_block must be None if num_encoder_blocks == 0, got {self.encoder_block}"
                )
        else:
            # num_encoder_blocks > 0
            if self.encoder_block is None:
                raise ValueError(
                    f"encoder_block must not be None if num_encoder_blocks > 0"
                )

        if self.num_decoder_blocks < 0:
            raise ValueError(
                f"num_decoder_blocks must be non-negative, got {self.num_decoder_blocks}"
            )
        elif self.num_decoder_blocks == 0:
            if self.decoder_block is not None:
                raise ValueError(
                    f"decoder_block must be None if num_decoder_blocks == 0, got {self.decoder_block}"
                )
        else:
            # num_decoder_blocks > 0
            if self.decoder_block is None:
                raise ValueError(
                    f"decoder_block must not be None if num_decoder_blocks > 0"
                )

    @classmethod
    def from_blocks(cls,
                   vocab_size: int,
                   hidden_size: int,
                   num_encoder_blocks: int,
                   encoder_block: Optional[Block],
                   num_decoder_blocks: int,
                   decoder_block: Block,
                   metadata: Optional[ModelMetadata] = None) -> "Transformer":
        """工厂方法，与APEX的接口保持一致"""
        embedding = Embedding(vocab_size, hidden_size)
        sampler = Sampler(vocab_size, hidden_size)
        return cls(
            embedding=embedding,
            num_encoder_blocks=num_encoder_blocks,
            encoder_block=encoder_block,
            num_decoder_blocks=num_decoder_blocks,
            decoder_block=decoder_block,
            sampler=sampler,
            hidden_size=hidden_size,
            metadata=metadata
        )

    def validate(self) -> bool:
        """验证IR结构的合法性"""
        try:
            # 检查embedding和sampler的词表大小是否一致
            if self.embedding.vocab_size != self.sampler.vocab_size:
                return False

            # 检查hidden_size是否一致
            if self.embedding.hidden_size != self.hidden_size:
                return False
            if self.sampler.hidden_size != self.hidden_size:
                return False

            # 检查decoder block中的cells
            if self.decoder_block:
                for cell in self.decoder_block.cells:
                    if hasattr(cell, 'hidden_size'):
                        if cell.hidden_size != self.hidden_size:
                            return False

            return True
        except Exception:
            return False

    def get_summary(self) -> str:
        """获取模型摘要"""
        lines = []
        lines.append(f"Transformer Model Summary:")
        lines.append(f"  Hidden Size: {self.hidden_size}")
        lines.append(f"  Vocab Size: {self.embedding.vocab_size}")
        lines.append(f"  Encoder Blocks: {self.num_encoder_blocks}")
        lines.append(f"  Decoder Blocks: {self.num_decoder_blocks}")

        if self.decoder_block:
            lines.append(f"  Decoder Block Structure:")
            for i, cell in enumerate(self.decoder_block.cells):
                lines.append(f"    Cell {i}: {cell.get_name()}")

        if self.metadata:
            lines.append(f"  Model Type: {self.metadata.model_type}")
            lines.append(f"  Max Position: {self.metadata.max_position}")

        return "\n".join(lines)

    def __repr__(self):
        return (
            f"Transformer(embedding={self.embedding}, "
            f"num_encoder_blocks={self.num_encoder_blocks}, "
            f"encoder_block={self.encoder_block}, "
            f"num_decoder_blocks={self.num_decoder_blocks}, "
            f"decoder_block={self.decoder_block}, "
            f"sampler={self.sampler}, "
            f"hidden_size={self.hidden_size})"
        )