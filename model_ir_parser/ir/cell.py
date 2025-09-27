"""Cell层定义 - 包含具体的网络层实现"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List
from .task import Task, AttentionHead, FFNFilter


class Cell(ABC):
    """Cell基类"""

    def __init__(self):
        raise NotImplementedError

    @classmethod
    def get_name(cls) -> str:
        """获取Cell名称"""
        return cls.__name__

    @abstractmethod
    def get_tasks(self) -> List[Task]:
        """获取Cell包含的所有Task"""
        raise NotImplementedError

    @abstractmethod
    def get_num_task_types(self) -> int:
        """获取Task类型数量"""
        raise NotImplementedError

    @abstractmethod
    def has_same_spec(self, other: object) -> bool:
        """判断是否具有相同规格"""
        raise NotImplementedError

    def is_attn(self) -> bool:
        """判断是否为注意力Cell"""
        return any(task.is_attn() for task in self.get_tasks())


@dataclass
class Embedding(Cell):
    """词嵌入层"""
    vocab_size: int
    hidden_size: int

    def get_tasks(self) -> List[Task]:
        """Embedding没有tasks"""
        return []

    def get_num_task_types(self) -> int:
        return 0

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, Embedding):
            return False
        return (self.vocab_size == other.vocab_size and
                self.hidden_size == other.hidden_size)


@dataclass
class Sampler(Cell):
    """输出采样层"""
    vocab_size: int
    hidden_size: int

    def get_tasks(self) -> List[Task]:
        """Sampler没有tasks"""
        return []

    def get_num_task_types(self) -> int:
        return 0

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, Sampler):
            return False
        return (self.vocab_size == other.vocab_size and
                self.hidden_size == other.hidden_size)


@dataclass
class MHA(Cell):
    """多头注意力（Multi-Head Attention）"""
    num_heads: int
    hidden_size: int
    head_size: int

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0
        expected_head_size = self.hidden_size // self.num_heads
        if self.head_size != expected_head_size:
            raise ValueError(f"head_size {self.head_size} must equal hidden_size/num_heads {expected_head_size}")

        # 创建注意力头
        self.heads = [
            AttentionHead(i, self.head_size, self.hidden_size)
            for i in range(self.num_heads)
        ]

    def get_tasks(self) -> List[AttentionHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MHA):
            return False
        return (self.num_heads == other.num_heads and
                self.hidden_size == other.hidden_size)


@dataclass
class MQA(Cell):
    """多查询注意力（Multi-Query Attention）- 支持GQA场景"""
    num_query_heads: int  # 查询头数
    num_kv_heads: int     # KV头数（GQA时小于查询头数）
    head_size: int
    hidden_size: int

    def __post_init__(self):
        assert self.num_query_heads % self.num_kv_heads == 0, \
            f"num_query_heads {self.num_query_heads} must be divisible by num_kv_heads {self.num_kv_heads}"
        self.num_query_per_kv = self.num_query_heads // self.num_kv_heads

        # 创建查询头（注意：这里简化处理，实际APEX中MQA的task创建逻辑更复杂）
        self.heads = [
            AttentionHead(i, self.head_size, self.hidden_size)
            for i in range(self.num_query_heads)
        ]

    def get_tasks(self) -> List[AttentionHead]:
        return self.heads

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MQA):
            return False
        return (self.num_query_heads == other.num_query_heads and
                self.num_kv_heads == other.num_kv_heads and
                self.head_size == other.head_size and
                self.hidden_size == other.hidden_size)


@dataclass
class MLP(Cell):
    """标准MLP前馈网络"""
    hidden_size: int
    intermediate_size: int

    def __post_init__(self):
        self.filters = [
            FFNFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[FFNFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, MLP):
            return False
        return (self.hidden_size == other.hidden_size and
                self.intermediate_size == other.intermediate_size)


@dataclass
class GLU(Cell):
    """GLU激活的前馈网络"""
    hidden_size: int
    intermediate_size: int

    def __post_init__(self):
        self.filters = [
            FFNFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[FFNFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, GLU):
            return False
        return (self.hidden_size == other.hidden_size and
                self.intermediate_size == other.intermediate_size)


@dataclass
class SwiGLU(Cell):
    """SwiGLU激活的前馈网络（九歌模型使用）"""
    hidden_size: int
    intermediate_size: int

    def __post_init__(self):
        self.filters = [
            FFNFilter(i, self.hidden_size, self.intermediate_size)
            for i in range(self.intermediate_size)
        ]

    def get_tasks(self) -> List[FFNFilter]:
        return self.filters

    def get_num_task_types(self) -> int:
        return 1

    def has_same_spec(self, other: object) -> bool:
        if not isinstance(other, SwiGLU):
            return False
        return (self.hidden_size == other.hidden_size and
                self.intermediate_size == other.intermediate_size)