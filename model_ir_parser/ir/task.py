"""Task层定义 - 最底层的计算单元"""

from abc import ABC, abstractmethod
from typing import List


class Task(ABC):
    """任务基类"""

    def __init__(self):
        pass

    def get_type(self) -> str:
        """获取任务类型"""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def get_param_size(tasks: List["Task"], dtype) -> int:
        """计算参数大小"""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_kv_token_size(tasks: List["Task"], dtype) -> int:
        """计算KV缓存大小"""
        raise NotImplementedError

    @classmethod
    def is_attn(cls) -> bool:
        """判断是否为注意力任务"""
        return False


class AttentionHead(Task):
    """注意力头任务"""

    def __init__(self, head_id: int, head_size: int, hidden_size: int):
        super().__init__()
        self.head_id = head_id
        self.head_size = head_size
        self.hidden_size = hidden_size

    @staticmethod
    def get_param_size(tasks: List["AttentionHead"], dtype) -> int:
        """计算注意力头的参数大小"""
        if not tasks:
            return 0
        # Q, K, V, O projections
        head_size = tasks[0].head_size
        hidden_size = tasks[0].hidden_size
        num_heads = len(tasks)
        # 4个投影矩阵：Q, K, V, O
        return 4 * num_heads * head_size * hidden_size * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["AttentionHead"], dtype) -> int:
        """计算每个token的KV缓存大小"""
        if not tasks:
            return 0
        head_size = tasks[0].head_size
        num_heads = len(tasks)
        # K和V各需要缓存
        return 2 * num_heads * head_size * dtype.size

    @classmethod
    def is_attn(cls) -> bool:
        return True

    def __repr__(self):
        return f"AttentionHead(id={self.head_id}, head_size={self.head_size})"


class FFNFilter(Task):
    """前馈网络过滤器任务"""

    def __init__(self, filter_id: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.filter_id = filter_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

    @staticmethod
    def get_param_size(tasks: List["FFNFilter"], dtype) -> int:
        """计算FFN的参数大小"""
        if not tasks:
            return 0
        hidden_size = tasks[0].hidden_size
        intermediate_size = tasks[0].intermediate_size
        # 2个矩阵：up projection和down projection
        return 2 * hidden_size * intermediate_size * dtype.size

    @staticmethod
    def get_kv_token_size(tasks: List["FFNFilter"], dtype) -> int:
        """FFN不需要KV缓存"""
        return 0

    def __repr__(self):
        return f"FFNFilter(id={self.filter_id}, intermediate={self.intermediate_size})"