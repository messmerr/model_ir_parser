"""集合通信类型定义"""

from enum import Enum, auto
from dataclasses import dataclass


class CommType(Enum):
    """集合通信类型枚举"""
    AllReduce = auto()
    AllGather = auto()
    AllToAll = auto()
    ReduceScatter = auto()


@dataclass
class CollectiveComm:
    """集合通信操作描述

    Attributes:
        comm_type: 通信类型
        num_devices: 参与通信的设备数量
        size_factor: 数据大小因子（默认1.0）
    """
    comm_type: CommType
    num_devices: int
    size_factor: float = 1.0

    def __repr__(self) -> str:
        if self.size_factor == 1.0:
            return f"CollectiveComm({self.comm_type.name}, {self.num_devices} devices)"
        else:
            return (
                f"CollectiveComm({self.comm_type.name}, "
                f"{self.num_devices} devices, "
                f"size_factor={self.size_factor})"
            )
