"""Cell 间数据重分布逻辑

参考: apex_plus/parallel/reshard.py
"""

from typing import List

from .schedule import CellSchedule
from .utils import CollectiveComm, CommType


def is_reshardable(c1: CellSchedule, c2: CellSchedule) -> bool:
    """判断从 c1 到 c2 是否可以进行数据重分布

    Args:
        c1: 第一个 CellSchedule
        c2: 第二个 CellSchedule

    Returns:
        True 如果可以 reshard，否则 False
    """
    # 检查设备总数是否相同
    if c1.get_num_devices() != c2.get_num_devices():
        return False

    # 检查副本数是否兼容
    # 允许的情况：
    # 1. 副本数相同
    # 2. 一个是另一个的倍数（可以通过 AllGather 或 ReduceScatter）
    if c1.num_replicas != c2.num_replicas:
        if c1.num_replicas % c2.num_replicas != 0 and \
           c2.num_replicas % c1.num_replicas != 0:
            return False

    return True


def get_reshard_comm(c1: CellSchedule, c2: CellSchedule) -> List[CollectiveComm]:
    """计算从 c1 到 c2 的数据重分布所需的集合通信操作

    Args:
        c1: 第一个 CellSchedule
        c2: 第二个 CellSchedule

    Returns:
        所需的集合通信操作列表
    """
    comms = []

    if c1.num_replicas == c2.num_replicas:
        # 副本数相同
        # 检查 task_mapping 的设备数是否相同
        c1_mapping_devices = c1.task_mapping.get_num_devices()
        c2_mapping_devices = c2.task_mapping.get_num_devices()

        if c1_mapping_devices == c2_mapping_devices:
            # 完全相同，无需通信
            return comms
        else:
            # 需要 reshard（简化实现：使用 AllToAll）
            comms.append(CollectiveComm(
                CommType.AllToAll,
                c1_mapping_devices
            ))
    elif c1.num_replicas < c2.num_replicas:
        # 从少副本到多副本：需要 AllGather
        factor = c2.num_replicas // c1.num_replicas
        comms.append(CollectiveComm(
            CommType.AllGather,
            factor
        ))
    else:
        # 从多副本到少副本：需要 ReduceScatter
        factor = c1.num_replicas // c2.num_replicas
        comms.append(CollectiveComm(
            CommType.ReduceScatter,
            factor
        ))

    return comms
