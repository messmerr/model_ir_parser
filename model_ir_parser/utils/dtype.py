"""数据类型定义（参考APEX实现）"""

import enum


class DTYPE(enum.Enum):
    """数据类型枚举"""
    FLOAT32 = 0
    FLOAT16 = 1
    FLOAT8 = 2
    BFLOAT16 = 3


# 设置每种类型的字节大小
DTYPE.FLOAT32.size = 4
DTYPE.FLOAT16.size = 2
DTYPE.FLOAT8.size = 1
DTYPE.BFLOAT16.size = 2


# 数据类型映射表
_DTYPE_REGISTRY = {
    # Float32
    "float32": DTYPE.FLOAT32,
    "float": DTYPE.FLOAT32,
    "fp32": DTYPE.FLOAT32,
    # Float16
    "float16": DTYPE.FLOAT16,
    "half": DTYPE.FLOAT16,
    "fp16": DTYPE.FLOAT16,
    # Float8
    "float8": DTYPE.FLOAT8,
    "fp8": DTYPE.FLOAT8,
    # BFloat16
    "bfloat16": DTYPE.BFLOAT16,
    "bf16": DTYPE.BFLOAT16,
}


def get_dtype(dtype_str: str) -> DTYPE:
    """从字符串获取DTYPE枚举

    Args:
        dtype_str: 数据类型字符串，如"float16", "bfloat16"等

    Returns:
        DTYPE: 对应的数据类型枚举

    Raises:
        ValueError: 如果数据类型未知
    """
    if dtype_str.lower() not in _DTYPE_REGISTRY:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return _DTYPE_REGISTRY[dtype_str.lower()]


def dtype_to_str(dtype: DTYPE) -> str:
    """将DTYPE枚举转换为字符串

    Args:
        dtype: DTYPE枚举值

    Returns:
        str: 数据类型字符串

    Raises:
        ValueError: 如果数据类型未知
    """
    if dtype == DTYPE.FLOAT32:
        return "float32"
    elif dtype == DTYPE.FLOAT16:
        return "float16"
    elif dtype == DTYPE.FLOAT8:
        return "float8"
    elif dtype == DTYPE.BFLOAT16:
        return "bfloat16"
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


def get_dtype_size(dtype_str: str) -> int:
    """获取数据类型的字节大小

    Args:
        dtype_str: 数据类型字符串

    Returns:
        int: 字节大小
    """
    dtype = get_dtype(dtype_str)
    return dtype.size