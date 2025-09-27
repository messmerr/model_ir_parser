# Model IR Parser

一个独立的模块，用于将九歌（Jiuge/FM9G）模型的配置文件解析为与APEX兼容的Transformer IR。

## 项目结构

```
model_ir_parser/
├── __init__.py
├── ir/                    # IR数据结构
│   ├── __init__.py
│   ├── transformer.py    # Transformer顶层
│   ├── block.py         # Block定义
│   ├── cell.py          # Cell定义
│   └── task.py          # Task定义
├── parser.py            # 九歌解析器
├── utils/               # 工具函数
│   └── dtype.py        # 数据类型
└── examples/            # 使用示例
    └── parse_jiuge.py
```

## 安装

```bash
pip install -e .
```

## 使用方法

```python
from model_ir_parser import JiugeParser

# 解析配置文件
parser = JiugeParser()
ir = parser.parse_file("config.json")

# 访问IR信息
print(f"模型类型: {ir.metadata.model_type}")
print(f"层数: {ir.num_decoder_blocks}")
print(f"隐藏层维度: {ir.hidden_size}")
```

## 依赖

- Python >= 3.8
- dataclasses
- typing
- json
- enum

## 设计文档

详见 [DESIGN.md](DESIGN.md)